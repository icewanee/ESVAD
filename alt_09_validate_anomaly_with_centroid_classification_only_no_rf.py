import os
import json
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torch.nn import Module
from torch_geometric.nn import GATConv, global_max_pool
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report
import joblib
import pickle
import random
from datetime import datetime
from scipy.spatial.distance import cosine
from config import (
    TUNE_THRESHOLD_OUTPUT_JSON,
    VALIDATE_ANOMALY_BATCH_SIZE,
    TOOLKIT_PATH,
    SPECIALIST_MODELS_OUTPUT_PATH,
    UNSEEN_CWE_LIST,
    KNOWN_CWE_LIST,
    RESULTS_PATH,
    METRIC_LEARNING_MODEL_OUTPUT_PATH,
    FINAL_GLOBAL_DIR,
)

TEST_KNOWN_DATA_PATH = f"{RESULTS_PATH}/test_known_data.json"
TEST_UNSEEN_DATA_PATH = f"{RESULTS_PATH}/test_unseen_data.json"
TRAIN_DATA_PATH = f"{RESULTS_PATH}/train_data.json"

BEST_MODEL_PATH = f"{METRIC_LEARNING_MODEL_OUTPUT_PATH}/best_hard_mining_model.pth"
VOCAB_DIR = FINAL_GLOBAL_DIR
GAT_MODELS_DIR = SPECIALIST_MODELS_OUTPUT_PATH
TOOLKIT_PATH = TOOLKIT_PATH
FINAL_REPORT_PATH = f"{RESULTS_PATH}/final_validation_report_anomaly_only.txt"
VISUALIZATION_DATA_PATH = f"{RESULTS_PATH}/final_validation_data_anomaly_only.json"
BATCH_SIZE = VALIDATE_ANOMALY_BATCH_SIZE

with open(TUNE_THRESHOLD_OUTPUT_JSON, encoding="utf-8") as f:
    OPTIMAL_THRESHOLD = json.load(f)["threshold"]

GAT_HIDDEN_CHANNELS = 128
NUM_SCALAR_FEATURES = 11
PROJECTION_HEAD_OUTPUT_DIM = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 55


class GlobalFeatureGAT(Module):
    def __init__(
        self,
        vocabs,
        scalar_feature_count,
        hidden_channels=128,
        num_heads=4,
        embedding_dim=32,
    ):
        super(GlobalFeatureGAT, self).__init__()
        torch.manual_seed(RANDOM_STATE)
        self.opcode_embedding = nn.Embedding(
            len(vocabs["opcodes"]), embedding_dim, padding_idx=0
        )
        self.source_embedding = nn.Embedding(
            len(vocabs["sources"]), embedding_dim, padding_idx=0
        )
        self.sink_embedding = nn.Embedding(
            len(vocabs["sinks"]), embedding_dim, padding_idx=0
        )
        self.string_manip_embedding = nn.Embedding(
            len(vocabs["string_manipulation"]), embedding_dim, padding_idx=0
        )
        self.payload_embedding = nn.Embedding(
            len(vocabs["payloads"]), embedding_dim, padding_idx=0
        )
        total_input_dim = scalar_feature_count + (embedding_dim * 5)
        self.norm = nn.LayerNorm(total_input_dim)
        self.conv1 = GATConv(
            total_input_dim, hidden_channels, heads=num_heads, dropout=0.6
        )
        self.conv2 = GATConv(
            hidden_channels * num_heads, hidden_channels, heads=1, dropout=0.6
        )
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_channels // 2, 2),
        )

    def _get_embedding_feature(self, embedding_layer, data_indices):
        embeds = embedding_layer(data_indices)
        mask = (data_indices != 0).unsqueeze(-1).float()
        return (embeds * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)

    def get_graph_embedding(self, data):
        opcode_feature = self._get_embedding_feature(
            self.opcode_embedding, data.x_opcode
        )
        source_feature = self._get_embedding_feature(
            self.source_embedding, data.x_source
        )
        sink_feature = self._get_embedding_feature(self.sink_embedding, data.x_sink)
        string_manip_feature = self._get_embedding_feature(
            self.string_manip_embedding, data.x_string_manip
        )
        payload_feature = self._get_embedding_feature(
            self.payload_embedding, data.x_payload
        )
        final_node_features = torch.cat(
            [
                data.x_scalar,
                opcode_feature,
                source_feature,
                sink_feature,
                string_manip_feature,
                payload_feature,
            ],
            dim=1,
        )
        final_node_features = self.norm(final_node_features)
        x = F.elu(self.conv1(final_node_features, data.edge_index))
        x = F.elu(self.conv2(x, data.edge_index))
        return global_max_pool(x, data.batch)


class ProjectionHead_V2(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim=512):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.GELU(),
            nn.Dropout(p=0.6),
            nn.Linear(hidden_dim_1, output_dim),
        )

    def forward(self, x):
        output = self.model(x)
        return F.normalize(output, p=2, dim=-1)


class MetricLearningModel(nn.Module):
    def __init__(self, gat_models_dict, projection_head):
        super().__init__()
        self.gat_models = nn.ModuleDict(gat_models_dict)
        self.projection_head = projection_head

    def forward(self, data_object):
        expert_vectors = []
        for cwe_name, model in self.gat_models.items():
            expert_vectors.append(model.get_graph_embedding(data_object))
        master_vector = torch.cat(expert_vectors, dim=1)
        return self.projection_head(master_vector)


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        data = torch.load(sample_info["path"], weights_only=False, map_location="cpu")
        return data, sample_info["label"]


def run_inference_pipeline_with_threshold(
    model, test_samples, classifier, threshold, device, class_centroids
):
    model.eval()
    known_results = []
    detected_anomalies = []

    if not test_samples:
        return known_results, detected_anomalies

    dataset = InferenceDataset(test_samples)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)

    with torch.no_grad():
        for data_batch, labels_batch in tqdm(
            loader, desc="  Running Inference Pipeline"
        ):
            embeddings = model(data_batch.to(DEVICE)).cpu().numpy()

            for i in range(len(labels_batch)):
                true_label = labels_batch[i]
                embedding = embeddings[i]

                # Calculate distances to all centroids
                distances = {
                    label: cosine(embedding, centroid)
                    for label, centroid in class_centroids.items()
                }

                if not distances:
                    min_distance = float("inf")
                    closest_label = "N/A"
                else:
                    closest_label = min(distances, key=distances.get)
                    min_distance = distances[closest_label]

                if min_distance <= threshold:
                    predicted_label = closest_label
                    known_results.append(
                        {"true_label": true_label, "predicted_label": predicted_label}
                    )
                else:
                    detected_anomalies.append(
                        {
                            "true_label": true_label,
                            "closest_known": closest_label,
                            "distance": min_distance,
                        }
                    )

    return known_results, detected_anomalies


def run_validation(
    test_description, test_samples, model, classifier, threshold, class_centroids
):
    report_lines = []
    report_lines.append("\n\n" + "=" * 80)
    report_lines.append(f"--- RUNNING VALIDATION FOR: {test_description} ---")
    report_lines.append(f"--- Total samples: {len(test_samples)} ---")
    report_lines.append(f"--- Anomaly Threshold (Cosine Distance): {threshold:.3f} ---")
    report_lines.append("=" * 80)

    serializable_anomalies = []

    if not test_samples:
        report_lines.append("No samples found for this validation run. Skipping.")
        run_data = {
            "test_description": test_description,
            "total_samples": 0,
            "known_results": [],
            "detected_anomalies": [],
        }
        return "\n".join(report_lines), run_data

    known_results, detected_anomalies = run_inference_pipeline_with_threshold(
        model, test_samples, classifier, threshold, DEVICE, class_centroids
    )

    num_classified_as_known = len(known_results)
    report_lines.append(
        f"\n--- Samples Classified as KNOWN: {num_classified_as_known} / {len(test_samples)} ({num_classified_as_known/len(test_samples):.2%}) ---"
    )
    if known_results:
        known_true_labels = [res["true_label"] for res in known_results]
        known_pred_labels = [res["predicted_label"] for res in known_results]
        all_known_labels = sorted(list(class_centroids.keys()))
        report_lines.append(
            classification_report(
                known_true_labels,
                known_pred_labels,
                labels=all_known_labels,
                zero_division=0,
            )
        )
    else:
        report_lines.append("No samples were classified as known.")

    num_anomalies = len(detected_anomalies)
    report_lines.append(
        f"\n--- Samples Flagged as ANOMALIES: {num_anomalies} / {len(test_samples)} ({num_anomalies/len(test_samples):.2%}) ---"
    )

    if detected_anomalies:
        for item in detected_anomalies:
            serializable_anomalies.append(
                {
                    "true_label": item["true_label"],
                    "closest_known": item["closest_known"],
                    "distance": float(item["distance"]),
                }
            )
        report_lines.append(
            f"  -> Detected {num_anomalies} anomalies. (Clustering disabled)"
        )
    else:
        report_lines.append("No samples were flagged as anomalies.")

    report_lines.append(f"\n--- FINISHED VALIDATION FOR: {test_description} ---")

    run_data = {
        "test_description": test_description,
        "total_samples": len(test_samples),
        "known_results": known_results,
        "detected_anomalies": serializable_anomalies,
    }

    return "\n".join(report_lines), run_data


if __name__ == "__main__":
    all_run_data = {}

    TRAINED_CWE_LIST = sorted(KNOWN_CWE_LIST)
    print("Configuration loaded:")
    print(f"  - Trained CWEs: {TRAINED_CWE_LIST}")
    print(f"  - Unseen Test CWEs: {UNSEEN_CWE_LIST}")
    print(f"  - Optimal Threshold: {OPTIMAL_THRESHOLD}")
    print(f"  - Mode: Anomaly Detection Only (No Clustering)")

    def load_samples_from_map(map_path, cwe_whitelist=None):
        if not os.path.exists(map_path):
            return []
        with open(map_path, "r") as f:
            data_map = json.load(f)
        samples = []
        for l, v in data_map.items():
            if cwe_whitelist and l not in cwe_whitelist:
                continue
            samples.extend([{"path": p, "label": l} for p in v.get("BAD", [])])
        return samples

    print("\nLoading data...")
    train_samples = load_samples_from_map(
        TRAIN_DATA_PATH, cwe_whitelist=TRAINED_CWE_LIST
    )
    test_known_samples = load_samples_from_map(
        TEST_KNOWN_DATA_PATH, cwe_whitelist=TRAINED_CWE_LIST
    )
    test_unseen_samples = load_samples_from_map(
        TEST_UNSEEN_DATA_PATH, cwe_whitelist=UNSEEN_CWE_LIST
    )

    if not train_samples:
        raise ValueError("Training data is essential.")

    def load_vocab(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    all_vocabs = {
        "opcodes": load_vocab(os.path.join(VOCAB_DIR, "global_opcode_vocabulary.json")),
        "sources": load_vocab(
            os.path.join(VOCAB_DIR, "global_cwe_source_vocabulary.json")
        ),
        "sinks": load_vocab(os.path.join(VOCAB_DIR, "global_cwe_sink_vocabulary.json")),
        "string_manipulation": load_vocab(
            os.path.join(VOCAB_DIR, "global_cwe_string_manipulation_vocabulary.json")
        ),
        "payloads": load_vocab(
            os.path.join(VOCAB_DIR, "global_cwe_payload_vocabulary.json")
        ),
    }

    print(f"\nLoading GAT experts for: {TRAINED_CWE_LIST}")
    gat_models_dict = {}
    for cwe_name in TRAINED_CWE_LIST:
        model_path = os.path.join(GAT_MODELS_DIR, f"best_model_{cwe_name}.pth")
        if not os.path.exists(model_path):
            continue
        gat_model = GlobalFeatureGAT(
            vocabs=all_vocabs, scalar_feature_count=NUM_SCALAR_FEATURES
        )
        gat_model.load_state_dict(
            torch.load(model_path, map_location=DEVICE), strict=False
        )
        gat_models_dict[cwe_name] = gat_model

    print(f"Loading best metric learning model from: {BEST_MODEL_PATH}")
    master_vector_dim = GAT_HIDDEN_CHANNELS * len(gat_models_dict)
    ph_hidden_1 = master_vector_dim * 2
    projection_head = ProjectionHead_V2(
        input_dim=master_vector_dim,
        hidden_dim_1=ph_hidden_1,
        hidden_dim_2=0,
        output_dim=PROJECTION_HEAD_OUTPUT_DIM,
    )
    model = MetricLearningModel(
        gat_models_dict=gat_models_dict, projection_head=projection_head
    )

    try:
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    except RuntimeError as e:
        print("\nError loading state_dict.")
        raise e

    model.to(DEVICE)
    model.eval()

    classifier_path = os.path.join(TOOLKIT_PATH, "rf_classifier.joblib")
    centroids_path = os.path.join(TOOLKIT_PATH, "centroids.pkl")
    classifier = joblib.load(classifier_path)
    with open(centroids_path, "rb") as f:
        class_centroids = pickle.load(f)
    print("Classifier and Centroids loaded successfully.")

    print(f"\n--- Starting Final Validation Runs (Anomaly Detection Only) ---")

    with open(FINAL_REPORT_PATH, "w", encoding="utf-8") as report_file:
        report_file.write(f"Final Validation Report (Anomaly Detection Only)\n")
        report_file.write(
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        report_file.write(f"Optimal Anomaly Threshold: {OPTIMAL_THRESHOLD:.4f}\n")

        report1, data1 = run_validation(
            "Sanity Check",
            train_samples,
            model,
            classifier,
            OPTIMAL_THRESHOLD,
            class_centroids,
        )
        print(report1)
        report_file.write(report1)
        all_run_data["sanity_check"] = data1

        report2, data2 = run_validation(
            "Validation Set",
            test_known_samples,
            model,
            classifier,
            OPTIMAL_THRESHOLD,
            class_centroids,
        )
        print(report2)
        report_file.write(report2)
        all_run_data["validation_known"] = data2

        report3, data3 = run_validation(
            "Test Set (Unseen)",
            test_unseen_samples,
            model,
            classifier,
            OPTIMAL_THRESHOLD,
            class_centroids,
        )
        print(report3)
        report_file.write(report3)
        all_run_data["test_unseen"] = data3

    with open(VISUALIZATION_DATA_PATH, "w", encoding="utf-8") as f_json:
        json.dump(all_run_data, f_json, indent=2)
    print(f"\n--- Complete. Data saved to {VISUALIZATION_DATA_PATH} ---")
