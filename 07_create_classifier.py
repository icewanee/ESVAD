import os
import json
import torch
import torch.nn as nn
import joblib
import pickle
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_max_pool
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial.distance import cosine
from config import (
    RESULTS_PATH,
    METRIC_LEARNING_MODEL_OUTPUT_PATH,
    FINAL_GLOBAL_DIR,
    SPECIALIST_MODELS_OUTPUT_PATH,
    KNOWN_CWE_LIST,
    UNSEEN_CWE_LIST,
    TUNE_THRESHOLD_BATCH_SIZE,
    TOOLKIT_PATH,
    CREATE_CLASSIFIER_BATCH_SIZE,
)


BEST_MODEL_PATH = f"{METRIC_LEARNING_MODEL_OUTPUT_PATH}/best_hard_mining_model.pth"
TRAIN_DATA_MAP_PATH = f"{RESULTS_PATH}/train_data.json"
VOCAB_DIR = FINAL_GLOBAL_DIR
GAT_MODELS_DIR = SPECIALIST_MODELS_OUTPUT_PATH
TOOLKIT_OUTPUT_DIR = TOOLKIT_PATH
BATCH_SIZE = CREATE_CLASSIFIER_BATCH_SIZE


GAT_HIDDEN_CHANNELS = 128
NUM_SCALAR_FEATURES = 11
PROJECTION_HEAD_OUTPUT_DIM = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RANDOM_STATE = 55


class GlobalFeatureGAT(nn.Module):
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
    def __init__(
        self, input_dim, hidden_dim_1, hidden_dim_2, output_dim=512
    ):  # hidden_dim_2 ไม่ถูกใช้
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


def main():
    print("--- Creating New Classifier and Centroids from the Best Model ---")
    os.makedirs(TOOLKIT_OUTPUT_DIR, exist_ok=True)

    TRAINED_CWE_LIST = sorted(KNOWN_CWE_LIST)
    print(f"Building tools based on trained CWEs: {TRAINED_CWE_LIST}")

    all_vocabs = {
        "opcodes": json.load(
            open(os.path.join(VOCAB_DIR, "global_opcode_vocabulary.json"))
        ),
        "sources": json.load(
            open(os.path.join(VOCAB_DIR, "global_cwe_source_vocabulary.json"))
        ),
        "sinks": json.load(
            open(os.path.join(VOCAB_DIR, "global_cwe_sink_vocabulary.json"))
        ),
        "string_manipulation": json.load(
            open(
                os.path.join(
                    VOCAB_DIR, "global_cwe_string_manipulation_vocabulary.json"
                )
            )
        ),
        "payloads": json.load(
            open(os.path.join(VOCAB_DIR, "global_cwe_payload_vocabulary.json"))
        ),
    }

    print("Loading pre-trained GAT experts...")
    gat_models_dict = {}
    for cwe_name in TRAINED_CWE_LIST:
        model_path = os.path.join(GAT_MODELS_DIR, f"best_model_{cwe_name}.pth")
        if not os.path.exists(model_path):
            print(
                f"Warning: GAT model not found for {cwe_name} at {model_path}. Skipping."
            )
            continue
        model_instance = GlobalFeatureGAT(
            vocabs=all_vocabs, scalar_feature_count=NUM_SCALAR_FEATURES
        )
        try:
            model_instance.load_state_dict(
                torch.load(model_path, map_location=DEVICE), strict=False
            )
            gat_models_dict[cwe_name] = model_instance.to(DEVICE)
        except Exception as e:
            print(f"Error loading GAT model for {cwe_name}: {e}")

    if not gat_models_dict:
        raise ValueError("No GAT models were loaded successfully. Cannot proceed.")
    print(f"Successfully loaded {len(gat_models_dict)} GAT experts.")

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
        print(
            "\nError loading state_dict. Model architecture might mismatch the saved file."
        )
        print("Expected input dim for ProjectionHead:", master_vector_dim)
        print(
            "Saved model state_dict keys:",
            torch.load(BEST_MODEL_PATH, map_location="cpu").keys(),
        )
        raise e

    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully.")

    print(f"Loading original training data from: {TRAIN_DATA_MAP_PATH}")
    with open(TRAIN_DATA_MAP_PATH, "r") as f:
        train_data_map = json.load(f)
    train_samples = [
        {"path": p, "label": l}
        for l, v in train_data_map.items()
        if l in TRAINED_CWE_LIST
        for p in v.get("BAD", [])
    ]

    if not train_samples:
        raise ValueError(
            f"No training samples found for the specified CWEs {TRAINED_CWE_LIST}. Check TRAIN_DATA_MAP_PATH."
        )
    print(f"Found {len(train_samples)} training samples to build tools.")

    train_dataset = InferenceDataset(train_samples)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    all_embeddings, all_labels = [], []
    with torch.no_grad():
        for data_batch, labels_batch in tqdm(
            train_loader, desc="Generating Embeddings"
        ):
            embeddings = model(data_batch.to(DEVICE)).cpu().numpy()
            all_embeddings.append(embeddings)
            all_labels.extend(labels_batch)
    all_embeddings = np.concatenate(all_embeddings)

    print("\nTraining RandomForest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"
    )
    rf_model.fit(all_embeddings, all_labels)
    print("Training complete.")

    print("\nCalculating and saving ROBUST centroids (Trimmed Mean)...")
    TRIM_PERCENTILE = 65.0

    embeddings_by_class = {label: [] for label in TRAINED_CWE_LIST}
    for i, label in enumerate(all_labels):
        if label in embeddings_by_class:
            embeddings_by_class[label].append(all_embeddings[i])

    robust_centroids = {}
    for label, embeds in tqdm(
        embeddings_by_class.items(), desc="Calculating Robust Centroids"
    ):
        if not embeds:
            print(f"Warning: No embeddings found for class {label}. Skipping centroid.")
            continue

        embeds_array = np.array(embeds)

        initial_centroid = np.mean(embeds_array, axis=0)

        distances = np.array([cosine(e, initial_centroid) for e in embeds_array])
        distance_threshold = np.percentile(distances, TRIM_PERCENTILE)

        core_embeddings_mask = distances <= distance_threshold
        core_embeddings = embeds_array[core_embeddings_mask]

        if len(core_embeddings) == 0:
            print(
                f"  - Warning: Trimming for {label} left no samples. Using initial centroid."
            )
            robust_centroids[label] = initial_centroid
        else:
            robust_centroids[label] = np.mean(core_embeddings, axis=0)

    centroids = robust_centroids

    if len(centroids) != len(TRAINED_CWE_LIST):
        print(
            "Warning: Some CWEs had no embeddings generated. Centroids might be incomplete."
        )
        print("CWEs with centroids:", list(centroids.keys()))

    centroids_path = os.path.join(TOOLKIT_OUTPUT_DIR, "centroids.pkl")
    with open(centroids_path, "wb") as f:
        pickle.dump(centroids, f)
    print(f"Robust Centroids saved to: {centroids_path}")

    classifier_path = os.path.join(TOOLKIT_OUTPUT_DIR, "rf_classifier.joblib")
    joblib.dump(rf_model, classifier_path)
    print(f"New RandomForest classifier saved to: {classifier_path}")


if __name__ == "__main__":
    main()
