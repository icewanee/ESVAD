import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torch.nn import Module
from torch_geometric.nn import GATConv, global_max_pool
import torch.nn.functional as F
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.miners import BatchHardMiner
from pytorch_metric_learning.samplers import MPerClassSampler
from datetime import datetime
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast, GradScaler
import gc

from config import (
    FINAL_GLOBAL_DIR,
    RESULTS_PATH,
    SPECIALIST_MODELS_OUTPUT_PATH,
    METRIC_LEARNING_MODEL_OUTPUT_PATH,
    KNOWN_CWE_LIST,
    METRIC_LEARNING_MODEL_TRAINING_M_SAMPLES,
    METRIC_LEARNING_MODEL_TRAINING_BATCH_SIZE,
    METRIC_LEARNING_MODEL_TRAINING_EPOCH,
    METRIC_LEARNING_MODEL_TRAINING_LEARNING_RATE_GAT,
    METRIC_LEARNING_MODEL_TRAINING_LEARNING_RATE_HEAD,
    METRIC_LEARNING_MODEL_TRAINING_PATIENCE,
)

VOCAB_DIR = FINAL_GLOBAL_DIR
TRAIN_DATA_MAP_PATH = f"{RESULTS_PATH}/train_data.json"
VAL_DATA_MAP_PATH = f"{RESULTS_PATH}/test_known_data.json"

GAT_MODELS_DIR = SPECIALIST_MODELS_OUTPUT_PATH
MODEL_OUTPUT_DIR = METRIC_LEARNING_MODEL_OUTPUT_PATH
TRAINING_LOG_PATH = f"{RESULTS_PATH}/metric_learning_training_log.txt"

EPOCHS = METRIC_LEARNING_MODEL_TRAINING_EPOCH
LEARNING_RATE_HEAD = METRIC_LEARNING_MODEL_TRAINING_LEARNING_RATE_HEAD
LEARNING_RATE_GAT = METRIC_LEARNING_MODEL_TRAINING_LEARNING_RATE_GAT
BATCH_SIZE = METRIC_LEARNING_MODEL_TRAINING_BATCH_SIZE
M_SAMPLES = METRIC_LEARNING_MODEL_TRAINING_M_SAMPLES
PATIENCE = METRIC_LEARNING_MODEL_TRAINING_PATIENCE

VALIDATION_INTERVAL = 3
GAT_HIDDEN_CHANNELS = 128
PROJECTION_HEAD_OUTPUT_DIM = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
RANDOM_STATE = 55


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, samples, label_map, max_nodes=90000):
        self.label_map = label_map
        self.valid_samples = []

        print(f"Filtering dataset (Max nodes: {max_nodes})...")
        skipped = 0
        for s in tqdm(samples, desc="Filtering"):
            try:
                data = torch.load(s["path"], weights_only=False, map_location="cpu")

                if hasattr(data, "x_scalar"):
                    num = data.x_scalar.shape[0]
                elif hasattr(data, "num_nodes"):
                    num = data.num_nodes
                else:
                    num = 0

                if num <= max_nodes:
                    self.valid_samples.append(s)
                else:
                    skipped += 1
            except:
                skipped += 1

        print(
            f"Dataset Ready. Kept {len(self.valid_samples)}, Skipped {skipped} (Too Large/Error)."
        )

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        sample_info = self.valid_samples[idx]
        data = torch.load(sample_info["path"], weights_only=False, map_location="cpu")
        label_int = self.label_map[sample_info["label"]]
        data.y = torch.tensor(label_int, dtype=torch.long)
        return data


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

    def run_conv1(self, x, edge_index):
        return F.elu(self.conv1(x, edge_index))

    def run_conv2(self, x, edge_index):
        return F.elu(self.conv2(x, edge_index))

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

        if self.training:
            x = checkpoint(
                self.run_conv1,
                final_node_features,
                data.edge_index,
                use_reentrant=False,
            )
            x = checkpoint(self.run_conv2, x, data.edge_index, use_reentrant=False)
        else:
            x = self.run_conv1(final_node_features, data.edge_index)
            x = self.run_conv2(x, data.edge_index)

        return global_max_pool(x, data.batch)

    def forward(self, data):
        return self.get_graph_embedding(data)


class ProjectionHead_V2(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim=512):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.GELU(),
            nn.Dropout(p=0.5),
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


def train_epoch(model, loader, optimizer, loss_fn, miner, device, log_file):
    model.train()
    total_loss = 0
    scaler = GradScaler()

    progress_bar = tqdm(loader, desc="Training Epoch", leave=False)

    for data_batch in progress_bar:
        data_batch = data_batch.to(device)
        labels_batch_int = data_batch.y

        optimizer.zero_grad()

        with autocast():
            embeddings = model(data_batch)
            loss = loss_fn(embeddings, labels_batch_int)

            if torch.isnan(loss):
                print("NaN detected")
                continue

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        del embeddings, loss, data_batch

    return total_loss / len(loader) if loader else 0


def validate_with_knn(
    model,
    train_loader_for_val,
    val_loader,
    device,
    int_to_label_map,
    log_file,
    k_neighbors=5,
):
    model.eval()

    train_embeddings, train_labels_int = [], []
    with torch.no_grad():
        for data in tqdm(
            train_loader_for_val, desc="  Gen Train Embeds (for KNN)", leave=False
        ):
            embeddings = model(data.to(device)).cpu().numpy()
            labels_int = data.y.cpu().numpy()
            train_embeddings.append(embeddings)
            train_labels_int.extend(labels_int)
    train_embeddings = np.concatenate(train_embeddings)

    val_embeddings, val_labels_int = [], []
    with torch.no_grad():
        for data in tqdm(val_loader, desc="  Gen Val Embeds (for KNN)", leave=False):
            embeddings = model(data.to(device)).cpu().numpy()
            labels_int = data.y.cpu().numpy()
            val_embeddings.append(embeddings)
            val_labels_int.extend(labels_int)
    val_embeddings = np.concatenate(val_embeddings)

    print(f"  -> Fitting KNN (k={k_neighbors})...")
    log_file.write(f"  -> Fitting KNN (k={k_neighbors})...\n")

    knn = KNeighborsClassifier(n_neighbors=k_neighbors, metric="cosine", n_jobs=-1)
    knn.fit(train_embeddings, train_labels_int)

    val_preds_int = knn.predict(val_embeddings)

    val_labels_str = [int_to_label_map[i] for i in val_labels_int]
    val_preds_str = [int_to_label_map[i] for i in val_preds_int]

    report_header = "\n--- Validation Results (k-NN) ---"
    print(report_header)
    log_file.write(report_header + "\n")

    report_str = classification_report(val_labels_str, val_preds_str, zero_division=0)
    print(report_str)
    log_file.write(report_str + "\n")

    report_dict = classification_report(
        val_labels_str, val_preds_str, output_dict=True, zero_division=0
    )
    macro_f1 = report_dict["macro avg"]["f1-score"]

    summary_str = (
        f"  -> Accuracy: {report_dict['accuracy']:.4f} | Macro F1: {macro_f1:.4f}"
    )
    print(summary_str)
    log_file.write(summary_str + "\n")

    return macro_f1


def main():
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    with open(TRAINING_LOG_PATH, "w", encoding="utf-8") as log_file:
        log_file.write(f"--- Training Log ---\n")
        log_file.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Model Output Dir: {MODEL_OUTPUT_DIR}\n")

        with open(TRAIN_DATA_MAP_PATH, "r") as f:
            train_data_map = json.load(f)
        with open(VAL_DATA_MAP_PATH, "r") as f:
            val_data_map = json.load(f)

        CWE_LIST = KNOWN_CWE_LIST
        header_str = f"--- Running with Selective Ensemble: {CWE_LIST} ---"
        print(header_str)
        log_file.write(header_str + "\n")

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

        label_to_int = {label: i for i, label in enumerate(CWE_LIST)}
        int_to_label = {i: label for label, i in label_to_int.items()}

        print(f"Training with {len(CWE_LIST)} selected CWEs.")
        log_file.write(f"Training with {len(CWE_LIST)} selected CWEs.\n")

        label_map_str = f"Label mapping: {label_to_int}"
        print(label_map_str)
        log_file.write(label_map_str + "\n")

        gat_models_dict = {}
        for cwe_name in CWE_LIST:
            model_path = os.path.join(GAT_MODELS_DIR, f"best_model_{cwe_name}.pth")
            model_instance = GlobalFeatureGAT(
                vocabs=all_vocabs, scalar_feature_count=11
            )
            model_instance.load_state_dict(
                torch.load(model_path, map_location=DEVICE), strict=False
            )
            gat_models_dict[cwe_name] = model_instance.to(DEVICE)

        train_samples = [
            {"path": p, "label": l}
            for l, v in train_data_map.items()
            if l in CWE_LIST
            for p in v.get("BAD", [])
        ]
        val_samples = [
            {"path": p, "label": l}
            for l, v in val_data_map.items()
            if l in CWE_LIST
            for p in v.get("BAD", [])
        ]

        train_dataset = GraphDataset(train_samples, label_to_int)
        val_dataset = GraphDataset(val_samples, label_to_int)

        print("Extracting labels for MPerClassSampler...")
        log_file.write("Extracting labels for MPerClassSampler...\n")
        train_labels_for_sampler = [
            label_to_int[s["label"]] for s in train_dataset.valid_samples
        ]

        m_samples_str = f"Using MPerClassSampler with M={M_SAMPLES} samples per class."
        print(m_samples_str)
        log_file.write(m_samples_str + "\n")

        sampler = MPerClassSampler(
            labels=train_labels_for_sampler,
            m=M_SAMPLES,
            batch_size=BATCH_SIZE,
            length_before_new_iter=len(train_dataset),
        )

        train_loader = DataLoader(
            train_dataset,
            num_workers=0,
            drop_last=True,
            sampler=sampler,
            batch_size=BATCH_SIZE,
        )

        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )
        train_loader_for_val = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )

        master_vector_dim = GAT_HIDDEN_CHANNELS * len(CWE_LIST)

        ph_hidden_1 = master_vector_dim * 2
        ph_hidden_2 = 0

        projection_head = ProjectionHead_V2(
            input_dim=master_vector_dim,
            hidden_dim_1=ph_hidden_1,
            hidden_dim_2=ph_hidden_2,
            output_dim=PROJECTION_HEAD_OUTPUT_DIM,
        )

        metric_model = MetricLearningModel(
            gat_models_dict=gat_models_dict, projection_head=projection_head
        ).to(DEVICE)

        base_loss = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5)
        miner = miners.MultiSimilarityMiner(epsilon=0.1)
        loss_function = losses.CrossBatchMemory(
            loss=base_loss,
            embedding_size=PROJECTION_HEAD_OUTPUT_DIM,
            memory_size=1024,
            miner=miner,
        ).to(DEVICE)

        optimizer_params = [
            {"params": metric_model.gat_models.parameters(), "lr": LEARNING_RATE_GAT},
            {
                "params": metric_model.projection_head.parameters(),
                "lr": LEARNING_RATE_HEAD,
            },
        ]
        optimizer = optim.AdamW(optimizer_params, weight_decay=1e-4)

        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

        best_val_metric = -1.0
        epochs_no_improve = 0

        start_header = (
            "\n--- Starting Training with Memory Bank (XBM) & MultiSimilarityLoss ---"
        )
        print(start_header)
        log_file.write(start_header + "\n")

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_epoch(
                metric_model,
                train_loader,
                optimizer,
                loss_function,
                miner,
                DEVICE,
                log_file,
            )

            current_lr_head = optimizer.param_groups[1]["lr"]
            current_lr_gat = optimizer.param_groups[0]["lr"]

            epoch_summary = f"\nEpoch {epoch}/{EPOCHS} | Avg Train Loss: {train_loss:.4f} | LR (Head): {current_lr_head:.1E} | LR (GAT): {current_lr_gat:.1E}"
            print(epoch_summary)
            log_file.write(epoch_summary + "\n")

            if epoch % VALIDATION_INTERVAL == 0:
                val_metric = validate_with_knn(
                    metric_model,
                    train_loader_for_val,
                    val_loader,
                    DEVICE,
                    int_to_label,
                    log_file,
                    k_neighbors=11,
                )
                scheduler.step(val_metric)

                if val_metric > best_val_metric:
                    save_msg = (
                        f"  -> New best Macro F1 ({val_metric:.4f})! Saving model..."
                    )
                    print(save_msg)
                    log_file.write(save_msg + "\n")

                    best_val_metric = val_metric
                    epochs_no_improve = 0
                    torch.save(
                        metric_model.state_dict(),
                        os.path.join(MODEL_OUTPUT_DIR, "best_hard_mining_model.pth"),
                    )
                else:
                    epochs_no_improve += 1
                    no_improve_msg = f"  -> No improvement for {epochs_no_improve} checks. Best F1: {best_val_metric:.4f}"
                    print(no_improve_msg)
                    log_file.write(no_improve_msg + "\n")

                if epochs_no_improve >= PATIENCE:
                    stop_msg = "\n--- Early stopping triggered. ---"
                    print(stop_msg)
                    log_file.write(stop_msg + "\n")
                    break

        finish_msg = "\n--- Training finished! ---"
        print(finish_msg)
        log_file.write(finish_msg + "\n")
        log_file.write(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    main()
