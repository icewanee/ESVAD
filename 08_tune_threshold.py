import os
import json
import torch
import math
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_max_pool
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import pickle
import random
from scipy.spatial.distance import cosine
import pandas as pd

from config import (
    RESULTS_PATH,
    METRIC_LEARNING_MODEL_OUTPUT_PATH,
    FINAL_GLOBAL_DIR,
    SPECIALIST_MODELS_OUTPUT_PATH,
    KNOWN_CWE_LIST,
    UNSEEN_CWE_LIST,
    TUNE_THRESHOLD_BATCH_SIZE,
    TOOLKIT_PATH,
    TUNE_THRESHOLD_OUTPUT_JSON,
)

random.seed(55)
np.random.seed(55)
torch.manual_seed(55)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = TUNE_THRESHOLD_BATCH_SIZE
BEST_MODEL_PATH = f"{METRIC_LEARNING_MODEL_OUTPUT_PATH}/best_hard_mining_model.pth"


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

    def _get_embedding_feature(self, embedding_layer, data_indices):
        embeds = embedding_layer(data_indices)
        mask = (data_indices != 0).unsqueeze(-1).float()
        return (embeds * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)

    def get_graph_embedding(self, data):
        final_node_features = torch.cat(
            [
                data.x_scalar,
                self._get_embedding_feature(self.opcode_embedding, data.x_opcode),
                self._get_embedding_feature(self.source_embedding, data.x_source),
                self._get_embedding_feature(self.sink_embedding, data.x_sink),
                self._get_embedding_feature(
                    self.string_manip_embedding, data.x_string_manip
                ),
                self._get_embedding_feature(self.payload_embedding, data.x_payload),
            ],
            dim=1,
        )
        x = F.elu(self.conv1(self.norm(final_node_features), data.edge_index))
        x = F.elu(self.conv2(x, data.edge_index))
        return global_max_pool(x, data.batch)


class ProjectionHead_V2(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, output_dim=512):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.GELU(),
            nn.Dropout(p=0.6),
            nn.Linear(hidden_dim_1, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.model(x), p=2, dim=-1)


class MetricLearningModel(nn.Module):
    def __init__(self, gat_models_dict, projection_head, order):
        super().__init__()
        self.gat_models = nn.ModuleDict(gat_models_dict)
        self.projection_head = projection_head
        self.order = order

    def forward(self, data):
        expert_vectors = [
            self.gat_models[cwe].get_graph_embedding(data) for cwe in self.order
        ]
        return self.projection_head(torch.cat(expert_vectors, dim=1))


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (
            torch.load(
                self.samples[idx]["path"], weights_only=False, map_location="cpu"
            ),
            0,
        )


def load_bad_samples(map_path, cwe_whitelist):
    if not os.path.exists(map_path):
        return []
    with open(map_path, "r") as f:
        data_map = json.load(f)
    samples = []
    for l, v in data_map.items():
        if l in cwe_whitelist:
            samples.extend([{"path": p, "label": l} for p in v.get("BAD", [])])
    return samples


def get_min_distances(model, samples, centroids, device):
    if not samples:
        return []
    model.eval()
    loader = DataLoader(InferenceDataset(samples), batch_size=BATCH_SIZE)
    all_dists = []
    c_vals = list(centroids.values())
    with torch.no_grad():
        for data_batch, _ in tqdm(loader, desc="Extracting Distances", leave=False):
            embs = model(data_batch.to(device)).cpu().numpy()
            for e in embs:
                all_dists.append(min([cosine(e, c) for c in c_vals]))
    return all_dists


def main_tuner():
    vocabs = {
        k: json.load(
            open(os.path.join(FINAL_GLOBAL_DIR, f"global_{v}_vocabulary.json"))
        )
        for k, v in {
            "opcodes": "opcode",
            "sources": "cwe_source",
            "sinks": "cwe_sink",
            "string_manipulation": "cwe_string_manipulation",
            "payloads": "cwe_payload",
        }.items()
    }
    with open(os.path.join(TOOLKIT_PATH, "centroids.pkl"), "rb") as f:
        centroids = pickle.load(f)

    sorted_knowns = sorted(KNOWN_CWE_LIST)
    gat_models = {}
    for cwe in sorted_knowns:
        m = GlobalFeatureGAT(vocabs, 11).to(DEVICE)
        m.load_state_dict(
            torch.load(
                os.path.join(SPECIALIST_MODELS_OUTPUT_PATH, f"best_model_{cwe}.pth"),
                map_location=DEVICE,
            ),
            strict=False,
        )
        gat_models[cwe] = m

    master_dim = 128 * len(sorted_knowns)
    ph = ProjectionHead_V2(master_dim, master_dim * 2, 512)
    model = MetricLearningModel(gat_models, ph, sorted_knowns)
    model.load_state_dict(
        torch.load(BEST_MODEL_PATH, map_location=DEVICE),
        strict=False,
    )
    model.to(DEVICE).eval()

    print("\n[1/2] Pre-calculating distances for BAD samples...")
    dist_tr_k = get_min_distances(
        model,
        load_bad_samples(f"{RESULTS_PATH}/train_data.json", KNOWN_CWE_LIST),
        centroids,
        DEVICE,
    )
    dist_tr_u = get_min_distances(
        model,
        load_bad_samples(f"{RESULTS_PATH}/train_unseen_data.json", UNSEEN_CWE_LIST),
        centroids,
        DEVICE,
    )
    dist_val_k = get_min_distances(
        model,
        load_bad_samples(f"{RESULTS_PATH}/test_known_data.json", KNOWN_CWE_LIST),
        centroids,
        DEVICE,
    )
    dist_ts_u = get_min_distances(
        model,
        load_bad_samples(f"{RESULTS_PATH}/test_unseen_data.json", UNSEEN_CWE_LIST),
        centroids,
        DEVICE,
    )

    results = []
    best_score, best_t = -1.0, 0.0
    for t in np.arange(0.01, 0.405, 0.005):
        acc_tr_k = np.mean(np.array(dist_tr_k) <= t) if dist_tr_k else 0
        acc_tr_u = np.mean(np.array(dist_tr_u) > t) if dist_tr_u else 0
        acc_val_k = np.mean(np.array(dist_val_k) <= t) if dist_val_k else 0
        acc_ts_u = np.mean(np.array(dist_ts_u) > t) if dist_ts_u else 0

        score = math.sqrt(acc_tr_k * acc_tr_u)

        results.append(
            {
                "Threshold": f"{t:.3f}",
                "Tr_Sum": f"{score:.4f}",
                "Tr_K_Acc": f"{acc_tr_k:.4f}",
                "Tr_U_Acc": f"{acc_tr_u:.4f}",
                "Val_K_Acc": f"{acc_val_k:.4f}",
                "Ts_U_Acc": f"{acc_ts_u:.4f}",
            }
        )
        if score > best_score:
            best_score, best_t = score, t

    print("\n" + pd.DataFrame(results).to_string(index=False))
    print(f"\n🏆 Optimal Threshold (Train-based): {best_t:.3f}")
    with open(TUNE_THRESHOLD_OUTPUT_JSON, "w") as f:
        json.dump({"threshold": float(best_t), "score": float(best_score)}, f, indent=4)


if __name__ == "__main__":
    main_tuner()
