import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch_geometric.nn import GATConv, global_max_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
import json
from tqdm import tqdm
from config import (
    RESULTS_PATH,
    FINAL_GLOBAL_DIR,
    SPECIALIST_MODELS_OUTPUT_PATH,
    SPECIALIST_MODELS_TRAINING_EPOCH,
    SPECIALIST_MODELS_TRAINING_ACCUMULATION_STEPS,
    SPECIALIST_MODELS_TRAINING_BATCH_SIZE,
    SPECIALIST_MODELS_TRAINING_LEARNING_RATE,
    SPECIALIST_MODELS_TRAINING_PATIENCE,
)

VOCAB_DIR = FINAL_GLOBAL_DIR
MASTER_TRAIN_MAP_PATH = f"{RESULTS_PATH}/train_data.json"
EPOCHS = SPECIALIST_MODELS_TRAINING_EPOCH
BATCH_SIZE = SPECIALIST_MODELS_TRAINING_BATCH_SIZE
ACCUMULATION_STEPS = SPECIALIST_MODELS_TRAINING_ACCUMULATION_STEPS
LEARNING_RATE = SPECIALIST_MODELS_TRAINING_LEARNING_RATE
PATIENCE = SPECIALIST_MODELS_TRAINING_PATIENCE

HIDDEN_CHANNELS = 128
NUM_GAT_HEADS = 4
EMBEDDING_DIM = 32
NUM_SCALAR_FEATURES = 11
RANDOM_STATE = 55
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = torch.load(self.file_paths[idx], weights_only=False)
        data.y = torch.tensor(self.labels[idx], dtype=torch.long)
        return data


class GlobalFeatureGAT(Module):
    def __init__(
        self,
        vocabs,
        scalar_feature_count,
        num_classes,
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
            nn.Linear(hidden_channels // 2, num_classes),
        )

    def _get_embedding_feature(self, embedding_layer, data_indices):
        embeds = embedding_layer(data_indices)
        mask = (data_indices != 0).unsqueeze(-1).float()
        feature = (embeds * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
        return feature

    def forward(self, data):
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
        x_pooled = global_max_pool(x, data.batch)
        return self.classifier_head(x_pooled)


def main():
    os.makedirs(SPECIALIST_MODELS_OUTPUT_PATH, exist_ok=True)

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

    with open(MASTER_TRAIN_MAP_PATH, "r") as f:
        master_train_map = json.load(f)

    cwe_labels = sorted(master_train_map.keys())
    label_to_idx = {cwe: i for i, cwe in enumerate(cwe_labels)}

    all_files, all_labels = [], []
    for cwe_name in cwe_labels:
        bad_files = master_train_map[cwe_name].get("BAD", [])
        all_files.extend(bad_files)
        all_labels.extend([label_to_idx[cwe_name]] * len(bad_files))

    print(
        f"Total BAD graphs: {len(all_files)} across {len(cwe_labels)} CWE categories."
    )

    train_files, val_files, train_y, val_y = train_test_split(
        all_files,
        all_labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=all_labels,
    )

    num_workers = 0 if os.name == "nt" else 4
    train_loader = DataLoader(
        GraphDataset(train_files, train_y),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        GraphDataset(val_files, val_y),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
    )

    model = GlobalFeatureGAT(
        vocabs=all_vocabs,
        scalar_feature_count=NUM_SCALAR_FEATURES,
        num_classes=len(cwe_labels),
        hidden_channels=HIDDEN_CHANNELS,
        num_heads=NUM_GAT_HEADS,
        embedding_dim=EMBEDDING_DIM,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_val_f1, epochs_no_improve = -1.0, 0
    best_model_path = os.path.join(
        SPECIALIST_MODELS_OUTPUT_PATH, "multi_class_cwe_gat.pth"
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            data = data.to(DEVICE)
            out = model(data)
            loss = criterion(out, data.y) / ACCUMULATION_STEPS
            loss.backward()

            if (i + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * ACCUMULATION_STEPS

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(DEVICE)
                out = model(data)
                val_preds.extend(out.argmax(dim=1).cpu().tolist())
                val_true.extend(data.y.cpu().tolist())

        acc = accuracy_score(val_true, val_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_true, val_preds, average="macro", zero_division=0
        )
        cm = confusion_matrix(val_true, val_preds)

        print(
            f"\nEpoch {epoch} | Loss: {total_loss/len(train_loader):.4f} | F1: {f1:.4f} | Acc: {acc:.4f}"
        )
        print("Confusion Matrix:")
        print(cm)

        if f1 > best_val_f1:
            best_val_f1 = f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Model saved (F1: {f1:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    print("\n" + "=" * 50)
    print("FINAL MULTI-CLASS PERFORMANCE REPORT")
    print("=" * 50)
    print(
        classification_report(
            val_true, val_preds, target_names=cwe_labels, zero_division=0
        )
    )


if __name__ == "__main__":
    main()
