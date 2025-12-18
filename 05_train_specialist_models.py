import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch_geometric.nn import GATConv, global_max_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import json
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import random
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
torch.cuda.set_device(0)


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        return torch.load(self.file_paths[idx], weights_only=False)


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

        x = self.conv1(final_node_features, data.edge_index)
        x = F.elu(x)
        x = self.conv2(x, data.edge_index)
        x = F.elu(x)

        x_pooled = global_max_pool(x, data.batch)
        output = self.classifier_head(x_pooled)
        return output


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

    print(f"--- Loading Master Train Map from: {MASTER_TRAIN_MAP_PATH} ---")
    try:
        with open(MASTER_TRAIN_MAP_PATH, "r") as f:
            master_train_map = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: {MASTER_TRAIN_MAP_PATH} not found.")
        print("Please run split_dataset_final.py first to create the master data maps.")
        return

    cwe_names_to_train = sorted(master_train_map.keys())
    print(
        f"--- Found {len(cwe_names_to_train)} CWEs in Train Map: {cwe_names_to_train} ---"
    )

    for cwe_name in cwe_names_to_train:
        print(f"\n{'='*20} Starting Training for: {cwe_name} {'='*20}")

        bad_files = master_train_map[cwe_name].get("BAD", [])
        if not bad_files:
            print(
                f"  - No BAD files found in train_data.json for {cwe_name}. Skipping."
            )
            continue

        good_files = []
        try:
            first_bad_file_path = bad_files[0]
            cwe_base_folder = os.path.dirname(os.path.dirname(first_bad_file_path))
            good_path = os.path.join(cwe_base_folder, "GOOD")

            if os.path.isdir(good_path):
                good_files = [
                    os.path.join(good_path, f)
                    for f in os.listdir(good_path)
                    if f.endswith(".pt")
                ]
            else:
                print(f"  - Warning: GOOD directory not found at {good_path}")
        except Exception as e:
            print(f"  - Error finding GOOD files: {e}. Skipping.")
            continue

        if not good_files:
            print(
                f"  - CRITICAL: No GOOD files found. Cannot train binary classifier for {cwe_name}. Skipping."
            )
            continue
        num_bad_files = len(bad_files)

        if len(good_files) < num_bad_files:
            print(
                f"  - Warning: Not enough GOOD files ({len(good_files)}) to match BAD files ({num_bad_files})."
            )
            print(f"  - Using {len(good_files)} files for both.")
            num_files_to_use = len(good_files)
            sampled_good = good_files
            sampled_bad = random.sample(bad_files, num_files_to_use)
        else:
            print(
                f"  - Balancing data: Sampling {num_bad_files} GOOD files to match {num_bad_files} BAD files."
            )
            sampled_good = random.sample(good_files, num_bad_files)
            sampled_bad = bad_files

        print(
            f"  - Sampling data: {len(sampled_good)} GOOD files, {len(sampled_bad)} BAD files (from train_data.json)."
        )

        all_files = sampled_good + sampled_bad
        all_labels = [0] * len(sampled_good) + [1] * len(sampled_bad)

        if len(all_files) < 10:
            print(f"Not enough data for {cwe_name}. Skipping.")
            continue

        train_files, val_files, _, _ = train_test_split(
            all_files,
            all_labels,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=all_labels,
        )

        train_dataset = GraphDataset(train_files)
        val_dataset = GraphDataset(val_files)

        num_workers = 0 if os.name == "nt" else 4
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers
        )
        NUM_SCALAR_FEATURES = 11

        model = GlobalFeatureGAT(
            vocabs=all_vocabs,
            scalar_feature_count=NUM_SCALAR_FEATURES,
            hidden_channels=HIDDEN_CHANNELS,
            num_heads=NUM_GAT_HEADS,
            embedding_dim=EMBEDDING_DIM,
        ).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        best_val_f1 = -1.0
        epochs_no_improve = 0
        best_accuracy = 0
        actualG = ""
        actualB = ""

        best_model_path = os.path.join(
            SPECIALIST_MODELS_OUTPUT_PATH, f"best_model_{cwe_name}.pth"
        )

        print("\n--- Starting Training of GlobalFeatureGAT Model ---")
        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss = 0
            optimizer.zero_grad()  # Initialize gradients once at start of epoch
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

            for i, data in enumerate(progress_bar):
                data = data.to(DEVICE)

                out = model(data)
                loss = criterion(out, data.y)

                loss = loss / ACCUMULATION_STEPS
                loss.backward()

                if (i + 1) % ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()

                total_loss += loss.item() * ACCUMULATION_STEPS
                progress_bar.set_postfix(
                    {"loss": f"{(loss.item() * ACCUMULATION_STEPS):.4f}"}
                )

            avg_loss = total_loss / len(train_loader)

            model.eval()
            val_preds, val_true = [], []
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(DEVICE)
                    out = model(data)
                    preds = out.argmax(dim=1)
                    val_preds.extend(preds.cpu().tolist())
                    val_true.extend(data.y.cpu().tolist())

            report = classification_report(
                val_true, val_preds, output_dict=True, zero_division=0
            )
            val_f1 = report["macro avg"]["f1-score"]
            val_accuracy = report["accuracy"]

            print(
                f"Epoch {epoch} | Avg Train Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_accuracy:.4f}"
            )

            cm = confusion_matrix(val_true, val_preds, labels=[0, 1])
            print("Confusion Matrix:")
            print("               Predicted")
            print("               GOOD | BAD")
            print(f"Actual GOOD: {cm[0][0]:>5} | {cm[0][1]:>3}")
            print(f"Actual BAD:  {cm[1][0]:>5} | {cm[1][1]:>3}")
            print("-" * 50)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"  -> New best model saved with F1-Score: {best_val_f1:.4f}")
            else:
                epochs_no_improve += 1

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                actualG = f"{cm[0][0]:>5} | {cm[0][1]:>3}"
                actualB = f"{cm[1][0]:>5} | {cm[1][1]:>3}"

            if epochs_no_improve >= PATIENCE:
                print(
                    f"\nEarly stopping triggered after {PATIENCE} epochs with no improvement."
                )
                break

        print(
            f"--- Finished training for {cwe_name}. Best F1-Score: {best_val_f1:.4f} ---"
        )  # [!! แก้ไข]
        print(f"Model saved at: {best_model_path}")

        results_file_path = os.path.join(
            RESULTS_PATH, "specialist_models_training_summary.txt"
        )
        with open(results_file_path, "a", encoding="utf-8") as f:
            summary = (
                f"CWE: {cwe_name}, "  # [!! แก้ไข]
                f"Best_F1_Macro: {best_val_f1:.4f}, "
                f"Best_Accuracy: {best_accuracy:.4f},\n"
                f"Confusion Matrix:\n"
                f"               Predicted\n"
                f"               GOOD | BAD\n"
                f"Actual GOOD: {actualG}\n"
                f"Actual BAD:  {actualB}\n"
                f'{"-" * 50}\n'
            )

            f.write(summary)

    print(f"\n--- Training Finished! ---")


if __name__ == "__main__":
    main()
