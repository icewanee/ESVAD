import json
import os
import random
import glob
import math
from config import (
    FEATURE_STORAGE_PATH,
    RESULTS_PATH,
    KNOWN_CWE_LIST,
    UNSEEN_CWE_LIST,
    CWE_DIRECTORY_PATTERN,
)

BASE_PROCESSED_DIR = FEATURE_STORAGE_PATH
OUTPUT_PATH = RESULTS_PATH
CONFIG_PATH = "config.json"
FIXED_TRAIN_SAMPLES = 2400
MAX_TEST_SAMPLES = 600
LARGE_DATASET_THRESHOLD = FIXED_TRAIN_SAMPLES + MAX_TEST_SAMPLES
SMALL_DATASET_RATIO = 0.8
MIN_SAMPLES_THRESHOLD = 50
MAX_UNSEEN_SAMPLES = 600

print("Loading CWE split configuration...")

KNOWN_CWES = KNOWN_CWE_LIST
UNSEEN_CWES = UNSEEN_CWE_LIST

print(f"Known CWEs: {KNOWN_CWES}")
print(f"Unseen CWEs: {UNSEEN_CWES}")

train_samples_list = []
test_known_samples_list = []
train_unseen_samples_list = []
test_unseen_samples_list = []

print("\nProcessing Known CWEs with hybrid logic...")
for cwe_name in KNOWN_CWES:
    cwe_folder_pattern = os.path.join(
        BASE_PROCESSED_DIR, f"{cwe_name}{CWE_DIRECTORY_PATTERN}"
    )
    found_folders = glob.glob(cwe_folder_pattern)

    if not found_folders:
        print(f"Warning: No folder found for pattern {cwe_folder_pattern}")
        continue

    bad_folder_path = os.path.join(found_folders[0], "BAD")
    if not os.path.isdir(bad_folder_path):
        continue

    all_files = [
        os.path.join(bad_folder_path, f)
        for f in os.listdir(bad_folder_path)
        if f.endswith(".pt")
    ]
    print(f"Processing {cwe_name}: Found {len(all_files)} files.")

    if len(all_files) < MIN_SAMPLES_THRESHOLD:
        print(
            f"  - Warning: Below minimum threshold of {MIN_SAMPLES_THRESHOLD}. Skipping."
        )
        continue

    random.shuffle(all_files)

    if len(all_files) >= LARGE_DATASET_THRESHOLD:
        print(
            f"  - Using FIXED split: Train={FIXED_TRAIN_SAMPLES}, Test=Up to {MAX_TEST_SAMPLES}."
        )
        train_files = all_files[:FIXED_TRAIN_SAMPLES]
        test_known_files = all_files[FIXED_TRAIN_SAMPLES:LARGE_DATASET_THRESHOLD]
    else:
        print(
            f"  - Using RATIO split ({SMALL_DATASET_RATIO*100}%/{ (1-SMALL_DATASET_RATIO)*100 }%)."
        )
        split_point = math.ceil(len(all_files) * SMALL_DATASET_RATIO)
        train_files = all_files[:split_point]
        test_known_files = all_files[split_point:]

        if not test_known_files and train_files:
            test_known_files.append(train_files.pop())

    for file_path in train_files:
        train_samples_list.append({"path": file_path, "label": cwe_name})
    for file_path in test_known_files:
        test_known_samples_list.append({"path": file_path, "label": cwe_name})

    print(
        f"  - Result: {len(train_files)} for train, {len(test_known_files)} for test-known."
    )

print(f"\nProcessing Unseen CWEs (Using up to {MAX_UNSEEN_SAMPLES} samples)...")
for cwe_name in UNSEEN_CWES:
    cwe_folder_pattern = os.path.join(
        BASE_PROCESSED_DIR, f"{cwe_name}{CWE_DIRECTORY_PATTERN}"
    )
    found_folders = glob.glob(cwe_folder_pattern)
    if not found_folders:
        continue
    bad_folder_path = os.path.join(found_folders[0], "BAD")
    if not os.path.isdir(bad_folder_path):
        continue
    all_files = [
        os.path.join(bad_folder_path, f)
        for f in os.listdir(bad_folder_path)
        if f.endswith(".pt")
    ]
    if not all_files:
        continue
    num_to_select = min(len(all_files), MAX_UNSEEN_SAMPLES)
    random.shuffle(all_files)
    selected_files = all_files[:num_to_select]

    split_point = math.ceil(len(selected_files) * SMALL_DATASET_RATIO)
    train_unseen_files = selected_files[:split_point]
    test_unseen_files = selected_files[split_point:]

    if not test_unseen_files and train_unseen_files:
        test_unseen_files.append(train_unseen_files.pop())

    for file_path in train_unseen_files:
        train_unseen_samples_list.append({"path": file_path, "label": cwe_name})
    for file_path in test_unseen_files:
        test_unseen_samples_list.append({"path": file_path, "label": cwe_name})
    print(
        f"Processing {cwe_name}: Selected {len(selected_files)} files "
        f"(Train: {len(train_unseen_files)}, Test: {len(test_unseen_files)})."
    )

os.makedirs(OUTPUT_PATH, exist_ok=True)


def save_data_map(samples, filename):
    data_map = {}
    for sample in samples:
        label = sample["label"]
        path = sample["path"]
        if label not in data_map:
            data_map[label] = {"BAD": []}
        data_map[label]["BAD"].append(path)

    full_output_path = os.path.join(OUTPUT_PATH, filename)
    with open(full_output_path, "w", encoding="utf-8") as f:
        json.dump(data_map, f, indent=4)
    print(f"Successfully saved data map to: {full_output_path}")


print("\nSaving final data maps...")
save_data_map(train_samples_list, "train_data.json")
save_data_map(test_known_samples_list, "test_known_data.json")
save_data_map(train_unseen_samples_list, "train_unseen_data.json")
save_data_map(test_unseen_samples_list, "test_unseen_data.json")
print("\nProcess complete.")
