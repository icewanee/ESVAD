import os
import json
from tqdm import tqdm
from config import CPG_PATH, FINAL_GLOBAL_DIR
from multiprocessing import Pool, cpu_count

OUTPUT_DIR = FINAL_GLOBAL_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)
JUNK_BLACKLIST = {
    ";",
    "None",
    # Types
    "double",
    "double*",
    "double**",
    "double***",
    "i1",
    "i16",
    "i16*",
    "i16**",
    "i16***",
    "i32",
    "i32*",
    "i32**",
    "i32***",
    "i64",
    "i64*",
    "i64**",
    "i64***",
    "i8",
    "i8*",
    "i8**",
    "i8***",
    "void",
    "metadata",
    "token",
    # Directives / Linkage
    "declare",
    "dso_local",
    "external",
    "internal",
    "linkonce_odr",
    "private",
    "unnamed_addr",
    "weak_odr",
    "appending",
}


def process_cpg_file(json_path):
    """Worker function to process a single CPG JSON file and extract opcodes."""
    opcodes_in_file = set()
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)

        for node in graph_data.get("nodes", []):
            node_code = str(node.get("properties", {}).get("code", "")).strip()

            if not node_code:
                continue

            code_to_parse = node_code
            if "=" in node_code:
                parts = node_code.split("=", 1)
                if len(parts) > 1:
                    code_to_parse = parts[1].strip()

            opcode = code_to_parse.split(None, 1)[0].strip()

            if opcode in JUNK_BLACKLIST:
                continue

            if (
                opcode
                and not opcode.startswith("%")
                and not opcode.startswith("[")
                and not opcode.endswith(":")
            ):
                opcodes_in_file.add(opcode)
        return opcodes_in_file
    except Exception:
        # Return an empty set if any error occurs during file processing
        return set()


print(f"--- Starting Opcode Discovery from CPGs in: {CPG_PATH} ---")
print(f"--- Using JUNK Blacklist: {len(JUNK_BLACKLIST)} words ---")

all_json_files = []
try:
    print("Gathering all CPG files to process...")
    for cwe_folder in os.listdir(CPG_PATH):
        cwe_path = os.path.join(CPG_PATH, cwe_folder)
        if not os.path.isdir(cwe_path):
            continue
        for state_folder in ["GOOD", "BAD"]:
            state_path = os.path.join(cwe_path, state_folder)
            if not os.path.isdir(state_path):
                continue
            for filename in os.listdir(state_path):
                if filename.endswith(".json"):
                    all_json_files.append(os.path.join(state_path, filename))
except FileNotFoundError:
    print(f"FATAL: Source CPG directory not found at '{CPG_PATH}'. Pipeline stopped.")
    exit()

num_workers = min(cpu_count(), 16)
print(
    f"Found {len(all_json_files)} files. Starting analysis with {num_workers} workers..."
)

all_opcodes_set = set()
with Pool(processes=num_workers) as pool:
    for opcodes_from_file in tqdm(
        pool.imap_unordered(process_cpg_file, all_json_files),
        total=len(all_json_files),
        desc="Analyzing CPGs",
    ):
        all_opcodes_set.update(opcodes_from_file)

sorted_opcodes = sorted(list(all_opcodes_set))

print("\n--- Discovered Opcodes (Junk-List Cleaned) ---")
print(sorted_opcodes)
print(f"Total unique opcodes found: {len(sorted_opcodes)}")

global_opcode_vocabulary = {}
global_opcode_vocabulary["<PAD>"] = 0
global_opcode_vocabulary["<UNK>"] = 1

for i, opcode in enumerate(sorted_opcodes, start=2):
    global_opcode_vocabulary[opcode] = i

output_filename = "global_opcode_vocabulary.json"
output_path = os.path.join(OUTPUT_DIR, output_filename)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(global_opcode_vocabulary, f, indent=4)

print(f"\n--- Completed! ---")
print(f"Global Opcode Vocabulary (Cleaned) saved to: {output_path}")
