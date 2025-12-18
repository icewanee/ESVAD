import os
import json
import re
from multiprocessing import Pool, cpu_count
import networkx as nx
from tqdm import tqdm

from config import CPG_PATH, BASE_VOCAB_DIR, FINAL_GLOBAL_DIR

GLOBAL_FEATURES_DIR = "CWE_features_global"


def parse_feature_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return set(re.findall(r'[\'"](.*?)[\'"]', content))
    except Exception as e:
        print(f"  - Error reading {file_path}: {e}")
        return set()


def make_features_for_discovery(graph, all_vocabs):
    source_vocab = all_vocabs["sources"]
    sink_vocab = all_vocabs["sinks"]
    string_manip_vocab = all_vocabs["string_manipulation"]

    def find_functions_in_node(vocab, node_code_str):
        found = []
        for func_name in vocab.keys():
            if func_name in node_code_str and func_name not in ["<PAD>", "<UNK>"]:
                found.append(func_name)
        return found

    graph_sources, graph_sinks, graph_strings = [], [], []
    for _, node_data in graph.nodes(data=True):
        node_code = str(node_data["properties"].get("code", ""))

        graph_sources.extend(find_functions_in_node(source_vocab, node_code))
        graph_sinks.extend(find_functions_in_node(sink_vocab, node_code))
        graph_strings.extend(find_functions_in_node(string_manip_vocab, node_code))

    return {
        "sources": graph_sources,
        "sinks": graph_sinks,
        "string_manipulation": graph_strings,
    }


def clean_graph(graph_data):
    new_edges, relavant_node_ids = [], set()
    for edge in graph_data.get("edges", []):
        if edge.get("type") not in ["AST", "DFG", "CDG"]:
            continue
        relavant_node_ids.add(edge["startNode"])
        relavant_node_ids.add(edge["endNode"])
        new_edges.append(edge)
    new_nodes = [
        node for node in graph_data.get("nodes", []) if node["id"] in relavant_node_ids
    ]
    return {"nodes": new_nodes, "edges": new_edges}


def step1_prepare_base_data():
    print(f"\n{'='*20} STEP 1: Preparing Base Data {'='*20}")
    os.makedirs(BASE_VOCAB_DIR, exist_ok=True)
    os.makedirs(FINAL_GLOBAL_DIR, exist_ok=True)

    try:
        feature_types = [
            d
            for d in os.listdir(GLOBAL_FEATURES_DIR)
            if os.path.isdir(os.path.join(GLOBAL_FEATURES_DIR, d))
        ]
    except FileNotFoundError:
        print(
            f"Error: Base features directory not found at '{GLOBAL_FEATURES_DIR}'. Pipeline stopped."
        )
        return False

    discovery_types = ["cwe_source", "cwe_sink", "cwe_string_manipulation"]
    for feature_type in feature_types:
        feature_type_path = os.path.join(GLOBAL_FEATURES_DIR, feature_type)
        all_features_set = set()
        for filename in os.listdir(feature_type_path):
            if filename.endswith(".txt"):
                all_features_set.update(
                    parse_feature_file(os.path.join(feature_type_path, filename))
                )

        if not all_features_set:
            continue

        sorted_features = sorted(list(all_features_set))
        vocabulary = {"<PAD>": 0, "<UNK>": 1}
        for i, feature in enumerate(sorted_features, start=2):
            vocabulary[feature] = i

        if feature_type.lower() in discovery_types:
            output_filename = f"base_global_{feature_type.lower()}_vocabulary.json"
            output_path = os.path.join(BASE_VOCAB_DIR, output_filename)
            print(f"  - Saved base vocab for '{feature_type}' to {output_path}")
        else:
            output_filename = f"global_{feature_type.lower()}_vocabulary.json"
            output_path = os.path.join(FINAL_GLOBAL_DIR, output_filename)
            print(f"  - Saved FINAL vocab for '{feature_type}' to {output_path}")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(vocabulary, f, indent=4)

    print("--- Step 1 finished successfully! ---")
    return True


def process_cpg_file(args):
    """Worker function to process a single CPG JSON file."""
    json_path, all_vocabs = args
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)

        relavent_graph_data = clean_graph(graph_data)
        if not relavent_graph_data["nodes"]:
            return None

        G = nx.node_link_graph(
            relavent_graph_data,
            edges="edges",
            source="startNode",
            target="endNode",
            directed=True,
        )

        found = make_features_for_discovery(G, all_vocabs)
        return found

    except Exception:
        # Errors are ignored in the original script, so we'll return None to signify an issue.
        return None


def step2_discover_and_finalize():
    print(f"\n{'='*20} STEP 2: Discovering and Finalizing Vocabularies {'='*20}")

    def load_base_vocab(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    source_vocab = load_base_vocab(
        os.path.join(BASE_VOCAB_DIR, "base_global_cwe_source_vocabulary.json")
    )
    sink_vocab = load_base_vocab(
        os.path.join(BASE_VOCAB_DIR, "base_global_cwe_sink_vocabulary.json")
    )
    string_manip_vocab = load_base_vocab(
        os.path.join(
            BASE_VOCAB_DIR, "base_global_cwe_string_manipulation_vocabulary.json"
        )
    )

    if not any([source_vocab, sink_vocab, string_manip_vocab]):
        print("Error: Base vocabularies from Step 1 not found. Pipeline stopped.")
        return False

    all_vocabs = {
        "sources": source_vocab,
        "sinks": sink_vocab,
        "string_manipulation": string_manip_vocab,
    }
    overall_unique_sources, overall_unique_sinks, overall_unique_strings = (
        set(),
        set(),
        set(),
    )
    try:
        all_json_files = []
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
        print(
            f"FATAL: Source CPG directory not found at '{CPG_PATH}'. Pipeline stopped."
        )
        return False

    num_workers = min(cpu_count(), 16)
    print(f"Starting CPG analysis with {num_workers} worker processes...")

    with Pool(processes=num_workers) as pool:
        # Prepare arguments for the worker function
        tasks = [(path, all_vocabs) for path in all_json_files]

        # Process files in parallel and show progress
        for found in tqdm(
            pool.imap_unordered(process_cpg_file, tasks),
            total=len(all_json_files),
            desc="Analyzing CPGs",
        ):
            if found:
                overall_unique_sources.update(found["sources"])
                overall_unique_sinks.update(found["sinks"])
                overall_unique_strings.update(found["string_manipulation"])

    def create_and_save_final_vocab(unique_set, filename):
        if not unique_set:
            print(f"  - No discovered functions for {filename}, skipping.")
            return

        sorted_features = sorted(list(unique_set))
        vocabulary = {"<PAD>": 0, "<UNK>": 1}
        for i, feature in enumerate(sorted_features, start=2):
            vocabulary[feature] = i

        output_path = os.path.join(FINAL_GLOBAL_DIR, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(vocabulary, f, indent=4)
        print(
            f"  - Saved FINAL vocab with {len(vocabulary)} discovered words to: {output_path}"
        )

    create_and_save_final_vocab(
        overall_unique_sources, "global_cwe_source_vocabulary.json"
    )
    create_and_save_final_vocab(overall_unique_sinks, "global_cwe_sink_vocabulary.json")
    create_and_save_final_vocab(
        overall_unique_strings, "global_cwe_string_manipulation_vocabulary.json"
    )

    print("--- Step 2 finished successfully! ---")
    return True


def main():
    if step1_prepare_base_data():
        if step2_discover_and_finalize():
            print(f"\n{'='*25} PIPELINE COMPLETED SUCCESSFULLY! {'='*25}")
            print(f"Final vocabularies are ready in: {FINAL_GLOBAL_DIR}")


if __name__ == "__main__":
    main()
