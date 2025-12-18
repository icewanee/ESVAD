import json
import networkx as nx
import os
import re
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from torch_geometric.utils import from_networkx
from multiprocessing import Pool, cpu_count
from config import CPG_PATH, FINAL_GLOBAL_DIR, FEATURE_STORAGE_PATH

VOCAB_DIR = FINAL_GLOBAL_DIR
OUTPUT_PATH = FEATURE_STORAGE_PATH


def load_vocab(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Vocabulary not found at: {path}")
        return {}


print("loading Global Vocabularies...")
OPCODE_VOCAB = load_vocab(os.path.join(VOCAB_DIR, "global_opcode_vocabulary.json"))
SOURCE_VOCAB = load_vocab(os.path.join(VOCAB_DIR, "global_cwe_source_vocabulary.json"))
SINK_VOCAB = load_vocab(os.path.join(VOCAB_DIR, "global_cwe_sink_vocabulary.json"))
PAYLOAD_VOCAB = load_vocab(
    os.path.join(VOCAB_DIR, "global_cwe_payload_vocabulary.json")
)
STRING_MANIPULATION_VOCAB = load_vocab(
    os.path.join(VOCAB_DIR, "global_cwe_string_manipulation_vocabulary.json")
)

WRITE_OPS = {"strcpy", "memcpy", "sprintf", "store", "strcat"}
READ_OPS = {"load", "read", "recv", "gets"}


def make_features(graph, all_vocabs):
    opcode_vocab = all_vocabs["opcodes"]
    source_vocab = all_vocabs["sources"]
    sink_vocab = all_vocabs["sinks"]
    payload_vocab = all_vocabs["payloads"]
    string_manip_vocab = all_vocabs["string_manipulation"]

    dfg_edges = [
        (u, v) for u, v, data in graph.edges(data=True) if data.get("type") == "DFG"
    ]
    dfg_graph = nx.DiGraph(dfg_edges)

    payload_patterns = {
        re.escape(p): idx
        for p, idx in payload_vocab.items()
        if p not in ["<PAD>", "<UNK>"]
    }

    all_dangerous_nodes = {
        node_id
        for node_id, node_data in graph.nodes(data=True)
        if "Literal" in node_data.get("labels", [])
        or "Identifier" in node_data.get("labels", [])
        if any(
            re.search(pattern, str(node_data["properties"].get("code", "")))
            for pattern in payload_patterns.keys()
        )
    }

    all_source_call_nodes = {
        node_id
        for node_id, node_data in graph.nodes(data=True)
        if "CallExpression" in node_data.get("labels", [])
        if any(
            src in str(node_data["properties"].get("code", "")) for src in source_vocab
        )
    }

    all_concatenation_source_nodes = set()
    for node_id in all_dangerous_nodes:
        if dfg_graph.has_node(node_id):
            for successor_id in dfg_graph.successors(node_id):
                successor_data = graph.nodes.get(successor_id, {})
                if "CallExpression" in successor_data.get("labels", []) and any(
                    func in str(successor_data.get("properties", {}).get("code", ""))
                    for func in string_manip_vocab
                ):
                    all_concatenation_source_nodes.add(node_id)
                    break

    node_features = {}
    in_degree_dict = dict(graph.in_degree)
    out_degree_dict = dict(graph.out_degree)

    for node_id, node_data in graph.nodes(data=True):
        node_code = str(node_data["properties"].get("code", ""))
        code_tokens = set(node_code.strip().replace("(", " ").replace(")", " ").split())
        contains_write_op = 1 if not code_tokens.isdisjoint(WRITE_OPS) else 0
        contains_read_op = 1 if not code_tokens.isdisjoint(READ_OPS) else 0
        in_degree = in_degree_dict.get(node_id, 0)
        out_degree = out_degree_dict.get(node_id, 0)

        is_pointer = 1 if "*" in node_code else 0
        is_array = (
            1
            if ("[" in node_code and "x" in node_code)
            or "NewArrayExpression" in node_data["labels"]
            else 0
        )

        def get_indices_from_code(vocab, tokens):
            indices = []
            for word, word_id in vocab.items():
                if word in tokens and word not in ["<PAD>", "<UNK>"]:
                    indices.append(word_id)
            return indices

        def get_payload_indices_from_code(payload_patterns_dict, code_str):
            indices = []
            for pattern, idx in payload_patterns_dict.items():
                if re.search(pattern, code_str):
                    indices.append(idx)
            return sorted(list(set(indices)))

        opcode_indices = get_indices_from_code(opcode_vocab, node_code)
        source_indices = get_indices_from_code(source_vocab, node_code)
        sink_indices = get_indices_from_code(sink_vocab, node_code)
        string_manip_indices = get_indices_from_code(string_manip_vocab, node_code)
        payload_indices = get_payload_indices_from_code(payload_patterns, node_code)

        has_payload = 1 if node_id in all_dangerous_nodes else 0
        is_string_concatenation = 1 if node_id in all_concatenation_source_nodes else 0

        has_dangerous_ci_anc = 0
        has_concatenated_anc = 0
        source_type = 0

        if dfg_graph.has_node(node_id):
            ancestors = nx.ancestors(dfg_graph, node_id)
            dangerous_ancestors = ancestors.intersection(all_dangerous_nodes)

            if dangerous_ancestors:
                has_dangerous_ci_anc = 1

                if not dangerous_ancestors.isdisjoint(all_source_call_nodes):
                    source_type = 2
                else:
                    source_type = 1

                if not dangerous_ancestors.isdisjoint(all_concatenation_source_nodes):
                    has_concatenated_anc = 1

        node_features[node_id] = {
            "scalar_features": [
                in_degree,
                out_degree,
                contains_write_op,
                contains_read_op,
                is_pointer,
                is_array,
                has_payload,
                is_string_concatenation,
                has_dangerous_ci_anc,
                has_concatenated_anc,
                source_type,
            ],
            "opcode_indices": opcode_indices,
            "source_indices": source_indices,
            "sink_indices": sink_indices,
            "string_manip_indices": string_manip_indices,
            "payload_indices": payload_indices,
        }

    return node_features


def clean_graph(graph_data):
    new_edges = []
    relavant_node_ids = set()
    for edge in graph_data["edges"]:
        if edge["type"] not in ["AST", "DFG", "CDG"]:
            continue
        relavant_node_ids.add(edge["startNode"])
        relavant_node_ids.add(edge["endNode"])
        new_edges.append(edge)

    new_nodes = [
        node for node in graph_data["nodes"] if node["id"] in relavant_node_ids
    ]
    return {"nodes": new_nodes, "edges": new_edges}


def process_file(args):
    """Worker function to process a single CPG JSON file and save the feature tensor."""
    json_path, target_dir, all_vocabs, state_folder = args
    filename = os.path.basename(json_path)

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)

        relavent_graph_data = clean_graph(graph_data)
        if not relavent_graph_data["nodes"]:
            return f"Skipped (no relevant nodes): {filename}"

        G = nx.node_link_graph(
            relavent_graph_data,
            edges="edges",
            source="startNode",
            target="endNode",
            directed=True,
        )

        feature_dict = make_features(G, all_vocabs)
        node_order = list(G.nodes())

        scalar_feature_list = [
            feature_dict[node_id]["scalar_features"] for node_id in node_order
        ]
        scalar_features_tensor = torch.tensor(scalar_feature_list, dtype=torch.float)

        def pad_sequences(sequences, max_len, pad_value):
            return [seq + ([pad_value] * (max_len - len(seq))) for seq in sequences]

        # Opcode features
        opcode_seqs = [
            feature_dict[node_id]["opcode_indices"] for node_id in node_order
        ]
        max_len_op = len(all_vocabs["opcodes"])
        padded_opcodes = pad_sequences(opcode_seqs, max_len_op, 0)
        opcode_indices_tensor = torch.tensor(padded_opcodes, dtype=torch.long)

        # Source features
        source_seqs = [
            feature_dict[node_id]["source_indices"] for node_id in node_order
        ]
        max_len_src = len(all_vocabs["sources"])
        padded_sources = pad_sequences(source_seqs, max_len_src, 0)
        source_indices_tensor = torch.tensor(padded_sources, dtype=torch.long)

        # Sink features
        sink_seqs = [feature_dict[node_id]["sink_indices"] for node_id in node_order]
        max_len_snk = len(all_vocabs["sinks"])
        padded_sinks = pad_sequences(sink_seqs, max_len_snk, 0)
        sink_indices_tensor = torch.tensor(padded_sinks, dtype=torch.long)

        # String manipulation features
        string_manip_seqs = [
            feature_dict[node_id]["string_manip_indices"] for node_id in node_order
        ]
        max_len_str = len(all_vocabs["string_manipulation"])
        padded_string_manip = pad_sequences(string_manip_seqs, max_len_str, 0)
        string_manip_indices_tensor = torch.tensor(
            padded_string_manip, dtype=torch.long
        )

        # Payload features
        payload_seqs = [
            feature_dict[node_id]["payload_indices"] for node_id in node_order
        ]
        max_len_payload = len(all_vocabs["payloads"])
        padded_payloads = pad_sequences(payload_seqs, max_len_payload, 0)
        payload_indices_tensor = torch.tensor(padded_payloads, dtype=torch.long)

        pyg_data = from_networkx(G)

        pyg_data.x_scalar = scalar_features_tensor
        pyg_data.x_opcode = opcode_indices_tensor
        pyg_data.x_source = source_indices_tensor
        pyg_data.x_sink = sink_indices_tensor
        pyg_data.x_string_manip = string_manip_indices_tensor
        pyg_data.x_payload = payload_indices_tensor

        pyg_data.y = torch.tensor([0 if state_folder == "GOOD" else 1])

        output_filename = os.path.join(target_dir, f"{filename[:-5]}.pt")
        torch.save(pyg_data, output_filename)

        return None  # Success

    except Exception as e:
        return f"Error processing {filename}: {e}"


def main():
    print("--- Starting Feature Generation from CPGs ---")
    try:
        cwe_folders = [
            d for d in os.listdir(CPG_PATH) if os.path.isdir(os.path.join(CPG_PATH, d))
        ]
    except FileNotFoundError:
        print(f"FATAL: Source CPG directory not found at '{CPG_PATH}'")
        return

    all_vocabs = {
        "opcodes": OPCODE_VOCAB,
        "sources": SOURCE_VOCAB,
        "sinks": SINK_VOCAB,
        "payloads": PAYLOAD_VOCAB,
        "string_manipulation": STRING_MANIPULATION_VOCAB,
    }

    tasks = []
    print("Gathering all CPG files to process...")
    for cwe_folder in cwe_folders:
        for state_folder in ["GOOD", "BAD"]:
            state_path = os.path.join(CPG_PATH, cwe_folder, state_folder)
            if not os.path.isdir(state_path):
                continue

            target_dir = os.path.join(OUTPUT_PATH, cwe_folder, state_folder)
            os.makedirs(target_dir, exist_ok=True)

            for filename in os.listdir(state_path):
                if not filename.endswith(".json"):
                    continue
                json_path = os.path.join(state_path, filename)
                tasks.append((json_path, target_dir, all_vocabs, state_folder))

    num_workers = min(cpu_count(), 16)
    print(
        f"Found {len(tasks)} files. Starting feature generation with {num_workers} workers..."
    )

    with Pool(processes=num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(process_file, tasks),
            total=len(tasks),
            desc="Generating Features",
        ):
            pass

    print(f"\n--- Feature Generation Complete! ---")
    print(f"Generated features saved in: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
