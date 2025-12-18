import json

with open("config.json", encoding="utf-8") as f:
    configs = json.load(f)

_BASE_PATH = configs["base_path"]
_RESOURCE_PATH = f"{_BASE_PATH}/Resource"

RESULTS_PATH = f"{_BASE_PATH}/Results"
CPG_PATH = f"{_BASE_PATH}/testcases_cpg"
BASE_VOCAB_DIR = f"{_RESOURCE_PATH}/CWE_features"
FINAL_GLOBAL_DIR = f"{_RESOURCE_PATH}/Global"
FEATURE_STORAGE_PATH = f"{RESULTS_PATH}/pt_features"
KNOWN_CWE_LIST = configs["known_cwe_list"]
UNSEEN_CWE_LIST = configs["unseen_cwe_list"]
CWE_DIRECTORY_PATTERN = configs["cwe_directory_pattern"]

SPECIALIST_MODELS_OUTPUT_PATH = f"{RESULTS_PATH}/specialist_models"
SPECIALIST_MODELS_TRAINING_EPOCH = configs["specialist_models_training"]["epochs"]
SPECIALIST_MODELS_TRAINING_BATCH_SIZE = configs["specialist_models_training"][
    "batch_size"
]
SPECIALIST_MODELS_TRAINING_ACCUMULATION_STEPS = configs["specialist_models_training"][
    "accumulation_steps"
]
SPECIALIST_MODELS_TRAINING_LEARNING_RATE = configs["specialist_models_training"][
    "learning_rate"
]
SPECIALIST_MODELS_TRAINING_PATIENCE = configs["specialist_models_training"]["patience"]

METRIC_LEARNING_MODEL_OUTPUT_PATH = f"{RESULTS_PATH}/metric_learning_model"
METRIC_LEARNING_MODEL_TRAINING_EPOCH = configs["metric_learning_training"]["epochs"]
METRIC_LEARNING_MODEL_TRAINING_LEARNING_RATE_HEAD = configs["metric_learning_training"][
    "learning_rate_head"
]
METRIC_LEARNING_MODEL_TRAINING_LEARNING_RATE_GAT = configs["metric_learning_training"][
    "learning_rate_gat"
]
METRIC_LEARNING_MODEL_TRAINING_BATCH_SIZE = configs["metric_learning_training"][
    "batch_size"
]
METRIC_LEARNING_MODEL_TRAINING_M_SAMPLES = configs["metric_learning_training"][
    "m_samples"
]
METRIC_LEARNING_MODEL_TRAINING_PATIENCE = configs["metric_learning_training"][
    "patience"
]

TOOLKIT_PATH = f"{RESULTS_PATH}/reference"

CREATE_CLASSIFIER_BATCH_SIZE = configs["create_classifier"]["batch_size"]
TUNE_THRESHOLD_OUTPUT_JSON = f"{RESULTS_PATH}/08_optimal_threshold.json"
TUNE_THRESHOLD_BATCH_SIZE = configs["tune_threshold"]["batch_size"]

VALIDATE_ANOMALY_BATCH_SIZE = configs["validate_anomaly"]["batch_size"]
