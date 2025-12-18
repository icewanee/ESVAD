cp config.realsard.json config.json
python3.11 01_global_vocab.py && \
python3.11 02_global_opcode.py && \
python3.11 03_generate_feature.py && \
python3.11 04_split_dataset.py && \
python3.11 05_train_specialist_models.py && \
python3.11 06_train_metric_learning.py && \
python3.11 07_create_classifier.py && \
python3.11 08_tune_threshold.py && \
python3.11 09_validate_anomaly.py
