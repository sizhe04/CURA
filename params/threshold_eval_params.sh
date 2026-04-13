#!/usr/bin/env bash
# Stage-2 selective evaluation parameters

# Stage-2 experiment directory
STAGE2_DIR=""

# Python interpreter
PYTHON_BIN=""

# Coverage/threshold sequence (comma-separated)
THRESHOLDS=""

# Metrics to compute (comma-separated)
METRICS=""

# Stage-1 embeddings root (default: auto-inferred from STAGE2_DIR)
EMBEDDINGS_ROOT=""

# Embedding source
EMBEDDING_SOURCE=""

# Pooling (default: read from hparams_stage1.json)
POOLING=""

# Inference batch size
BATCH_SIZE=

# Device (e.g. "cuda:0" or "cpu"; empty for auto-detect)
DEVICE=""
