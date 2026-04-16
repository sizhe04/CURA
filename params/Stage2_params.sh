#!/usr/bin/env bash
# Stage-2 parameters for CURA uncertainty fine-tuning

# ========= Environment =========
CACHE_BASE=""

# ========= Device =========
CUDA_DEVICES=""

# ========= Input (Stage-1 embeddings) =========
# Path to Stage-1 checkpoint directory containing fold_*/embeddings_last/ or embeddings/
# Example: ./results/Stage1/model_name/.../death_in_30/checkpoint
EMBEDDINGS_ROOT=""

# ========= Embedding Source =========
#   - embeddings_last: last checkpoint embeddings
#   - embeddings: best checkpoint embeddings
#   - embeddings_last_llm_uncertain_K-<int>: embeddings with pre-computed neighbor stats
EMBEDDING_SOURCE=""

# Must match Stage-1 pooling
POOLING=""
MODEL_NAME=""

# Single label only. EMBEDDINGS_ROOT points to one task's Stage-1 checkpoint directory,
# so Stage-2 must be run separately for each task (e.g. "death_in_30").
TARGET_LABELS=""

# ========= Multi-head architecture =========
NUM_HEADS=
HEAD_HIDDEN=
DROPOUT=

# ========= L_ind: Individual Uncertainty Calibration (paper's lambda_ind) =========
LAMBDA_CAL=
CAL_ALPHA=
U_MIN=
U_MAX=

# ========= L_coh: Cohort-Aware Risk Alignment (paper's lambda_coh) =========
# Set LAMBDA_POP=0 to disable cohort-aware loss
LAMBDA_POP=
POP_K_NEIGHBORS=
POP_DISTANCE_METRIC=""
POP_SEARCH_SPLIT=""

# ========= Metrics =========
ECE_BINS=
POS_ECE_BINS=
F1_FIND_BEST=
F1_THRESHOLD=
F1_GRID_STEP=

# ========= Test Eval during training =========
#   - end_only: evaluate test set only at end (default)
#   - epoch: evaluate test set every epoch
#   - steps: evaluate test set every TEST_EVAL_STEPS
#   - none: never evaluate test set during training
TEST_EVAL_STRATEGY=""
TEST_EVAL_STEPS=

# ========= Training schedule =========
EPOCHS=
WARMUP=

EVAL_STRATEGY=""
EVAL_STEPS=

SAVE_STRATEGY=""
SAVE_STEPS=
SAVE_TOTAL_LIMIT=

# ========= Best metric for checkpoint selection =========
BEST_METRIC=""
GREATER_IS_BETTER=

# ========= Training hyperparameters =========
LR=
WD=
BSZ=
EVAL_BSZ=
GA=
SEED=
FP16_FLAG=""

# ========= Fold selection (optional) =========
# Specify fold ids (1-based), e.g. "2 3 4 5"; leave empty for all 5 folds
FOLD_IDS=""

# ========= Early stopping =========
USE_EARLY_STOPPING=
EARLY_STOPPING_PATIENCE=
EARLY_STOPPING_THRESHOLD=

# ========= Weights & Biases =========
USE_WANDB=
WANDB_PROJECT=""
WANDB_ENTITY=""
WANDB_RUN_NAME=""
