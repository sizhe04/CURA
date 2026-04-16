#!/usr/bin/env bash
# Stage-1 parameters for CURA clinical LM fine-tuning
#
# Set MODEL_NAME and TARGET_LABELS to select the model and task(s).
# TARGET_LABELS accepts one or more label columns separated by spaces;
# Stage 1 will run a separate fine-tuning experiment for each label and
# write results to independent directories under OUTPUT_ROOT.
# Example: TARGET_LABELS="death_in_30 death_in_7"

# ========= Environment =========
CACHE_BASE=""

# ========= Device =========
CUDA_DEVICES=""

# ========= Output =========
OUTPUT_ROOT=""

# ========= Data =========
DATA_CSV=""

TEXT_COL=""
# One or more label columns separated by spaces, e.g. "death_in_30" or "death_in_30 death_in_7"
TARGET_LABELS=""

# ========= Model =========
MODEL_NAME=""
HEAD_HIDDEN=
DROPOUT=
POOLING=""          # cls | pooler | mean | last_hidden_state | last_token
MAX_LEN=
LOSS_TYPE=""        # ce | edl
EDL_KL_WEIGHT=
CLASSIFIER_ARCH=""  # custom | hf

# ========= L_cal for single-head (Stage-1) =========
USE_LCAL=
LAMBDA_CAL=
CAL_ALPHA=
U_MIN=
U_MAX=

# ========= Metrics =========
ECE_BINS=
POS_ECE_BINS=
F1_FIND_BEST=
F1_THRESHOLD=
F1_GRID_STEP=

# ========= Test Eval during training =========
TEST_EVAL_STRATEGY=""
TEST_EVAL_STEPS=

# ========= Training schedule =========
EPOCHS=
WARMUP=
EVAL_STRATEGY=""    # steps | epoch
EVAL_STEPS=
SAVE_STRATEGY=""    # auto | epoch | steps | no
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

# ========= U_repr =========
USE_REPR_UNCERT=
REPR_K_NEIGHBORS=
REPR_DISTANCE_METRIC=""   # cosine | euclidean
REPR_SEARCH_SPLIT=""      # train | train+val
REPR_SCORE_TYPE=""
REPR_NORMALIZE=           # 1: entropy/log2, margin*2; 0: raw
