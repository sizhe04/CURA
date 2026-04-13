#!/usr/bin/env bash
# Stage-1 parameters: Bio-ClinicalBERT, death_in_30

# ========= Environment =========
CACHE_BASE=""

# ========= Device =========
CUDA_DEVICES=""

# ========= Output =========
OUTPUT_ROOT=""

# ========= Data =========
DATA_CSV=""

TEXT_COL=""
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
