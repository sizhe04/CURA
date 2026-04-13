#!/usr/bin/env bash
# Stage-1 parameters: ClinicalBERT, death_in_30

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
POOLING=""
MAX_LEN=
LOSS_TYPE=""
EDL_KL_WEIGHT=
CLASSIFIER_ARCH=""

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
EVAL_STRATEGY=""
EVAL_STEPS=
SAVE_STRATEGY=""
SAVE_STEPS=
SAVE_TOTAL_LIMIT=
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
REPR_DISTANCE_METRIC=""
REPR_SEARCH_SPLIT=""
REPR_SCORE_TYPE=""
REPR_NORMALIZE=
