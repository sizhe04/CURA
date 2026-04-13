#!/usr/bin/env bash
# Parameters for computing U_repr (neighborhood uncertainty) on existing Stage-1 outputs

# ====== Required: Stage-1 label directory ======
# Example: STAGE1_LABEL_DIR="./results/Stage1/emilyalsentzer-Bio_ClinicalBERT/.../death_in_30"
STAGE1_LABEL_DIR=""

# ====== Optional: device and cache ======
CUDA_DEVICES=""
CACHE_BASE=""

# ====== U_repr hyperparameters ======
REPR_K_NEIGHBORS=
REPR_DISTANCE_METRIC=""     # cosine | euclidean
REPR_SEARCH_SPLIT=""        # train | train+val
REPR_SCORE_TYPE=""          # recorded in meta; both entropy and margin are always computed
REPR_NORMALIZE=             # 1: entropy/log2, margin*2; 0: no normalization
