#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash run_stage1.sh <param_file> [KEY=VALUE ...]"
  echo "Example: bash run_stage1.sh params/Stage1_Bio-ClinicalBERT_30_params.sh"
  exit 1
fi

PARAM_FILE="$1"
shift
echo "[run_stage1.sh] Using param file: ${PARAM_FILE}"

if [[ ! -f "${PARAM_FILE}" ]]; then
  echo "Error: param file not found: ${PARAM_FILE}"
  exit 1
fi

source "${PARAM_FILE}"

# Optional WANDB API key (set via environment variable)
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  export WANDB_API_KEY
fi

# HuggingFace cache directories
export HF_HOME="${CACHE_BASE}"
export HF_HUB_CACHE="${CACHE_BASE}/hub"
export HUGGINGFACE_HUB_CACHE="${CACHE_BASE}/hub"
export HF_DATASETS_CACHE="${CACHE_BASE}/datasets"
export TRANSFORMERS_CACHE="${CACHE_BASE}/models"
export XDG_CACHE_HOME="${CACHE_BASE}/xdg"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME"

# GPU selection
if [[ -n "${CUDA_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
  echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

# Weights & Biases
WANDB_FLAGS=""
if [[ "${USE_WANDB}" == "1" ]]; then
  export WANDB_PROJECT="${WANDB_PROJECT}"
  export WANDB_ENTITY="${WANDB_ENTITY}"
  export WANDB_MODE=online
  WANDB_FLAGS="--use_wandb --wandb_run_name ${WANDB_RUN_NAME}"
else
  export WANDB_MODE=offline
fi

LAUNCHER="python"

${LAUNCHER} "${SCRIPT_DIR}/stage1_cv_train.py" \
  --data_csv "${DATA_CSV}" \
  --text_col "${TEXT_COL}" \
  --labels ${TARGET_LABELS} \
  --model_name "${MODEL_NAME}" \
  --head_hidden_dim ${HEAD_HIDDEN} \
  --dropout ${DROPOUT} \
  --pooling "${POOLING}" \
  --loss_type "${LOSS_TYPE}" \
  --edl_kl_weight ${EDL_KL_WEIGHT} \
  --classifier_arch "${CLASSIFIER_ARCH}" \
  --lambda_cal ${LAMBDA_CAL} \
  --cal_alpha ${CAL_ALPHA} \
  --u_min ${U_MIN} \
  --u_max ${U_MAX} \
  --max_length ${MAX_LEN} \
  --ece_bins ${ECE_BINS} \
  --pos_ece_bins ${POS_ECE_BINS} \
  $( [[ ${F1_FIND_BEST} -eq 1 ]] && echo "--f1_find_best" ) \
  --f1_threshold ${F1_THRESHOLD} \
  --f1_grid_step ${F1_GRID_STEP} \
  --test_eval_strategy ${TEST_EVAL_STRATEGY:-end_only} \
  --test_eval_steps ${TEST_EVAL_STEPS:-2000} \
  --eval_strategy ${EVAL_STRATEGY} \
  --eval_steps ${EVAL_STEPS} \
  --save_strategy ${SAVE_STRATEGY} \
  --save_steps ${SAVE_STEPS} \
  --save_total_limit ${SAVE_TOTAL_LIMIT} \
  --output_root "${OUTPUT_ROOT}" \
  --num_train_epochs ${EPOCHS} \
  --learning_rate ${LR} \
  --weight_decay ${WD} \
  --warmup_steps ${WARMUP} \
  --best_metric ${BEST_METRIC} \
  --greater_is_better ${GREATER_IS_BETTER} \
  --per_device_train_batch_size ${BSZ} \
  --per_device_eval_batch_size ${EVAL_BSZ} \
  --gradient_accumulation_steps ${GA} \
  --seed ${SEED} \
  ${FP16_FLAG} \
  $( [[ ${USE_LCAL} -eq 1 ]] && echo "--use_lcal" ) \
  $( [[ ${USE_EARLY_STOPPING} -eq 1 ]] && echo "--use_early_stopping" ) \
  --early_stopping_patience ${EARLY_STOPPING_PATIENCE} \
  --early_stopping_threshold ${EARLY_STOPPING_THRESHOLD} \
  $( [[ ${USE_REPR_UNCERT:-0} -eq 1 ]] && echo "--repr_k_neighbors ${REPR_K_NEIGHBORS} --repr_distance_metric ${REPR_DISTANCE_METRIC} --repr_search_split ${REPR_SEARCH_SPLIT} --repr_score_type ${REPR_SCORE_TYPE} --repr_normalize ${REPR_NORMALIZE}" ) \
  $( [[ -n "${FOLD_IDS:-}" ]] && echo "--fold_ids ${FOLD_IDS}" ) \
  ${WANDB_FLAGS}

echo "Stage-1 finished. Results saved to ${OUTPUT_ROOT}"
