#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash run_stage2.sh <param_file> [KEY=VALUE ...]"
  echo "Example: bash run_stage2.sh params/Stage2_params.sh"
  exit 1
fi

PARAM_FILE="$1"
shift
EXTRA_OVERRIDES=("$@")
echo "[run_stage2.sh] Using param file: ${PARAM_FILE}"

if [[ ! -f "${PARAM_FILE}" ]]; then
  echo "Error: param file not found: ${PARAM_FILE}"
  exit 1
fi

source "${PARAM_FILE}"

if [[ ${#EXTRA_OVERRIDES[@]} -gt 0 ]]; then
  echo "[run_stage2.sh] Command-line overrides: ${EXTRA_OVERRIDES[*]}"
  for item in "${EXTRA_OVERRIDES[@]}"; do
    if [[ "$item" != *=* ]]; then
      echo "Error: extra arguments must be KEY=VALUE, got: '$item'"
      exit 1
    fi
    key="${item%%=*}"
    value="${item#*=}"
    if [[ -z "${key}" ]]; then
      echo "Error: empty variable name: '$item'"
      exit 1
    fi
    if [[ ! "${key}" =~ ^[A-Z0-9_]+$ ]]; then
      echo "Error: invalid variable name '${key}' (only A-Z, 0-9, _ allowed)"
      exit 1
    fi
    printf "[run_stage2.sh] Override %s=%s\n" "${key}" "${value}"
    eval "${key}=\"${value}\""
  done
fi

export HF_HOME="${CACHE_BASE}"
export HF_HUB_CACHE="${CACHE_BASE}/hub"
export HUGGINGFACE_HUB_CACHE="${CACHE_BASE}/hub"
export HF_DATASETS_CACHE="${CACHE_BASE}/datasets"
export TRANSFORMERS_CACHE="${CACHE_BASE}/models"
export XDG_CACHE_HOME="${CACHE_BASE}/xdg"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME"

if [[ -n "${CUDA_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
  echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

WANDB_FLAGS=""
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  export WANDB_API_KEY
fi
if [[ "${USE_WANDB}" == "1" ]]; then
  export WANDB_PROJECT="${WANDB_PROJECT}"
  export WANDB_ENTITY="${WANDB_ENTITY}"
  export WANDB_MODE=online
  WANDB_FLAGS="--use_wandb --wandb_run_name ${WANDB_RUN_NAME}"
else
  export WANDB_MODE=offline
fi

LAUNCHER="python"

${LAUNCHER} "${SCRIPT_DIR}/stage2_train_from_embeddings.py" \
  --embeddings_root "${EMBEDDINGS_ROOT}" \
  --pooling "${POOLING}" \
  --embedding_source "${EMBEDDING_SOURCE}" \
  --labels ${TARGET_LABELS} \
  --model_name "${MODEL_NAME}" \
  --num_heads ${NUM_HEADS} \
  --head_hidden_dim ${HEAD_HIDDEN} \
  --dropout ${DROPOUT} \
  --lambda_cal ${LAMBDA_CAL} \
  --lambda_pop ${LAMBDA_POP} \
  --pop_k_neighbors ${POP_K_NEIGHBORS} \
  --pop_distance_metric "${POP_DISTANCE_METRIC}" \
  --pop_search_split "${POP_SEARCH_SPLIT}" \
  --cal_alpha ${CAL_ALPHA} \
  --u_min ${U_MIN} \
  --u_max ${U_MAX} \
  --ece_bins ${ECE_BINS} \
  --pos_ece_bins ${POS_ECE_BINS} \
  $( [[ ${F1_FIND_BEST} -eq 1 ]] && echo "--f1_find_best" ) \
  --f1_threshold ${F1_THRESHOLD} \
  --f1_grid_step ${F1_GRID_STEP} \
  --eval_strategy ${EVAL_STRATEGY} \
  --eval_steps ${EVAL_STEPS} \
  --save_strategy ${SAVE_STRATEGY} \
  --save_steps ${SAVE_STEPS} \
  --save_total_limit ${SAVE_TOTAL_LIMIT} \
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
  $( [[ ${USE_EARLY_STOPPING} -eq 1 ]] && echo "--use_early_stopping" ) \
  --early_stopping_patience ${EARLY_STOPPING_PATIENCE} \
  --early_stopping_threshold ${EARLY_STOPPING_THRESHOLD} \
  --test_eval_strategy ${TEST_EVAL_STRATEGY} \
  --test_eval_steps ${TEST_EVAL_STEPS} \
  $( [[ -n "${FOLD_IDS:-}" ]] && echo "--fold_ids ${FOLD_IDS}" ) \
  ${WANDB_FLAGS}

echo "Stage-2 finished. Results saved under $(dirname "${EMBEDDINGS_ROOT}")/Stage2/"
