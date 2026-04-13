#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash run_stage1_repr_uncert.sh <param_file>"
  echo "Example: bash run_stage1_repr_uncert.sh params/Stage1_repr_uncert_params.sh"
  exit 1
fi

PARAM_FILE="$1"
echo "[run_stage1_repr_uncert.sh] Using param file: ${PARAM_FILE}"

if [[ ! -f "${PARAM_FILE}" ]]; then
  echo "Error: param file not found: ${PARAM_FILE}"
  exit 1
fi

source "${PARAM_FILE}"

# Optional: pass through CUDA/cache settings (low compute; CPU usually suffices)
if [[ -n "${CUDA_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
  echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

if [[ -n "${CACHE_BASE:-}" ]]; then
  export HF_HOME="${CACHE_BASE}"
  export HF_HUB_CACHE="${CACHE_BASE}/hub"
  export HUGGINGFACE_HUB_CACHE="${CACHE_BASE}/hub"
  export HF_DATASETS_CACHE="${CACHE_BASE}/datasets"
  export TRANSFORMERS_CACHE="${CACHE_BASE}/models"
  export XDG_CACHE_HOME="${CACHE_BASE}/xdg"
fi

python "${SCRIPT_DIR}/stage1_compute_repr_uncert.py" \
  --stage1_label_dir "${STAGE1_LABEL_DIR}" \
  --k_neighbors ${REPR_K_NEIGHBORS} \
  --distance_metric "${REPR_DISTANCE_METRIC}" \
  --search_split "${REPR_SEARCH_SPLIT}" \
  --score_type "${REPR_SCORE_TYPE}" \
  $( [[ ${REPR_NORMALIZE} -eq 1 ]] && echo "--normalize" || echo "--no-normalize" )

echo "Done. Results saved under each fold's embeddings_last_llm_uncertain/."
