#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PARAM_FILE="${SCRIPT_DIR}/params/threshold_eval_params.sh"

if [[ $# -ge 1 ]]; then
  PARAM_FILE="$1"
  echo "[run_stage2_threshold_eval] Using parameter file: ${PARAM_FILE}"
else
  PARAM_FILE="${DEFAULT_PARAM_FILE}"
  echo "[run_stage2_threshold_eval] Using default parameter file: ${PARAM_FILE}"
fi

if [[ ! -f "${PARAM_FILE}" ]]; then
  echo "Error: Parameter file does not exist: ${PARAM_FILE}"
  exit 1
fi

# shellcheck disable=SC1090
source "${PARAM_FILE}"

if [[ -z "${STAGE2_DIR:-}" ]]; then
  echo "Error: STAGE2_DIR must be set in the parameter file, e.g. STAGE2_DIR=\"/path/to/.../Stage2\""
  exit 1
fi

STAGE2_DIR="$(cd "${STAGE2_DIR}" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
THRESHOLDS="${THRESHOLDS:-0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}"
METRICS="${METRICS:-auroc,auprc}"
BATCH_SIZE="${BATCH_SIZE:-512}"

EMBED_ROOT_FLAG=()
if [[ -n "${EMBEDDINGS_ROOT:-}" ]]; then
  EMBED_ROOT_FLAG=(--embeddings_root "${EMBEDDINGS_ROOT}")
fi
EMBED_SOURCE_FLAG=()
if [[ -n "${EMBEDDING_SOURCE:-}" ]]; then
  EMBED_SOURCE_FLAG=(--embedding_source "${EMBEDDING_SOURCE}")
fi
POOLING_FLAG=()
if [[ -n "${POOLING:-}" ]]; then
  POOLING_FLAG=(--pooling "${POOLING}")
fi
DEVICE_FLAG=()
if [[ -n "${DEVICE:-}" ]]; then
  DEVICE_FLAG=(--device "${DEVICE}")
fi

shopt -s nullglob
FOUND=0
for exp_dir in "${STAGE2_DIR}"/Active*; do
  [[ -d "${exp_dir}" ]] || continue
  FOUND=1
  echo "======================================================================="
  echo "[run] Processing experiment: ${exp_dir}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/stage2_threshold_eval.py" \
    --experiment_dir "${exp_dir}" \
    --thresholds "${THRESHOLDS}" \
    --batch_size "${BATCH_SIZE}" \
    --metrics "${METRICS}" \
    "${EMBED_ROOT_FLAG[@]}" \
    "${EMBED_SOURCE_FLAG[@]}" \
    "${POOLING_FLAG[@]}" \
    "${DEVICE_FLAG[@]}"
  echo "[run] Finished and saved threshold evaluation results for ${exp_dir}"
  echo "======================================================================="
done

if [[ ${FOUND} -eq 0 ]]; then
  echo "[Warning] No Active* subdirectories found under ${STAGE2_DIR}."
fi
