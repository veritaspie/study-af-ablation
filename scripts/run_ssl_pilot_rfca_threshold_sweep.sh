#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AI_ECG_ROOT="${AI_ECG_ROOT:-/data/projects/ai-ecg}"
MANIFEST_DIR="${STUDY_ROOT}/manifests"
CONFIG_DIR="${STUDY_ROOT}/configs"

run_in_ai_ecg_env() {
  "${AI_ECG_ROOT}/scripts/with_ai_ecg_env.sh" "$@"
}

#run_in_ai_ecg_env python "${STUDY_ROOT}/scripts/build_finetune_manifest_rfca_zarr.py" \
#  --label-threshold 0 \
#  --label-comparison gt \
#  --output "${MANIFEST_DIR}/finetune_lvr05_high_rfca_th0.parquet"

run_in_ai_ecg_env python "${STUDY_ROOT}/scripts/build_finetune_manifest_rfca_zarr.py" \
  --label-threshold 5 \
  --output "${MANIFEST_DIR}/finetune_lvr05_high_rfca_th5.parquet"

#run_in_ai_ecg_env python -m cli hparam-search \
#  --config "${CONFIG_DIR}/hparam_search_lvr05_high_rfca_ssl_i0001_mtae_th0.yaml"

run_in_ai_ecg_env python -m cli hparam-search \
  --config "${CONFIG_DIR}/hparam_search_lvr05_high_rfca_ssl_i0001_mtae_th5.yaml"
