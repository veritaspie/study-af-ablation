#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AI_ECG_ROOT="${AI_ECG_ROOT:-/data/projects/ai-ecg}"
CONFIG_PATH="${STUDY_ROOT}/configs/extract_features_rfca_ssl_i0001_mtae.yaml"

exec "${AI_ECG_ROOT}/scripts/with_ai_ecg_env.sh" \
  python -m cli extract-features --config "${CONFIG_PATH}" "$@"
