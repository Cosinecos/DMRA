#!/usr/bin/env bash
# Run the conventional forecasting baselines in the full-data comparison.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo "Usage: bash run_forecast.sh [MODEL=all] [DATASET=all] [GPU=0] [SEED=42] [trainer options]"
    exec "${PYTHON_BIN:-python}" -m forecast --help
fi
model="${1:-all}"
dataset="${2:-all}"
gpu="${3:-0}"
seed="${4:-42}"
if (( $# >= 4 )); then shift 4; else shift "$#"; fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
workers=4
if [[ ${OS:-} == Windows_NT ]]; then workers=0; fi
exec "${PYTHON_BIN:-python}" -u -m forecast --model "$model" --dataset "$dataset" \
    --gpuid "$gpu" --seed "$seed" --num-workers "${NUM_WORKERS:-$workers}" "$@"
