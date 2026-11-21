#!/usr/bin/env bash
# Run the full-data raw protocol. --check-data verifies the input files.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo "Usage: bash run_dmra.sh [all|PEMS|AIR|CA-S|CA-M|Madrid|dataset-list] [GPU=0] [SEED=42] [trainer options]"
    exit 0
fi
dataset="${1:-all}"
gpu="${2:-0}"
seed="${3:-42}"
if (( $# >= 3 )); then shift 3; else shift "$#"; fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

if [[ "${dataset,,}" == all ]]; then
    datasets=(PEMS AIR CA_NODE_SHIFT CA_MULTIYEAR_NODE_SHIFT MADRID)
else
    IFS=',' read -ra datasets <<< "$dataset"
fi
names=()
for name in "${datasets[@]}"; do
    case "${name^^}" in
        PEMS|AIR) name="${name^^}" ;;
        CA-S|CA_NODE_SHIFT) name=CA_NODE_SHIFT ;;
        CA-M|CA_MULTIYEAR_NODE_SHIFT) name=CA_MULTIYEAR_NODE_SHIFT ;;
        MADRID|MADRID_2022_2026_MMD) name=MADRID_2022_2026_MMD ;;
        *) echo "Unknown dataset: $name" >&2; exit 2 ;;
    esac
    names+=("$name")
done
for arg in "$@"; do
    if [[ $arg == --help || $arg == -h ]]; then
        module=dmra.train
        if [[ ${names[0]} == MADRID_2022_2026_MMD ]]; then module=dmra.train_madrid; fi
        exec "${PYTHON_BIN:-python}" -m "$module" --help
    fi
done
"${PYTHON_BIN:-python}" prepare_data.py
for name in "${names[@]}"; do
    module=dmra.train
    if [[ $name == MADRID_2022_2026_MMD ]]; then module=dmra.train_madrid; fi
    "${PYTHON_BIN:-python}" -u -m "$module" --dataset "$name" --gpuid "$gpu" --seed "$seed" --num-workers "${NUM_WORKERS:-4}" "$@"
done
