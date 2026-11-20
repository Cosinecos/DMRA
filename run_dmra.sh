#!/usr/bin/env bash
# Run the full-data raw protocol. --check-data verifies the input files.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
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
for name in "${datasets[@]}"; do
    module=dmra.train
    case "${name^^}" in
        PEMS|AIR) name="${name^^}" ;;
        CA-S|CA_NODE_SHIFT) name=CA_NODE_SHIFT ;;
        CA-M|CA_MULTIYEAR_NODE_SHIFT) name=CA_MULTIYEAR_NODE_SHIFT ;;
        MADRID|MADRID_2022_2026_MMD) name=MADRID_2022_2026_MMD; module=dmra.train_madrid ;;
        *) echo "Unknown dataset: $name" >&2; exit 2 ;;
    esac
    "${PYTHON_BIN:-python}" -u -m "$module" --dataset "$name" --gpuid "$gpu" --seed "$seed" --num-workers "${NUM_WORKERS:-4}" "$@"
done
