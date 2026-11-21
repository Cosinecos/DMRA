#!/usr/bin/env bash
# bash run_ablation.sh ABLATION DATASET raw|unified|both GPU SEED [options]
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  echo "Usage: bash run_ablation.sh [ABLATION=all] [DATASET=all] [raw|unified|both] [GPU=0] [SEED=42] [trainer options]"
  exit 0
fi
ablation="${1:-all}"; if (( $# )); then shift; fi
dataset="${1:-all}"; if (( $# )); then shift; fi
variant="${1:-raw}"; if (( $# )); then shift; fi
gpu="${1:-0}"; if (( $# )); then shift; fi
seed="${1:-42}"; if (( $# )); then shift; fi
python_bin=${PYTHON_BIN:-python}
args=(--ablation "$ablation" --dataset "$dataset" --variant "$variant"
      --gpuid "$gpu" --seed "$seed" --num-workers "${NUM_WORKERS:-4}" "$@")
for arg in "$@"; do
  if [[ $arg == --help || $arg == -h || $arg == --plan ]]; then
    exec "$python_bin" -u -m ablations.train "${args[@]}"
  fi
done
"$python_bin" prepare_data.py
exec "$python_bin" -u -m ablations.train "${args[@]}"
