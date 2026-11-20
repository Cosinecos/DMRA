#!/usr/bin/env bash
# bash run_ablation.sh ABLATION DATASET raw|unified|both GPU SEED [options]
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
ablation="${1:-all}"; if (( $# )); then shift; fi
dataset="${1:-all}"; if (( $# )); then shift; fi
variant="${1:-raw}"; if (( $# )); then shift; fi
gpu="${1:-0}"; if (( $# )); then shift; fi
seed="${1:-42}"; if (( $# )); then shift; fi
exec "${PYTHON_BIN:-python}" -u -m ablations.train \
  --ablation "$ablation" --dataset "$dataset" --variant "$variant" \
  --gpuid "$gpu" --seed "$seed" --num-workers "${NUM_WORKERS:-4}" "$@"
