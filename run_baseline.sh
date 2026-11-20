#!/usr/bin/env bash
# Full-data, raw-input continual forecasting. Run from any working directory.
set -euo pipefail
cd "$(dirname "$0")"
if [[ $# -lt 2 || ${1:-} == --help ]]; then
  echo "Usage: bash run_baseline.sh {all|TS|STKEC|PECPM|EAC|STRAP|STBP} {all|PEMS|AIR|CA-S|CA-M|Madrid} [GPU=0] [SEED=42] [trainer options]"
  echo "Prepare predictors with: python continual/install_upstream.py all"
  exit 0
fi
model=${1^^}
dataset=${2^^}
gpu=${3:-0}
seed=${4:-42}
shift "$(( $# < 4 ? $# : 4 ))"
python_bin=${PYTHON_BIN:-python}
case "$model" in
  ALL) models=(TS STKEC PECPM EAC STRAP STBP);;
  TS|STKEC|PECPM|EAC|STRAP|STBP) models=("$model");;
  *) echo "Unknown baseline: $model" >&2; exit 2;;
esac
case "$dataset" in
  ALL) datasets=(PEMS AIR CA_NODE_SHIFT CA_MULTIYEAR_NODE_SHIFT MADRID);;
  CA-S|CA_NODE_SHIFT) datasets=(CA_NODE_SHIFT);;
  CA-M|CA_MULTIYEAR_NODE_SHIFT) datasets=(CA_MULTIYEAR_NODE_SHIFT);;
  MADRID|MADRID_2022_2026_MMD) datasets=(MADRID);;
  PEMS|AIR) datasets=("$dataset");;
  *) echo "Unknown dataset: $dataset" >&2; exit 2;;
esac
for method in "${models[@]}"; do
  for ds in "${datasets[@]}"; do
    if [[ $ds == MADRID && $method == STRAP ]]; then
      echo "Madrid STRAP has no mask-aware training port in the original project." >&2
      if [[ $dataset != ALL && $model != ALL ]]; then exit 2; fi
      continue
    fi
    namespace=${method,,}
    if [[ $method == TS ]]; then namespace=trafficstream; fi
    if [[ ! -d continual/_upstream/$namespace ]]; then
      echo "Install the $method predictor first: $python_bin continual/install_upstream.py $method" >&2
      exit 2
    fi
    batch=128
    lr=0.005
    if [[ $method == STBP && ( $ds == PEMS || $ds == AIR ) ]]; then batch=64; lr=0.01; fi
    opts=(--gpuid "$gpu" --seed "$seed" --epochs 200 --batch-size "$batch" --lr "$lr")
    if [[ $ds == MADRID && $method == STBP ]]; then
      "$python_bin" continual/main_madrid_stbp.py "${opts[@]}" --set patience=30 --normalize-x false --output-dir runs/madrid_stbp/raw "$@"
    elif [[ $ds == MADRID ]]; then
      "$python_bin" continual/main_madrid_continual_baselines.py --model "$method" "${opts[@]}" --patience 30 --variant raw --output-dir runs/madrid_continual "$@"
    else
      "$python_bin" continual/baseline_main.py --model "$method" --dataset "$ds" "${opts[@]}" --set patience=30 --variant raw --output-dir runs/baselines "$@"
    fi
  done
done

