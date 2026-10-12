#!/usr/bin/env bash
# Full experiment: merge -> quantize -> evaluate every variant -> report.
# Quick pass:  N=50 SEEDS="0" bash run_all.sh
# Paper runs:  N=500 SEEDS="0 1 2" bash run_all.sh   (several hours on a T4; run in pieces on Colab)
set -euo pipefail
cd "$(dirname "$0")"

N="${N:-200}"
SEEDS="${SEEDS:-0 1 2}"

[ -d models/merged-16bit ] || python src/merge_lora.py --out models/merged-16bit
python src/fertility.py

eval_once() {  # skip variants that already have results, so a disconnected session can resume
  [ -f "results/$1.json" ] || python src/evaluate.py --n "$N" --variant "$1" ${2:+--path "$2"}
}

eval_once 16bit
eval_once bnb-int8
eval_once bnb-nf4

for seed in $SEEDS; do
  for spec in "4 bangla" "4 english-math" "4 english" "3 bangla"; do
    set -- $spec
    name="gptq-w$1-$2-s$seed"
    [ -d "models/$name" ] || python src/quantize_gptq.py --bits "$1" --calib "$2" --seed "$seed" --out "models/$name"
    eval_once "$name" "models/$name"
  done
done

python src/report.py
