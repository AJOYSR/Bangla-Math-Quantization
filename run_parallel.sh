#!/usr/bin/env bash
# Parallel-calibration experiment: the same matched problems in Bangla and in English.
# Waits for any running simulation (e.g. the noise baseline) so runs don't compete for the GPU.
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_DISABLE_XET=1 PYTORCH_ENABLE_MPS_FALLBACK=1
PY=.venv/bin/python

while pgrep -f "src/simulate.py" > /dev/null; do sleep 60; done
[ -f results/english_parallel.json ] || $PY src/match_english.py

for seed in 0 1 2; do
  for calib in bangla-parallel english-parallel; do
    $PY src/simulate.py --n 200 --method gptq --bits 4 --calib "$calib" --seed "$seed"
  done
done
echo "PARALLEL DONE"
