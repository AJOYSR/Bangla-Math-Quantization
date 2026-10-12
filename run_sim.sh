#!/usr/bin/env bash
# Simulated-quantization experiment (runs on Apple Silicon, CUDA or CPU).
# Resumable: finished variants are skipped and interrupted ones continue where they stopped.
#   N=200 bash run_sim.sh            (~12 h on an M1 Pro)
# Runs are ordered so the main comparisons finish first.
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_DISABLE_XET=1 PYTORCH_ENABLE_MPS_FALLBACK=1
PY=.venv/bin/python

N="${N:-200}"

[ -d models/merged-16bit ] || $PY src/merge_lora.py --out models/merged-16bit
[ -f results/fertility.json ] || $PY src/fertility.py

run() { $PY src/simulate.py --n "$N" "$@"; $PY src/report.py > /dev/null; }

run --method 16bit
run --method rtn --bits 4
for calib in bangla english-math english; do
  run --method gptq --bits 4 --calib "$calib" --seed 0
done
run --method rtn --bits 8 --group-size -1
run --method nf4
run --method rtn --bits 3
run --method gptq --bits 3 --calib bangla --seed 0
for seed in 1 2; do
  for calib in bangla english-math english; do
    run --method gptq --bits 4 --calib "$calib" --seed "$seed"
  done
done

$PY src/report.py
bash paper/build.sh
echo "ALL DONE"
