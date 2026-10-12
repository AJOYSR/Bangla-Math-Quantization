#!/usr/bin/env bash
# Show how far run_sim.sh has got.   bash progress.sh
cd "$(dirname "$0")"
N="${N:-200}"
VARIANTS=(16bit rtn-w4 gptq-w4-bangla-s0 gptq-w4-english-math-s0 gptq-w4-english-s0 rtn-w8 nf4 rtn-w3
          gptq-w3-bangla-s0 gptq-w4-bangla-s1 gptq-w4-english-math-s1 gptq-w4-english-s1
          gptq-w4-bangla-s2 gptq-w4-english-math-s2 gptq-w4-english-s2
          noise/16bit-b8
          gptq-w4-bangla-parallel-s0 gptq-w4-english-parallel-s0 gptq-w4-bangla-parallel-s1
          gptq-w4-english-parallel-s1 gptq-w4-bangla-parallel-s2 gptq-w4-english-parallel-s2)

done_problems=0
echo "Variant                     Status"
echo "--------------------------  ------------------------------"
for v in "${VARIANTS[@]}"; do
  if [ -f "results/$v.json" ]; then
    acc=$(python3 -c "import json; print(f\"{json.load(open('results/$v.json'))['accuracy']:.1%}\")")
    printf "%-26s  done, accuracy %s (raw; report.py re-scores)\n" "$v" "$acc"
    done_problems=$((done_problems + N))
  elif [ -f "results/$v.partial.jsonl" ]; then
    k=$(wc -l < "results/$v.partial.jsonl" | tr -d ' ')
    printf "%-26s  RUNNING: %s/%s problems\n" "$v" "$k" "$N"
    done_problems=$((done_problems + k))
  else
    printf "%-26s  waiting\n" "$v"
  fi
done

total=$(( ${#VARIANTS[@]} * N ))
echo
echo "Overall: $done_problems / $total problems = $(( 100 * done_problems / total ))% done"
if pgrep -f "run_sim.sh|run_parallel.sh|simulate.py" > /dev/null; then
  echo "Experiment process: running"
elif grep -q "PARALLEL DONE" logs/parallel.log 2>/dev/null; then
  echo "Experiment process: FINISHED"
else
  echo "Experiment process: NOT running (restart with: nohup caffeinate -i bash run_parallel.sh > logs/parallel.log 2>&1 &)"
fi
