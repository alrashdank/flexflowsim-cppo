#!/usr/bin/env bash
# Advances up to N archival re-evaluations in parallel (resumable).
# Usage: bash archival_tick.sh [n_parallel]
set -u
NP="${1:-2}"
n=0
for run in v4_electronics v4_bakery v6_electronics v6_bakery; do
  case $run in
    v4_*) src=results/$( [ $run = v4_bakery ] && echo bakery_v4 || echo electronics_v4 );;
    v6_*) src=results/$( [ $run = v6_bakery ] && echo v6_bakery_multiseed || echo v6_electronics_multiseed );;
  esac
  for sd in $src/seed_*; do
    seed=${sd##*_}
    [ -f results_r1/archival/$run/seed_$seed/summary.json ] && continue
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 560 python3 archival_reeval.py \
      --run $run --seed $seed --max-minutes 7.5 2>&1 | grep -E "val |published|paused|skip|missing" \
      | sed "s/^/[$run:$seed] /" &
    n=$((n+1))
    [ "$n" -ge "$NP" ] && break 2
  done
done
[ "$n" -eq 0 ] && { echo "ALL ARCHIVAL RE-EVALUATIONS COMPLETE"; exit 0; }
wait
echo "--- archival: $(find results_r1/archival -name summary.json 2>/dev/null | wc -l)/18 finalised"
