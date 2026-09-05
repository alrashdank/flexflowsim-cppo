#!/usr/bin/env bash
# One unit of R2 (symmetric, reward-scaled) full-budget work: advances up to 2 (cell, seed) runs in
# parallel by one step each — a 400K training segment if training is
# incomplete, otherwise a (resumable, cached) validation/test finish.
# Usage: bash r1_tick.sh <testbed> [n_parallel]
set -u
TB="${1:-electronics}"
NP="${2:-2}"
OUT="results_r2/${TB}"
CELLS="cost-episode-sym"
SEEDS="42 7 2024 123 999"
BUDGET=$([ "$TB" = "bakery" ] && echo 1500000 || echo 1600000)

launch() {  # $1=cell $2=seed
  local cell="$1" seed="$2" dir="$OUT/$1/seed_$2"
  local stage="train"
  [ -f "$dir/checkpoints/ckpt_${BUDGET}_steps.zip" ] && stage="finish"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 540 python3 run_ablation_2x2.py \
    --testbed "$TB" --cells "$cell" --seeds "$seed" \
    --outdir "$OUT" --segment-steps 400000 --stage "$stage" \
    2>&1 | grep -E "segment done|val |selected|train (complete|segment)" \
          | sed "s/^/[$cell:$seed] /"
}

n=0
for cell in $CELLS; do
  for seed in $SEEDS; do
    [ -f "$OUT/$cell/seed_$seed/summary.json" ] && continue
    launch "$cell" "$seed" &
    n=$((n+1))
    [ "$n" -ge "$NP" ] && break 2
  done
done
[ "$n" -eq 0 ] && { echo "ALL RUNS COMPLETE for $TB"; exit 0; }
wait
done_n=$(find "$OUT" -name summary.json 2>/dev/null | wc -l)
echo "--- R2 $TB: $done_n/5 runs finalised"
