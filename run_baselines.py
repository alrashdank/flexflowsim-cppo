"""
Paper 6 — baseline evaluation script.

Runs all dispatching-rule baselines (LeastUtilised, ShortestQueue,
CostMinimising, FastServerFirst) on a testbed with 50 evaluation
episodes on test seeds [11000-11049].

Output: one CSV per (baseline, testbed) with one row per episode.

USAGE:
  python run_baselines.py --testbed bakery --output-dir results/baselines/bakery
  python run_baselines.py --testbed electronics --output-dir results/baselines/electronics
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import FlexFlowSimEnv
from baselines import (
    LeastUtilisedPolicy, ShortestQueuePolicy,
    CostMinimisingPolicy, FastServerFirstPolicy,
)

# ============================================================
# Protocol-defined constants (must match run_experiment.py)
# ============================================================
TEST_SEEDS = list(range(11000, 11050))
WEIGHTS = (0.8, 0.1, 0.1)

TESTBED_CONFIG = {
    "bakery":      {"config": "configs/bakery_bk50.json",
                    "tp_target": 18, "u_min": 0.50,
                    "constrained_servers": [0, 2]},
    "electronics": {"config": "configs/electronics_3stage.json",
                    "tp_target": 50, "u_min": 0.50,
                    "constrained_servers": [0, 2]},
}

BASELINE_POLICIES = {
    "LeastUtilised":   LeastUtilisedPolicy,
    "ShortestQueue":   ShortestQueuePolicy,
    "CostMinimising":  CostMinimisingPolicy,
    "FastServerFirst": FastServerFirstPolicy,
}


def run_episode(policy, env):
    """Run one episode under a deterministic policy."""
    obs, _ = env.reset()
    policy.env = env  # ensure policy has env reference
    if hasattr(policy, "reset"):
        policy.reset()

    done = False
    last_info = None
    while not done:
        action = policy.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        last_info = info

    total_cost = float(last_info.get("total_cost", 0.0))
    total_throughput = float(last_info.get("total_departed", 0))
    util = list(last_info.get("utilisation", []))
    return {
        "throughput": total_throughput,
        "cost_per_unit": float(total_cost / max(total_throughput, 1e-9)),
        "total_cost": total_cost,
        "utilisations": [float(u) for u in util],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--testbed", choices=["bakery", "electronics"], required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    cfg = TESTBED_CONFIG[args.testbed]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for name, PolicyClass in BASELINE_POLICIES.items():
        print(f"\n{'='*60}")
        print(f"  Baseline: {name} on {args.testbed}")
        print(f"{'='*60}")

        records = []
        for seed in TEST_SEEDS:
            env = FlexFlowSimEnv(config=cfg["config"], weights=WEIGHTS, seed=seed)
            policy = PolicyClass(env=env)
            result = run_episode(policy, env)

            util = np.array(result["utilisations"])
            util_sat = all(util[i] >= cfg["u_min"]
                           for i in cfg["constrained_servers"])
            tp_sat = result["throughput"] >= cfg["tp_target"]

            records.append({
                "method": name,
                "training_seed": -1,
                "checkpoint_steps": -1,
                "selection_status": "baseline",
                "eval_seed": seed,
                "throughput": result["throughput"],
                "cost_per_unit": result["cost_per_unit"],
                "total_cost": result["total_cost"],
                "util_satisfied": int(util_sat),
                "tp_satisfied": int(tp_sat),
                "utilisations": result["utilisations"],
            })

        # Write per-baseline CSV
        if not records:
            continue
        n_servers = len(records[0]["utilisations"])
        fieldnames = (
            ["method", "training_seed", "checkpoint_steps", "selection_status",
             "eval_seed", "throughput", "cost_per_unit", "total_cost",
             "util_satisfied", "tp_satisfied"]
            + [f"util_{i}" for i in range(n_servers)]
        )
        csv_path = output_dir / f"{name}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in records:
                row = {k: r[k] for k in fieldnames if k in r}
                for i, u in enumerate(r["utilisations"]):
                    row[f"util_{i}"] = u
                w.writerow(row)

        cpus = [r["cost_per_unit"] for r in records]
        tps = [r["throughput"] for r in records]
        util_sat_count = sum(r["util_satisfied"] for r in records)
        tp_sat_count = sum(r["tp_satisfied"] for r in records)

        summary_rows.append({
            "method": name,
            "n_episodes": len(records),
            "mean_throughput": np.mean(tps),
            "ci95_throughput": 1.96 * np.std(tps, ddof=1) / np.sqrt(len(tps)),
            "mean_cpu": np.mean(cpus),
            "ci95_cpu": 1.96 * np.std(cpus, ddof=1) / np.sqrt(len(cpus)),
            "util_sat_rate": util_sat_count / len(records),
            "tp_sat_rate": tp_sat_count / len(records),
        })

        print(f"  TP:  {np.mean(tps):6.2f}  CI±{1.96*np.std(tps,ddof=1)/np.sqrt(len(tps)):.2f}")
        print(f"  CPU: ${np.mean(cpus):6.2f}  CI±{1.96*np.std(cpus,ddof=1)/np.sqrt(len(cpus)):.2f}")
        print(f"  UtilSat: {util_sat_count}/{len(records)}   TpSat: {tp_sat_count}/{len(records)}")

    # Aggregate summary
    summary_path = output_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
