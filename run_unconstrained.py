"""
Paper 6 — Unconstrained PPO baseline (method M5).

Trains standard PPO with scalarised reward (no Lagrangian wrapper)
across 5 seeds and the same protocol as run_experiment.py for fair comparison.

USAGE:
  python run_unconstrained.py --testbed bakery --output-dir results/bakery_unconstrained
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import FlexFlowSimEnv

# Same constants as run_experiment.py
TRAINING_SEEDS = [42, 7, 2024, 123, 999]
VALIDATION_SEEDS = list(range(10000, 10020))
TEST_SEEDS = list(range(11000, 11050))

CHECKPOINT_INTERVAL = 100_000
WEIGHTS = (0.8, 0.1, 0.1)

TESTBED_CONFIG = {
    "bakery":      {"config": "configs/bakery_bk50.json",
                    "tp_target": 18, "u_min": 0.50,
                    "constrained_servers": [0, 2],
                    "default_budget": 1_500_000},
    "electronics": {"config": "configs/electronics_3stage.json",
                    "tp_target": 50, "u_min": 0.50,
                    "constrained_servers": [0, 2],
                    "default_budget": 1_600_000},
}

PPO_HYPERPARAMS = dict(
    learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
)


def evaluate_episode(model, env):
    obs, _ = env.reset()
    done = False
    last_info = None
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
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


def evaluate_on_seeds(model, cfg, seeds):
    records = []
    for s in seeds:
        env = FlexFlowSimEnv(config=cfg["config"], weights=WEIGHTS, seed=int(s))
        result = evaluate_episode(model, env)
        util = np.array(result["utilisations"])
        util_sat = all(util[i] >= cfg["u_min"] for i in cfg["constrained_servers"])
        tp_sat = result["throughput"] >= cfg["tp_target"]
        records.append({
            "eval_seed": int(s),
            "throughput": result["throughput"],
            "cost_per_unit": result["cost_per_unit"],
            "total_cost": result["total_cost"],
            "utilisations": result["utilisations"],
            "util_satisfied": int(util_sat),
            "tp_satisfied": int(tp_sat),
        })
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testbed", choices=["bakery", "electronics"], required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--budget", type=int, default=None)
    args = ap.parse_args()

    cfg = TESTBED_CONFIG[args.testbed]
    seeds = args.seeds or TRAINING_SEEDS
    budget = args.budget or cfg["default_budget"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"
        ckpt_dir = seed_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        env = FlexFlowSimEnv(config=cfg["config"], weights=WEIGHTS, seed=seed)
        model = PPO("MlpPolicy", env, seed=seed, verbose=0, **PPO_HYPERPARAMS)

        cb = CheckpointCallback(save_freq=CHECKPOINT_INTERVAL,
                                save_path=str(ckpt_dir),
                                name_prefix="ckpt")
        print(f"\nTraining unconstrained PPO, seed {seed}, {budget:,} steps...")
        t0 = time.time()
        model.learn(total_timesteps=budget, callback=cb, progress_bar=False)
        print(f"  Done in {(time.time()-t0)/60:.1f} min")
        model.save(str(seed_dir / "final.zip"))

        # Validation
        print(f"  Validating checkpoints...")
        ckpt_files = sorted(Path(ckpt_dir).glob("ckpt_*_steps.zip"))
        ckpt_records = {}
        for ckpt_path in ckpt_files:
            try:
                steps = int(ckpt_path.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            m = PPO.load(str(ckpt_path))
            ckpt_records[steps] = evaluate_on_seeds(m, cfg, VALIDATION_SEEDS)

        if not ckpt_records:
            print(f"  WARNING: no checkpoints found for seed {seed}")
            continue

        # Selection: lowest CPU among validation (no constraint requirement
        # for unconstrained baseline - it has no notion of constraint)
        all_ckpts = []
        for steps, records in sorted(ckpt_records.items()):
            mean_cpu = np.mean([r["cost_per_unit"] for r in records])
            all_ckpts.append((steps, mean_cpu))
        all_ckpts.sort(key=lambda x: x[1])
        selected_steps = all_ckpts[0][0]
        print(f"  Selected: {selected_steps:,} steps")

        # Test
        ckpt_path = ckpt_dir / f"ckpt_{selected_steps}_steps.zip"
        if not ckpt_path.exists():
            ckpt_path = list(ckpt_dir.glob(f"*{selected_steps}*.zip"))[0]
        m = PPO.load(str(ckpt_path))
        test_records = evaluate_on_seeds(m, cfg, TEST_SEEDS)

        # CSV
        n_servers = len(test_records[0]["utilisations"])
        fieldnames = (
            ["method", "training_seed", "checkpoint_steps", "selection_status",
             "eval_seed", "throughput", "cost_per_unit", "total_cost",
             "util_satisfied", "tp_satisfied"]
            + [f"util_{i}" for i in range(n_servers)]
        )
        csv_path = output_dir / f"results_seed_{seed}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in test_records:
                row = {
                    "method": "UnconstrainedPPO",
                    "training_seed": seed,
                    "checkpoint_steps": selected_steps,
                    "selection_status": "best_cpu_no_constraints",
                    "eval_seed": r["eval_seed"],
                    "throughput": r["throughput"],
                    "cost_per_unit": r["cost_per_unit"],
                    "total_cost": r["total_cost"],
                    "util_satisfied": r["util_satisfied"],
                    "tp_satisfied": r["tp_satisfied"],
                }
                for i, u in enumerate(r["utilisations"]):
                    row[f"util_{i}"] = u
                w.writerow(row)

        cpus = [r["cost_per_unit"] for r in test_records]
        tps = [r["throughput"] for r in test_records]
        summary_rows.append({
            "method": "UnconstrainedPPO",
            "seed": seed,
            "selected_checkpoint": selected_steps,
            "test_episodes": len(test_records),
            "mean_throughput": np.mean(tps),
            "mean_cpu": np.mean(cpus),
            "std_cpu": np.std(cpus),
            "util_sat_rate": sum(r["util_satisfied"] for r in test_records) / len(test_records),
            "tp_sat_rate": sum(r["tp_satisfied"] for r in test_records) / len(test_records),
        })

    if summary_rows:
        with open(output_dir / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            w.writeheader()
            w.writerows(summary_rows)
        cpus = [r["mean_cpu"] for r in summary_rows]
        if len(cpus) >= 2:
            ci = 1.96 * np.std(cpus, ddof=1) / np.sqrt(len(cpus))
            print(f"\nUnconstrained PPO CPU across seeds: {np.mean(cpus):.2f} ± {ci:.2f} (95% CI)")


if __name__ == "__main__":
    main()
