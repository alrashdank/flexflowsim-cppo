"""
Paper 6 — full experimental pipeline driver.

Trains Violation-Driven Lagrangian PPO (V4) for one method/testbed pair,
across 5 seeds, with checkpoint saving, validation-driven checkpoint
selection, and final test-seed evaluation.

Outputs one CSV per (method, training_seed) with one row per test
evaluation episode, plus an aggregate summary CSV.

USAGE:

  # Bakery, all 5 seeds, V4:
  python run_experiment.py --testbed bakery --method v4 \\
      --output-dir results/bakery_v4

  # Electronics, V4:
  python run_experiment.py --testbed electronics --method v4 \\
      --output-dir results/electronics_v4

  # Sensitivity sweep (single seed, varied thresholds):
  python run_experiment.py --testbed bakery --method v4 \\
      --seeds 42 --t-min 16 --u-min 0.40 \\
      --output-dir results/sensitivity/T16_U40

NOTE:
  Run `protocol.md` review BEFORE starting. This script implements
  the protocol as specified there. Do not modify the checkpoint
  selection rule, seed lists, or evaluation episode counts without
  documenting the deviation.
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

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import FlexFlowSimEnv
from pilot_constrained_v4_auto import AutoLagrangianFlowEnv
from pilot_constrained_v6_auto import (
    PIDLagrangianFlowEnv,
    PID_BAKERY,
    PID_ELECTRONICS,
)

# ============================================================
# Protocol-defined constants — do not change without updating protocol.md
# ============================================================
TRAINING_SEEDS = [42, 7, 2024, 123, 999]
VALIDATION_SEEDS = list(range(10000, 10020))   # 20 episodes for checkpoint selection
TEST_SEEDS = list(range(11000, 11050))         # 50 episodes for final test

CHECKPOINT_INTERVAL = 100_000
SATISFACTION_THRESHOLD = 8  # out of 10 -> using 16/20 in validation

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
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
)


# ============================================================
# Episode-evaluation helpers
# ============================================================

def evaluate_episode(model_or_policy, eval_env, deterministic=True):
    """Run one episode and return aggregated metrics."""
    obs, _ = eval_env.reset()
    done = False
    last_info = None

    while not done:
        if hasattr(model_or_policy, "predict"):
            action, _ = model_or_policy.predict(obs, deterministic=deterministic)
        else:
            action = model_or_policy(obs)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
        last_info = info

    # Cumulative info — total_cost, total_departed, utilisation are cumulative.
    total_cost = float(last_info.get("total_cost", 0.0))
    total_throughput = float(last_info.get("total_departed", 0))
    util = list(last_info.get("utilisation", []))
    cpu = total_cost / max(total_throughput, 1e-9)
    return {
        "throughput": total_throughput,
        "cost_per_unit": float(cpu),
        "total_cost": total_cost,
        "utilisations": [float(u) for u in util],
    }


def evaluate_on_seeds(model, base_env_factory, seeds, constraint_cfg):
    """Evaluate a trained model on a list of seeds. Returns one record per episode."""
    records = []
    for s in seeds:
        env = base_env_factory(seed=int(s))
        result = evaluate_episode(model, env)
        # Constraint satisfaction
        util = np.array(result["utilisations"])
        util_sat = all(util[i] >= constraint_cfg["u_min"]
                       for i in constraint_cfg["constrained_servers"])
        tp_sat = result["throughput"] >= constraint_cfg["tp_target"]
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


# ============================================================
# Checkpoint selection rule (per protocol §7)
# ============================================================

def select_best_checkpoint(checkpoint_records):
    """
    Apply the protocol-defined checkpoint selection rule.

    checkpoint_records: dict[checkpoint_steps -> list[per-episode record]]

    Returns: (selected_steps, selection_status)
        selection_status: "satisfied" | "fallback"
    """
    SAT_THRESHOLD = SATISFACTION_THRESHOLD * len(VALIDATION_SEEDS) / 10  # =16/20

    qualifying = []  # (steps, mean_cpu)
    all_ckpts = []   # (steps, mean_cpu)

    for steps, records in sorted(checkpoint_records.items()):
        mean_cpu = np.mean([r["cost_per_unit"] for r in records])
        util_sat_count = sum(r["util_satisfied"] for r in records)
        tp_sat_count = sum(r["tp_satisfied"] for r in records)
        all_ckpts.append((steps, mean_cpu))
        if util_sat_count >= SAT_THRESHOLD and tp_sat_count >= SAT_THRESHOLD:
            qualifying.append((steps, mean_cpu))

    if qualifying:
        qualifying.sort(key=lambda x: x[1])
        return qualifying[0][0], "satisfied"
    else:
        all_ckpts.sort(key=lambda x: x[1])
        return all_ckpts[0][0], "fallback"


# ============================================================
# Training loop
# ============================================================

def make_env(testbed, testbed_cfg, method, seed,
             t_min_override=None, u_min_override=None):
    """Construct a Lagrangian-wrapped env. Dispatches on method (v4|v6)."""
    base = FlexFlowSimEnv(config=testbed_cfg["config"],
                          weights=WEIGHTS, seed=seed)
    t_min = t_min_override if t_min_override is not None else testbed_cfg["tp_target"]
    u_min = u_min_override if u_min_override is not None else testbed_cfg["u_min"]
    if method == "v6":
        pid_cfg = PID_BAKERY if testbed == "bakery" else PID_ELECTRONICS
        return PIDLagrangianFlowEnv(
            base,
            testbed_cfg["constrained_servers"],
            util_floor=u_min,
            tp_rate_floor=t_min / 480.0,
            pid_config=pid_cfg,
        )
    return AutoLagrangianFlowEnv(
        base,
        testbed_cfg["constrained_servers"],
        util_floor=u_min,
        tp_rate_floor=t_min / 480.0,
    )


def train_one_seed(testbed, method, seed, budget, output_dir,
                   t_min_override=None, u_min_override=None,
                   checkpoint_freq=None):
    """Train one seed, save checkpoints every checkpoint_freq steps."""
    cfg = TESTBED_CONFIG[testbed]
    ckpt_freq = checkpoint_freq if checkpoint_freq is not None else CHECKPOINT_INTERVAL
    seed_dir = Path(output_dir) / f"seed_{seed}"
    ckpt_dir = seed_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(testbed, cfg, method, seed, t_min_override, u_min_override)

    print(f"\n{'='*60}")
    print(f"  Training: testbed={testbed}, method={method}, seed={seed}")
    print(f"  Budget: {budget:,} steps  (checkpoint every {ckpt_freq:,})")
    print(f"  Output: {seed_dir}")
    print(f"{'='*60}\n")

    model = PPO("MlpPolicy", env, seed=seed, verbose=0, **PPO_HYPERPARAMS)

    callback = CheckpointCallback(
        save_freq=ckpt_freq,
        save_path=str(ckpt_dir),
        name_prefix="ckpt",
    )

    t0 = time.time()
    model.learn(total_timesteps=budget, callback=callback, progress_bar=False)
    elapsed = (time.time() - t0) / 60.0
    print(f"  Training complete in {elapsed:.1f} min")

    # Save lambda history (for Figure 3 - mechanism diagnostic)
    try:
        import pandas as pd
        # env is the wrapped env; for SB3 we may need to peel a Monitor or VecEnv off
        wrapper = env
        while wrapper is not None and not hasattr(wrapper, "lambda_history"):
            wrapper = getattr(wrapper, "env", None)
        if wrapper is not None and wrapper.lambda_history:
            pd.DataFrame(wrapper.lambda_history).to_csv(
                seed_dir / "lambda_history.csv", index=False
            )
            print(f"  Wrote lambda_history.csv ({len(wrapper.lambda_history)} episodes)")
        else:
            print("  WARNING: no lambda_history found on wrapper chain")
    except Exception as e:
        print(f"  WARNING: failed to save lambda_history: {e}")

    # Save final
    model.save(str(seed_dir / "final.zip"))
    return seed_dir, ckpt_dir


def validate_checkpoints(testbed, ckpt_dir, t_min_override=None, u_min_override=None):
    """Run validation pass on every checkpoint. Returns dict of records per checkpoint."""
    cfg = TESTBED_CONFIG[testbed]
    constraint_cfg = {
        "tp_target": t_min_override if t_min_override is not None else cfg["tp_target"],
        "u_min": u_min_override if u_min_override is not None else cfg["u_min"],
        "constrained_servers": cfg["constrained_servers"],
    }
    base_env_factory = lambda seed: FlexFlowSimEnv(
        config=cfg["config"], weights=WEIGHTS, seed=seed)

    ckpt_records = {}
    ckpt_files = sorted(Path(ckpt_dir).glob("ckpt_*_steps.zip"))
    if not ckpt_files:
        # SB3 sometimes uses a different naming
        ckpt_files = sorted(Path(ckpt_dir).glob("*.zip"))

    # Fallback: also include the final.zip if it exists alongside ckpt_dir
    final_path = ckpt_dir.parent / "final.zip"
    has_final = final_path.exists()

    print(f"\n  Validating {len(ckpt_files)} checkpoints on {len(VALIDATION_SEEDS)} seeds...")
    for ckpt_path in ckpt_files:
        # Parse step count from filename
        name = ckpt_path.stem
        try:
            steps = int(name.split("_")[1])
        except (IndexError, ValueError):
            continue
        model = PPO.load(str(ckpt_path))
        records = evaluate_on_seeds(model, base_env_factory, VALIDATION_SEEDS, constraint_cfg)
        ckpt_records[steps] = records
        mean_cpu = np.mean([r["cost_per_unit"] for r in records])
        sat_count = sum(r["util_satisfied"] and r["tp_satisfied"] for r in records)
        print(f"    {steps:>8,} steps: CPU={mean_cpu:7.2f}  joint sat={sat_count}/{len(VALIDATION_SEEDS)}")

    # Fallback: if no intermediate checkpoints exist, use final.zip (only for very short test runs)
    if not ckpt_records and has_final:
        model = PPO.load(str(final_path))
        records = evaluate_on_seeds(model, base_env_factory, VALIDATION_SEEDS, constraint_cfg)
        ckpt_records[-1] = records  # use -1 to mark "final" as a special key
        mean_cpu = np.mean([r["cost_per_unit"] for r in records])
        sat_count = sum(r["util_satisfied"] and r["tp_satisfied"] for r in records)
        print(f"    final (fallback): CPU={mean_cpu:7.2f}  joint sat={sat_count}/{len(VALIDATION_SEEDS)}")

    return ckpt_records


def test_evaluate(model, testbed, t_min_override=None, u_min_override=None):
    """Evaluate model on test seeds. Returns list of per-episode records."""
    cfg = TESTBED_CONFIG[testbed]
    constraint_cfg = {
        "tp_target": t_min_override if t_min_override is not None else cfg["tp_target"],
        "u_min": u_min_override if u_min_override is not None else cfg["u_min"],
        "constrained_servers": cfg["constrained_servers"],
    }
    base_env_factory = lambda seed: FlexFlowSimEnv(
        config=cfg["config"], weights=WEIGHTS, seed=seed)
    return evaluate_on_seeds(model, base_env_factory, TEST_SEEDS, constraint_cfg)


def write_csv(records, path, method, training_seed, checkpoint_steps, status):
    """Write per-episode records to CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    n_servers = len(records[0]["utilisations"])

    fieldnames = (
        ["method", "training_seed", "checkpoint_steps", "selection_status",
         "eval_seed", "throughput", "cost_per_unit", "total_cost",
         "util_satisfied", "tp_satisfied"]
        + [f"util_{i}" for i in range(n_servers)]
    )

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            row = {
                "method": method,
                "training_seed": training_seed,
                "checkpoint_steps": checkpoint_steps,
                "selection_status": status,
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
    print(f"  Wrote {len(records)} test records to {path}")


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--testbed", choices=["bakery", "electronics"], required=True)
    ap.add_argument("--method", choices=["v4", "v6"], default="v4")
    ap.add_argument("--checkpoint-freq", type=int, default=None,
                    help=f"Override checkpoint frequency (default: {CHECKPOINT_INTERVAL:,})")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="Override training seeds (default: protocol seeds)")
    ap.add_argument("--budget", type=int, default=None,
                    help="Override training budget (default: testbed default)")
    ap.add_argument("--t-min", type=float, default=None,
                    help="Override T_min for sensitivity sweep")
    ap.add_argument("--u-min", type=float, default=None,
                    help="Override U_min for sensitivity sweep")
    ap.add_argument("--skip-train", action="store_true",
                    help="Skip training; just run validation+test on existing checkpoints")
    args = ap.parse_args()

    cfg = TESTBED_CONFIG[args.testbed]
    seeds = args.seeds if args.seeds is not None else TRAINING_SEEDS
    budget = args.budget if args.budget is not None else cfg["default_budget"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    method_label = f"{args.method}"
    if args.t_min is not None or args.u_min is not None:
        method_label += f"_T{args.t_min or cfg['tp_target']}_U{args.u_min or cfg['u_min']}"

    summary_rows = []

    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"
        ckpt_dir = seed_dir / "checkpoints"

        if not args.skip_train:
            seed_dir, ckpt_dir = train_one_seed(
                args.testbed, args.method, seed, budget, output_dir,
                t_min_override=args.t_min, u_min_override=args.u_min,
                checkpoint_freq=args.checkpoint_freq,
            )

        # Validation
        ckpt_records = validate_checkpoints(
            args.testbed, ckpt_dir,
            t_min_override=args.t_min, u_min_override=args.u_min,
        )
        if not ckpt_records:
            print(f"  WARNING: no checkpoints found for seed {seed}")
            continue

        # Selection
        selected_steps, status = select_best_checkpoint(ckpt_records)
        print(f"\n  Selected checkpoint: {selected_steps:,} steps  (status: {status})")

        # Load and test
        if selected_steps == -1:
            ckpt_path = seed_dir / "final.zip"
        else:
            ckpt_path = ckpt_dir / f"ckpt_{selected_steps}_steps.zip"
            if not ckpt_path.exists():
                ckpt_path = list(ckpt_dir.glob(f"*{selected_steps}*.zip"))[0]
        model = PPO.load(str(ckpt_path))

        test_records = test_evaluate(
            model, args.testbed,
            t_min_override=args.t_min, u_min_override=args.u_min,
        )

        # Write CSV
        csv_path = output_dir / f"results_seed_{seed}.csv"
        write_csv(test_records, csv_path, method_label, seed, selected_steps, status)

        # Summary
        cpus = [r["cost_per_unit"] for r in test_records]
        tps = [r["throughput"] for r in test_records]
        util_sat = sum(r["util_satisfied"] for r in test_records)
        tp_sat = sum(r["tp_satisfied"] for r in test_records)
        summary_rows.append({
            "method": method_label,
            "seed": seed,
            "selected_checkpoint": selected_steps,
            "selection_status": status,
            "test_episodes": len(test_records),
            "mean_throughput": np.mean(tps),
            "mean_cpu": np.mean(cpus),
            "std_cpu": np.std(cpus),
            "util_sat_rate": util_sat / len(test_records),
            "tp_sat_rate": tp_sat / len(test_records),
        })

    # Aggregate summary
    summary_path = output_dir / "summary.csv"
    if summary_rows:
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\nSummary written to {summary_path}")

        # Console summary
        print(f"\n{'='*60}")
        print(f"  {method_label}, {args.testbed}: {len(summary_rows)} seeds completed")
        print(f"{'='*60}")
        cpus = [r["mean_cpu"] for r in summary_rows]
        tps = [r["mean_throughput"] for r in summary_rows]
        if len(cpus) >= 2:
            mean_cpu = np.mean(cpus)
            ci95 = 1.96 * np.std(cpus, ddof=1) / np.sqrt(len(cpus))
            mean_tp = np.mean(tps)
            ci95_tp = 1.96 * np.std(tps, ddof=1) / np.sqrt(len(tps))
            print(f"  CPU:        {mean_cpu:7.2f}  95% CI: ±{ci95:.2f}")
            print(f"  Throughput: {mean_tp:7.2f}  95% CI: ±{ci95_tp:.2f}")
        else:
            print(f"  CPU:        {cpus[0]:7.2f}  (n=1, no CI)")
            print(f"  Throughput: {tps[0]:7.2f}  (n=1, no CI)")


if __name__ == "__main__":
    main()
