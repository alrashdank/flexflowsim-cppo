#!/usr/bin/env python3
"""
Paper 6 revision: 2x2 ablation isolating what manufactured the published
V4 failure — the base reward vs the constraint slack signal.

Cells
-----
                     | cumrate slack (published)   | episode slack (corrected)
  shaped base        | shaped-cumrate  == V4       | shaped-episode
  (0.8, 0.1, 0.1)    |   (published control)       |   (isolates slack bias)
  -------------------+-----------------------------+---------------------------
  cost base          | cost-cumrate                | cost-episode
  (1.0, 0.0, 0.0)    |   (isolates shaping)        |   (the CMDP §3.3 claims)

Everything else is held at the published protocol: PPO hyperparameters,
seeds [42, 7, 2024, 123, 999], checkpoint interval 100K, validation seeds
[10000, 10020) with the 16/20 lowest-CPU selection rule, test seeds
[11000, 11050), budgets 1.5M (bakery) / 1.6M (electronics).

Usage
-----
  # decisive overnight pilot on the failing testbed (4 cells x 5 seeds x 400K):
  python run_ablation_2x2.py --testbed electronics --timesteps 400000

  # full protocol budget:
  python run_ablation_2x2.py --testbed electronics

  # single cell / subset of seeds (for parallel launches in separate shells):
  python run_ablation_2x2.py --testbed electronics --cells cost-episode --seeds 42,7

Runs are idempotent: a (cell, seed) whose summary.json exists is skipped, so
you can relaunch or spread cells across terminals freely. Aggregate anytime:
  python run_ablation_2x2.py --testbed electronics --aggregate-only
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from env import FlexFlowSimEnv
from lagrangian_slack import SlackLagrangianFlowEnv

# ============================================================
# Published protocol constants (mirrors run_experiment.py)
# ============================================================

VALIDATION_SEEDS = list(range(10000, 10020))
TEST_SEEDS = list(range(11000, 11050))
SAT_THRESHOLD = 16  # of 20 validation episodes, both constraints
DEFAULT_SEEDS = [42, 7, 2024, 123, 999]

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

CELLS = {
    "shaped-cumrate": {"weights": (0.8, 0.1, 0.1), "slack": "cumrate"},
    "shaped-episode": {"weights": (0.8, 0.1, 0.1), "slack": "episode"},
    "cost-cumrate":   {"weights": (1.0, 0.0, 0.0), "slack": "cumrate"},
    "cost-episode":   {"weights": (1.0, 0.0, 0.0), "slack": "episode"},
}

T_CRIT = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}  # 95% two-sided


# ============================================================
# Evaluation (identical semantics to run_experiment.py)
# ============================================================

def evaluate_episode(model, env, deterministic=True):
    obs, _ = env.reset()
    done, last_info = False, None
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, term, trunc, last_info = env.step(action)
        done = term or trunc
    tc = float(last_info.get("total_cost", 0.0))
    tp = float(last_info.get("total_departed", 0))
    return {"throughput": tp,
            "cost_per_unit": tc / max(tp, 1e-9),
            "total_cost": tc,
            "utilisations": [float(u) for u in last_info.get("utilisation", [])]}


def evaluate_on_seeds(model, cfg, weights, seeds, tb, deterministic=True):
    records = []
    for s in seeds:
        env = FlexFlowSimEnv(config=cfg["config"], weights=weights, seed=int(s))
        r = evaluate_episode(model, env, deterministic=deterministic)
        util = np.array(r["utilisations"])
        r["util_satisfied"] = int(all(util[i] >= cfg["u_min"]
                                      for i in cfg["constrained_servers"]))
        r["tp_satisfied"] = int(r["throughput"] >= cfg["tp_target"])
        r["eval_seed"] = int(s)
        records.append(r)
    return records


def select_best_checkpoint(checkpoint_records):
    """Published rule: among checkpoints with >=16/20 on BOTH constraints,
    lowest validation mean CPU; fallback = lowest mean CPU overall."""
    qualifying, all_ckpts = [], []
    for steps, recs in sorted(checkpoint_records.items()):
        mean_cpu = float(np.mean([r["cost_per_unit"] for r in recs]))
        u = sum(r["util_satisfied"] for r in recs)
        t = sum(r["tp_satisfied"] for r in recs)
        all_ckpts.append((steps, mean_cpu))
        if u >= SAT_THRESHOLD and t >= SAT_THRESHOLD:
            qualifying.append((steps, mean_cpu))
    if qualifying:
        qualifying.sort(key=lambda x: x[1])
        return qualifying[0][0], "satisfied"
    all_ckpts.sort(key=lambda x: x[1])
    return all_ckpts[0][0], "fallback"


# ============================================================
# Training
# ============================================================

def make_wrapped_env(tb, cfg, cell, seed, symmetric):
    base = FlexFlowSimEnv(config=cfg["config"],
                          weights=CELLS[cell]["weights"], seed=seed)
    return SlackLagrangianFlowEnv(
        base, cfg["constrained_servers"],
        util_floor=cfg["u_min"],
        tp_rate_floor=cfg["tp_target"] / 480.0,
        slack_mode=CELLS[cell]["slack"],
        symmetric=symmetric,
    )


def run_cell_seed(tb, cfg, cell, seed, budget, ckpt_freq, outdir, symmetric,
                  stage="full"):
    """stage: 'full' = train+evaluate; 'train' = train only (checkpoints +
    lambda history, no evaluation); 'finish' = evaluate existing checkpoints.
    Useful when each call must fit a wall-clock cap; all stages idempotent."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback

    seed_dir = Path(outdir) / cell / f"seed_{seed}"
    summary_path = seed_dir / "summary.json"
    if summary_path.exists():
        print(f"[skip] {cell} seed {seed} — summary exists")
        return json.loads(summary_path.read_text())

    ckpt_dir = seed_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt = ckpt_dir / f"ckpt_{budget}_steps.zip"
    mins = None

    if stage in ("full", "train") and not final_ckpt.exists():
        env = make_wrapped_env(tb, cfg, cell, seed, symmetric)
        model = PPO("MlpPolicy", env, seed=seed, verbose=0, **PPO_HYPERPARAMS)
        cb = CheckpointCallback(save_freq=ckpt_freq, save_path=str(ckpt_dir),
                                name_prefix="ckpt")
        print(f"\n=== {tb} | {cell} | seed {seed} | {budget:,} steps ===")
        t0 = time.time()
        model.learn(total_timesteps=budget, callback=cb, progress_bar=False)
        mins = (time.time() - t0) / 60.0
        model.save(str(ckpt_dir / f"ckpt_{budget}_steps"))
        print(f"    trained in {mins:.1f} min "
              f"({budget / max(time.time() - t0, 1e-9):,.0f} steps/s)")
        hist = env.lambda_history
        if hist:
            with open(seed_dir / "lambda_history.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
                w.writeheader()
                w.writerows(hist)

    if stage == "train":
        print(f"    [train stage complete: {cell} seed {seed}]")
        return None
    if not final_ckpt.exists():
        print(f"[finish] {cell} seed {seed}: no final checkpoint — "
              f"run --stage train first")
        return None

    # lambda history from disk (works for both stages)
    hist = []
    hp = seed_dir / "lambda_history.csv"
    if hp.exists():
        with open(hp) as f:
            hist = [{k: float(v) for k, v in row.items()}
                    for row in csv.DictReader(f)]

    # validate every checkpoint
    ckpts = sorted({int(p.stem.split("_")[1]) for p in ckpt_dir.glob("ckpt_*_steps.zip")})
    records = {}
    for st in ckpts:
        m = PPO.load(str(ckpt_dir / f"ckpt_{st}_steps"), device="cpu")
        records[st] = evaluate_on_seeds(m, cfg, CELLS[cell]["weights"],
                                        VALIDATION_SEEDS, tb)
        u = sum(r["util_satisfied"] for r in records[st])
        t = sum(r["tp_satisfied"] for r in records[st])
        cpu = np.mean([r["cost_per_unit"] for r in records[st]])
        print(f"    val {st:>9,}: CPU={cpu:8.2f}  Usat={u:2d}/20  Tsat={t:2d}/20")

    sel_steps, status = select_best_checkpoint(records)

    # test the selected checkpoint — deterministic (published protocol) AND
    # stochastic (the policy class Lagrangian training actually optimises;
    # CMDP optima are generically randomised, Altman 1999)
    m = PPO.load(str(ckpt_dir / f"ckpt_{sel_steps}_steps"), device="cpu")
    test = evaluate_on_seeds(m, cfg, CELLS[cell]["weights"], TEST_SEEDS, tb)
    test_s = evaluate_on_seeds(m, cfg, CELLS[cell]["weights"], TEST_SEEDS, tb,
                               deterministic=False)
    with open(seed_dir / "test_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(test[0].keys()))
        w.writeheader()
        w.writerows([{k: (v if not isinstance(v, list) else json.dumps(v))
                      for k, v in r.items()} for r in test])

    lam_tp_final = hist[-1]["lam_tp"] if hist else None
    summary = {
        "testbed": tb, "cell": cell, "seed": seed, "budget": budget,
        "selected_steps": sel_steps, "selection_status": status,
        "test_mean_cpu": float(np.mean([r["cost_per_unit"] for r in test])),
        "test_mean_tp": float(np.mean([r["throughput"] for r in test])),
        "test_util_sat": float(np.mean([r["util_satisfied"] for r in test])),
        "test_tp_sat": float(np.mean([r["tp_satisfied"] for r in test])),
        "stoch_test_mean_cpu": float(np.mean([r["cost_per_unit"] for r in test_s])),
        "stoch_test_mean_tp": float(np.mean([r["throughput"] for r in test_s])),
        "stoch_test_util_sat": float(np.mean([r["util_satisfied"] for r in test_s])),
        "stoch_test_tp_sat": float(np.mean([r["tp_satisfied"] for r in test_s])),
        "lam_tp_final": lam_tp_final,
        "lam_tp_saturated": bool(lam_tp_final is not None
                                 and lam_tp_final >= 0.99 * 20000.0),
        "train_minutes": round(mins, 1) if mins is not None else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"    selected {sel_steps:,} ({status})  "
          f"test CPU={summary['test_mean_cpu']:.2f} TP={summary['test_mean_tp']:.1f}  "
          f"lam_T final={lam_tp_final}")
    return summary


# ============================================================
# Aggregation / verdict
# ============================================================

def shortestqueue_reference(tb, cfg):
    from baselines import ShortestQueuePolicy

    class _Shim:
        def __init__(self, pol):
            self.pol = pol

        def predict(self, obs, deterministic=True):
            a = self.pol.predict(obs)
            return (a[0], None) if isinstance(a, tuple) else (a, None)

    cpus, tps = [], []
    for s in TEST_SEEDS:
        env = FlexFlowSimEnv(config=cfg["config"], weights=(0.8, 0.1, 0.1),
                             seed=int(s))
        r = evaluate_episode(_Shim(ShortestQueuePolicy(env)), env)
        cpus.append(r["cost_per_unit"])
        tps.append(r["throughput"])
    return float(np.mean(cpus)), float(np.mean(tps))


def aggregate(tb, cfg, outdir, cells):
    rows = []
    for cell in cells:
        summaries = []
        for p in sorted((Path(outdir) / cell).glob("seed_*/summary.json")):
            summaries.append(json.loads(p.read_text()))
        if not summaries:
            continue
        cpus = [s["test_mean_cpu"] for s in summaries]
        n = len(cpus)
        ci = T_CRIT.get(n, 1.96) * np.std(cpus, ddof=1) / np.sqrt(n) if n > 1 else 0.0
        scpus = [s["stoch_test_mean_cpu"] for s in summaries
                 if "stoch_test_mean_cpu" in s]
        rows.append({
            "cell": cell, "n_seeds": n,
            "cpu_mean": round(float(np.mean(cpus)), 2),
            "cpu_ci95": round(float(ci), 2),
            "tp_mean": round(float(np.mean([s["test_mean_tp"] for s in summaries])), 2),
            "joint_sat_%": round(100 * float(np.mean(
                [min(s["test_util_sat"], s["test_tp_sat"]) for s in summaries])), 1),
            "stoch_cpu_mean": round(float(np.mean(scpus)), 2) if scpus else None,
            "stoch_joint_sat_%": round(100 * float(np.mean(
                [min(s["stoch_test_util_sat"], s["stoch_test_tp_sat"])
                 for s in summaries if "stoch_test_util_sat" in s])), 1) if scpus else None,
            "seeds_validation_satisfied": sum(
                s["selection_status"] == "satisfied" for s in summaries),
            "lam_T_saturated": sum(s["lam_tp_saturated"] for s in summaries),
        })
    sq_cpu, sq_tp = shortestqueue_reference(tb, cfg)
    print(f"\n{'='*78}\n  VERDICT — {tb}  "
          f"(ShortestQueue reference: CPU=${sq_cpu:.2f}, TP={sq_tp:.1f})\n{'='*78}")
    print(f"  {'cell':<16}{'n':>3} {'argmax CPU±CI':>16} {'TP':>7} "
          f"{'sat%':>6} | {'stoch CPU':>10} {'sat%':>6} | {'val-sat':>8} {'lamT sat':>9}")
    for r in rows:
        sc = f"{r['stoch_cpu_mean']:>10.2f}" if r.get("stoch_cpu_mean") is not None else f"{'—':>10}"
        ss = (f"{r['stoch_joint_sat_%']:>5.1f}%"
              if r.get("stoch_joint_sat_%") is not None else f"{'—':>6}")
        print(f"  {r['cell']:<16}{r['n_seeds']:>3} "
              f"{r['cpu_mean']:>9.2f}±{r['cpu_ci95']:<6.2f} {r['tp_mean']:>7.2f} "
              f"{r['joint_sat_%']:>5.1f}% | {sc} {ss} | "
              f"{r['seeds_validation_satisfied']:>5}/{r['n_seeds']} "
              f"{r['lam_T_saturated']:>6}/{r['n_seeds']}")
    out = Path(outdir) / "ablation_summary.csv"
    if rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n  written: {out}")
    return rows


# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--testbed", choices=list(TESTBED_CONFIG),
                    default="electronics")
    ap.add_argument("--cells", default="all",
                    help="comma list from: " + ",".join(CELLS) + "  (or 'all')")
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--timesteps", type=int, default=None,
                    help="per-run budget (default: published protocol budget)")
    ap.add_argument("--checkpoint-interval", type=int, default=100_000)
    ap.add_argument("--symmetric", action="store_true",
                    help="signed (V6-lite) dual update in episode cells")
    ap.add_argument("--stage", choices=["full", "train", "finish"],
                    default="full",
                    help="split train and evaluation into separate calls")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    cfg = TESTBED_CONFIG[args.testbed]
    budget = args.timesteps or cfg["default_budget"]
    outdir = args.outdir or f"results_ablation/{args.testbed}"
    cells = list(CELLS) if args.cells == "all" else [c.strip() for c in args.cells.split(",")]
    for c in cells:
        if c not in CELLS:
            raise SystemExit(f"unknown cell {c!r}; choose from {list(CELLS)}")
    seeds = [int(s) for s in args.seeds.split(",")]

    if not args.aggregate_only:
        for cell in cells:
            for seed in seeds:
                run_cell_seed(args.testbed, cfg, cell, seed, budget,
                              args.checkpoint_interval, outdir, args.symmetric,
                              stage=args.stage)
    aggregate(args.testbed, cfg, outdir, cells)


if __name__ == "__main__":
    main()
