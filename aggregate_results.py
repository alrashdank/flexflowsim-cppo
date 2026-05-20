"""
Paper 6 — aggregate all per-method / per-seed CSVs into final results.

Reads:
  results/bakery_v4/results_seed_*.csv
  results/bakery_unconstrained/results_seed_*.csv
  results/baselines/bakery/*.csv
  results/electronics_v4/results_seed_*.csv
  results/baselines/electronics/*.csv

Writes:
  final_results_bakery.csv
  final_results_electronics.csv
  final_sensitivity.csv

Each row: (method, mean_throughput, ci95_throughput, mean_cpu, ci95_cpu,
           mean_total_cost, util_sat_rate, tp_sat_rate, n_seeds, n_episodes_total)

USAGE:
  python aggregate_results.py --root-dir results
"""

import argparse
import csv
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def aggregate_method_seeds(method_dir):
    """Read all per-seed CSVs in a method directory and return one row of aggregates."""
    csvs = sorted(Path(method_dir).glob("results_seed_*.csv"))
    if not csvs:
        return None

    seed_means = []
    seed_data = []
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        seed = df["training_seed"].iloc[0]
        method = df["method"].iloc[0]
        seed_means.append({
            "method": method,
            "seed": int(seed),
            "checkpoint": int(df["checkpoint_steps"].iloc[0]),
            "selection_status": df["selection_status"].iloc[0],
            "n_episodes": len(df),
            "throughput": df["throughput"].mean(),
            "cost_per_unit": df["cost_per_unit"].mean(),
            "total_cost": df["total_cost"].mean(),
            "util_sat_rate": df["util_satisfied"].mean(),
            "tp_sat_rate": df["tp_satisfied"].mean(),
        })
        seed_data.append(df)

    df_means = pd.DataFrame(seed_means)
    method = df_means["method"].iloc[0]
    n_seeds = len(df_means)

    def ci95(values):
        if len(values) < 2:
            return float("nan")
        return 1.96 * np.std(values, ddof=1) / np.sqrt(len(values))

    return {
        "method": method,
        "n_seeds": n_seeds,
        "mean_throughput": df_means["throughput"].mean(),
        "ci95_throughput": ci95(df_means["throughput"].values),
        "mean_cpu": df_means["cost_per_unit"].mean(),
        "ci95_cpu": ci95(df_means["cost_per_unit"].values),
        "mean_total_cost": df_means["total_cost"].mean(),
        "ci95_total_cost": ci95(df_means["total_cost"].values),
        "util_sat_rate": df_means["util_sat_rate"].mean(),
        "tp_sat_rate": df_means["tp_sat_rate"].mean(),
        "per_seed_summary": seed_means,
    }


def aggregate_baselines(baseline_dir):
    """Baselines have one CSV per method. Each has 50 episodes."""
    rows = []
    for csv_path in sorted(Path(baseline_dir).glob("*.csv")):
        if csv_path.name == "summary.csv":
            continue
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            continue
        method = df["method"].iloc[0]
        n_eps = len(df)
        # Baselines have no training seed - aggregate over evaluation seeds.
        cpu_vals = df["cost_per_unit"].values
        tp_vals = df["throughput"].values
        ci95_cpu = 1.96 * np.std(cpu_vals, ddof=1) / np.sqrt(len(cpu_vals)) if len(cpu_vals) >= 2 else float("nan")
        ci95_tp = 1.96 * np.std(tp_vals, ddof=1) / np.sqrt(len(tp_vals)) if len(tp_vals) >= 2 else float("nan")
        rows.append({
            "method": method,
            "n_seeds": 0,  # not seed-replicated; baseline runs are deterministic policies on eval seeds
            "n_episodes_total": n_eps,
            "mean_throughput": tp_vals.mean(),
            "ci95_throughput": ci95_tp,
            "mean_cpu": cpu_vals.mean(),
            "ci95_cpu": ci95_cpu,
            "mean_total_cost": df["total_cost"].mean(),
            "ci95_total_cost": 1.96 * df["total_cost"].std(ddof=1) / np.sqrt(n_eps) if n_eps >= 2 else float("nan"),
            "util_sat_rate": df["util_satisfied"].mean(),
            "tp_sat_rate": df["tp_satisfied"].mean(),
        })
    return rows


def write_table(rows, out_path):
    if not rows:
        print(f"  (no data for {out_path})")
        return
    cols = ["method", "n_seeds", "mean_throughput", "ci95_throughput",
            "mean_cpu", "ci95_cpu", "mean_total_cost", "ci95_total_cost",
            "util_sat_rate", "tp_sat_rate"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  Wrote {out_path}")

    # Console pretty-print
    print(f"\n  {'Method':<25} {'TP':>10} {'CPU':>12} {'TotCost':>12} {'USat':>6} {'TpSat':>6}")
    print(f"  {'-'*75}")
    for r in rows:
        tp_str = f"{r['mean_throughput']:.2f}±{r['ci95_throughput']:.2f}" if not np.isnan(r['ci95_throughput']) else f"{r['mean_throughput']:.2f}"
        cpu_str = f"${r['mean_cpu']:.2f}±{r['ci95_cpu']:.2f}" if not np.isnan(r['ci95_cpu']) else f"${r['mean_cpu']:.2f}"
        tc_str = f"${r['mean_total_cost']:.0f}"
        print(f"  {r['method']:<25} {tp_str:>10} {cpu_str:>12} {tc_str:>12}  "
              f"{r['util_sat_rate']:>5.0%} {r['tp_sat_rate']:>6.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-dir", default="results")
    ap.add_argument("--output-dir", default=".")
    args = ap.parse_args()

    root = Path(args.root_dir)
    out = Path(args.output_dir)

    for testbed in ["bakery", "electronics"]:
        print(f"\n{'='*60}")
        print(f"  Aggregating {testbed}")
        print(f"{'='*60}")
        rows = []

        # Baselines
        baseline_dir = root / "baselines" / testbed
        if baseline_dir.exists():
            rows.extend(aggregate_baselines(baseline_dir))

        # Unconstrained PPO
        unc_dir = root / f"{testbed}_unconstrained"
        if unc_dir.exists():
            agg = aggregate_method_seeds(unc_dir)
            if agg:
                rows.append(agg)

        # V4 (Violation-Driven Lagrangian PPO)
        v4_dir = root / f"{testbed}_v4"
        if v4_dir.exists():
            agg = aggregate_method_seeds(v4_dir)
            if agg:
                rows.append(agg)

        # Optional: PID-Lagrangian if present
        pid_dir = root / f"{testbed}_pid"
        if pid_dir.exists():
            agg = aggregate_method_seeds(pid_dir)
            if agg:
                rows.append(agg)

        write_table(rows, out / f"final_results_{testbed}.csv")

    # Sensitivity table
    print(f"\n{'='*60}")
    print(f"  Aggregating sensitivity sweep")
    print(f"{'='*60}")
    sens_dir = root / "sensitivity"
    sens_rows = []
    if sens_dir.exists():
        for cell_dir in sorted(sens_dir.glob("T*_U*")):
            agg = aggregate_method_seeds(cell_dir)
            if agg:
                # Add the threshold values from the directory name
                parts = cell_dir.name.split("_")
                t_min = int(parts[0][1:])
                u_min = int(parts[1][1:]) / 100
                agg["t_min"] = t_min
                agg["u_min"] = u_min
                sens_rows.append(agg)
        if sens_rows:
            cols = ["t_min", "u_min", "n_seeds", "mean_throughput", "ci95_throughput",
                    "mean_cpu", "ci95_cpu", "util_sat_rate", "tp_sat_rate"]
            with open(out / "final_sensitivity.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                w.writeheader()
                for r in sens_rows:
                    w.writerow(r)
            print(f"  Wrote final_sensitivity.csv ({len(sens_rows)} cells)")


if __name__ == "__main__":
    main()
