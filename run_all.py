"""
Paper 6 — master launcher.

Runs the full experimental pipeline in the protocol-defined order, with
progress tracking and safe resumption. Each step writes to a results
subdirectory and is skipped if already complete.

USAGE:

    # Run everything (40-55 CPU-hours total):
    python run_all.py

    # Run a quick smoke test first (~10 minutes total) to verify the
    # pipeline works on your machine before committing to the full sweep:
    python run_all.py --quick

    # Run only specific stages:
    python run_all.py --stages baselines bakery_v4
    python run_all.py --stages bakery_v4 bakery_unconstrained
    python run_all.py --stages sensitivity
    python run_all.py --stages aggregate

    # Different output root:
    python run_all.py --output-dir my_results

STAGES:
    baselines              — Dispatching-rule baselines, both testbeds (~30 min)
    bakery_v4              — V4 on bakery, 5 seeds × 1.5M (5-7 hours)
    bakery_unconstrained   — Unconstrained PPO on bakery, 5 seeds (4-5 hours)
    sensitivity            — T_min/U_min 9-cell sweep on bakery (10-15 hours)
    electronics_v4         — V4 on electronics, 5 seeds × 1.6M (10-16 hours)
    electronics_unconstrained — Unconstrained PPO on electronics (8-12 hours)
    aggregate              — Combine all CSVs into final results tables
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ============================================================
# Stage definitions
# ============================================================

ALL_STAGES = [
    "baselines",
    "bakery_v4",
    "bakery_unconstrained",
    "sensitivity",
    "electronics_v4",
    "electronics_unconstrained",
    "aggregate",
]


def stage_baselines(output_dir, quick):
    """Run all 4 dispatching baselines on both testbeds."""
    for testbed in ["bakery", "electronics"]:
        out = output_dir / "baselines" / testbed
        if (out / "summary.csv").exists():
            print(f"  [skip] {testbed} baselines already complete")
            continue
        cmd = [sys.executable, "run_baselines.py",
               "--testbed", testbed,
               "--output-dir", str(out)]
        run_cmd(cmd)


def stage_v4(output_dir, testbed, quick):
    """Train V4 on a testbed."""
    out = output_dir / f"{testbed}_v4"
    if (out / "summary.csv").exists():
        print(f"  [skip] {testbed}_v4 already complete")
        return

    if quick:
        # Quick mode: 1 seed, 200K timesteps
        cmd = [sys.executable, "run_experiment.py",
               "--testbed", testbed, "--method", "v4",
               "--output-dir", str(out),
               "--seeds", "42", "--budget", "200000"]
    else:
        cmd = [sys.executable, "run_experiment.py",
               "--testbed", testbed, "--method", "v4",
               "--output-dir", str(out)]
    run_cmd(cmd)


def stage_unconstrained(output_dir, testbed, quick):
    """Train unconstrained PPO on a testbed."""
    out = output_dir / f"{testbed}_unconstrained"
    if (out / "summary.csv").exists():
        print(f"  [skip] {testbed}_unconstrained already complete")
        return

    if quick:
        cmd = [sys.executable, "run_unconstrained.py",
               "--testbed", testbed, "--output-dir", str(out),
               "--seeds", "42", "--budget", "200000"]
    else:
        cmd = [sys.executable, "run_unconstrained.py",
               "--testbed", testbed, "--output-dir", str(out)]
    run_cmd(cmd)


def stage_sensitivity(output_dir, quick):
    """Run T_min/U_min sensitivity sweep on bakery."""
    out = output_dir / "sensitivity"
    if (out / "T18_U50" / "summary.csv").exists():
        print(f"  [skip] sensitivity already started — check {out} manually")
        return

    if quick:
        # Quick: 3 cells only, smaller budget
        for t_min, u_min in [(16, 0.40), (18, 0.50), (19, 0.60)]:
            label = f"T{t_min}_U{int(u_min*100):02d}"
            cell_out = out / label
            if (cell_out / "summary.csv").exists():
                continue
            cmd = [sys.executable, "run_experiment.py",
                   "--testbed", "bakery", "--method", "v4",
                   "--seeds", "42", "--budget", "200000",
                   "--t-min", str(t_min), "--u-min", str(u_min),
                   "--output-dir", str(cell_out)]
            run_cmd(cmd)
    else:
        cmd = [sys.executable, "run_sensitivity.py",
               "--output-dir", str(out)]
        run_cmd(cmd)


def stage_aggregate(output_dir, quick):
    """Aggregate all results into final tables."""
    cmd = [sys.executable, "aggregate_results.py",
           "--root-dir", str(output_dir),
           "--output-dir", str(output_dir)]
    run_cmd(cmd)


STAGE_FUNCS = {
    "baselines":                lambda d, q: stage_baselines(d, q),
    "bakery_v4":                lambda d, q: stage_v4(d, "bakery", q),
    "bakery_unconstrained":     lambda d, q: stage_unconstrained(d, "bakery", q),
    "sensitivity":              lambda d, q: stage_sensitivity(d, q),
    "electronics_v4":           lambda d, q: stage_v4(d, "electronics", q),
    "electronics_unconstrained":lambda d, q: stage_unconstrained(d, "electronics", q),
    "aggregate":                lambda d, q: stage_aggregate(d, q),
}


# ============================================================
# Helpers
# ============================================================

def run_cmd(cmd):
    print(f"\n  $ {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  ! Command failed (exit {result.returncode}) after {elapsed/60:.1f} min")
        print(f"    Continuing with next stage. Inspect logs and retry this stage manually.")
    else:
        print(f"  ✓ Done in {elapsed/60:.1f} min")


def precheck():
    """Verify required files are present and dependencies installed."""
    required_files = [
        "env.py", "baselines.py", "pilot_constrained_v4_auto.py",
        "run_experiment.py", "run_unconstrained.py", "run_baselines.py",
        "run_sensitivity.py", "aggregate_results.py",
        "configs/bakery_bk50.json", "configs/electronics_3stage.json",
        "protocol.md",
    ]
    missing = [f for f in required_files if not Path(f).exists()]
    if missing:
        print("ERROR: missing files in working directory:")
        for f in missing:
            print(f"  - {f}")
        sys.exit(1)

    # Check imports
    try:
        import simpy
        import gymnasium
        import stable_baselines3
        import numpy
        import pandas
        import scipy
    except ImportError as e:
        print(f"ERROR: missing Python package: {e.name}")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)

    # Check protocol committed (warn only; this should be done manually)
    if not Path(".git").exists():
        print("WARNING: not in a git repository.")
        print("  Recommended: git init && git add protocol.md && git commit -m 'Pre-register protocol'")
        print("  This timestamps the protocol before any results are observed.")
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default="results",
                    help="Root directory for all results (default: results/)")
    ap.add_argument("--stages", nargs="+", choices=ALL_STAGES + ["all"],
                    default=["all"],
                    help="Stages to run (default: all)")
    ap.add_argument("--quick", action="store_true",
                    help="Quick smoke test: 1 seed, 200K timesteps. ~10 min total.")
    args = ap.parse_args()

    print("="*70)
    print("  Paper 6 master launcher")
    print("="*70)

    precheck()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output root: {output_dir.resolve()}")
    print(f"  Mode: {'QUICK SMOKE TEST' if args.quick else 'FULL RUN'}")

    stages = ALL_STAGES if "all" in args.stages else args.stages
    print(f"  Stages: {', '.join(stages)}")

    if not args.quick:
        print()
        print("  Estimated time for full run: 40-55 CPU-hours")
        print("  This script can be safely interrupted and resumed.")
        print("  Already-completed stages will be skipped on re-run.")

    print()
    overall_start = time.time()

    for stage in stages:
        print()
        print("="*70)
        print(f"  Stage: {stage}")
        print("="*70)
        STAGE_FUNCS[stage](output_dir, args.quick)

    elapsed = (time.time() - overall_start) / 60
    print()
    print("="*70)
    print(f"  All requested stages complete in {elapsed:.1f} minutes")
    print(f"  Results in: {output_dir.resolve()}")
    print("="*70)
    if "aggregate" in stages or "all" in args.stages:
        print()
        print("  Final result tables:")
        for f in ["final_results_bakery.csv",
                  "final_results_electronics.csv",
                  "final_sensitivity.csv"]:
            p = output_dir / f
            if p.exists():
                print(f"    - {p}")


if __name__ == "__main__":
    main()
