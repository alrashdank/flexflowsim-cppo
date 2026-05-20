"""
Paper 6 — sensitivity sweep on T_min and U_min (bakery only, single seed).

Per protocol §9:
  - T_min ∈ {16, 18, 19}
  - U_min ∈ {0.40, 0.50, 0.60}
  - Seed: 42 only
  - Budget: 1.5M
  - Same checkpoint selection rule

USAGE:
  python run_sensitivity.py --output-dir results/sensitivity
"""

import argparse
import subprocess
import sys
from pathlib import Path


T_MINS = [16, 18, 19]
U_MINS = [0.40, 0.50, 0.60]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="results/sensitivity")
    ap.add_argument("--budget", type=int, default=1_500_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--testbed", default="bakery")
    args = ap.parse_args()

    base_dir = Path(args.output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    for t_min in T_MINS:
        for u_min in U_MINS:
            label = f"T{t_min}_U{int(u_min*100):02d}"
            sweep_dir = base_dir / label
            print(f"\n{'#'*60}")
            print(f"  Sensitivity cell: T_min={t_min}, U_min={u_min}")
            print(f"  Output: {sweep_dir}")
            print(f"{'#'*60}")

            cmd = [
                sys.executable, "run_experiment.py",
                "--testbed", args.testbed,
                "--method", "v4",
                "--output-dir", str(sweep_dir),
                "--seeds", str(args.seed),
                "--budget", str(args.budget),
                "--t-min", str(t_min),
                "--u-min", str(u_min),
            ]
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
