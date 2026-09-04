#!/usr/bin/env python3
"""
Protocol Amendment R1 §4 — archival re-evaluation.

Re-evaluates the ORIGINAL submission's checkpoints (results/*_v4, results/v6_*)
under both evaluation modes, with no retraining. Answers two questions:

  Q1  The exact checkpoint the original paper selected and reported (argmax
      selection): what is its test performance when evaluated stochastically?
  Q2  Under Amendment R1 (stochastic selection), which checkpoint would have
      been selected, and what does it score?

Resumable: per-checkpoint validation results are cached; each call stops
evaluating new checkpoints after --max-minutes and exits 0, so a long sweep can
be advanced by repeated calls.

  python3 archival_reeval.py --run v4_electronics --seed 42
  python3 archival_reeval.py --aggregate
"""
import argparse, csv, json, time
from pathlib import Path
import numpy as np

from run_ablation_2x2 import (TESTBED_CONFIG, VALIDATION_SEEDS, TEST_SEEDS,
                              evaluate_on_seeds, select_best_checkpoint, T_CRIT)

RUNS = {
    "v4_bakery":      ("results/bakery_v4",               "bakery",      1_500_000),
    "v4_electronics": ("results/electronics_v4",          "electronics", 1_600_000),
    "v6_bakery":      ("results/v6_bakery_multiseed",     "bakery",      1_500_000),
    "v6_electronics": ("results/v6_electronics_multiseed","electronics", 1_600_000),
}
WEIGHTS = (0.8, 0.1, 0.1)   # affects reward only, not the evaluated metrics
OUT = Path("results_r1/archival")


def published_selection(src, seed):
    """The checkpoint the original paper reported for this seed (from its own
    summary.csv), used for Q1 and as a fidelity check on our argmax selection."""
    p = Path(src) / "summary.csv"
    if not p.exists():
        return None, None
    for row in csv.DictReader(open(p)):
        if int(row["seed"]) == seed:
            return int(row["selected_checkpoint"]), row["selection_status"]
    return None, None


def summarise(recs):
    return {"cpu": float(np.mean([r["cost_per_unit"] for r in recs])),
            "tp": float(np.mean([r["throughput"] for r in recs])),
            "util_sat": float(np.mean([r["util_satisfied"] for r in recs])),
            "tp_sat": float(np.mean([r["tp_satisfied"] for r in recs])),
            "joint_sat": float(np.mean([min(r["util_satisfied"], r["tp_satisfied"])
                                        for r in recs]))}


def reeval(run, seed, max_minutes):
    from stable_baselines3 import PPO
    src, tb, budget = RUNS[run]
    cfg = TESTBED_CONFIG[tb]
    seed_dir = Path(src) / f"seed_{seed}"
    ckpt_dir = seed_dir / "checkpoints"
    if not ckpt_dir.exists():
        print(f"[missing] {run} seed {seed}")
        return None
    out = OUT / run / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    if summary_path.exists():
        print(f"[skip] {run} seed {seed} — done")
        return json.loads(summary_path.read_text())

    ckpts = {int(p.stem.split("_")[1]): p for p in ckpt_dir.glob("ckpt_*_steps.zip")}
    if budget not in ckpts and (seed_dir / "final.zip").exists():
        ckpts[budget] = seed_dir / "final.zip"

    cache_path = out / "validation_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    t0 = time.time()
    for st in sorted(ckpts):
        if str(st) in cache:
            continue
        if (time.time() - t0) / 60 > max_minutes:
            print(f"    [paused after {len(cache)}/{len(ckpts)} checkpoints — re-run to continue]")
            return None
        m = PPO.load(str(ckpts[st]).replace(".zip", ""), device="cpu")
        det = evaluate_on_seeds(m, cfg, WEIGHTS, VALIDATION_SEEDS, tb, deterministic=True)
        sto = evaluate_on_seeds(m, cfg, WEIGHTS, VALIDATION_SEEDS, tb, deterministic=False)
        cache[str(st)] = {"argmax": det, "stochastic": sto}
        cache_path.write_text(json.dumps(cache))
        print(f"    {run} s{seed} val {st:>9,}: argmax CPU={np.mean([r['cost_per_unit'] for r in det]):7.2f} "
              f"U={sum(r['util_satisfied'] for r in det):2d} T={sum(r['tp_satisfied'] for r in det):2d}"
              f" | stoch CPU={np.mean([r['cost_per_unit'] for r in sto]):7.2f} "
              f"U={sum(r['util_satisfied'] for r in sto):2d} T={sum(r['tp_satisfied'] for r in sto):2d}")

    rec_arg = {int(k): v["argmax"] for k, v in cache.items()}
    rec_sto = {int(k): v["stochastic"] for k, v in cache.items()}
    sel_arg, st_arg = select_best_checkpoint(rec_arg)
    sel_sto, st_sto = select_best_checkpoint(rec_sto)
    pub_sel, pub_status = published_selection(src, seed)

    # Q1: the published checkpoint, both modes.  Q2: stochastic-selected, both modes.
    def test_both(st):
        m = PPO.load(str(ckpts[st]).replace(".zip", ""), device="cpu")
        return {"argmax": summarise(evaluate_on_seeds(m, cfg, WEIGHTS, TEST_SEEDS, tb, True)),
                "stochastic": summarise(evaluate_on_seeds(m, cfg, WEIGHTS, TEST_SEEDS, tb, False))}

    q1_ckpt = pub_sel if pub_sel in ckpts else sel_arg
    q1 = test_both(q1_ckpt)
    q2 = q1 if sel_sto == q1_ckpt else test_both(sel_sto)

    summary = {
        "run": run, "testbed": tb, "seed": seed,
        "published_selected": pub_sel, "published_status": pub_status,
        "our_argmax_selected": sel_arg, "our_argmax_status": st_arg,
        "argmax_selection_reproduced": bool(pub_sel == sel_arg),
        "stoch_selected": sel_sto, "stoch_status": st_sto,
        "q1_published_ckpt": q1_ckpt, "q1": q1,
        "q2_stoch_ckpt": sel_sto, "q2": q2,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"    {run} s{seed}: published ckpt {q1_ckpt:,} ({pub_status}) — argmax ${q1['argmax']['cpu']:.2f} "
          f"joint {q1['argmax']['joint_sat']:.0%} | stoch ${q1['stochastic']['cpu']:.2f} "
          f"joint {q1['stochastic']['joint_sat']:.0%}; stoch-selected {sel_sto:,} ({st_sto}) "
          f"${q2['stochastic']['cpu']:.2f} joint {q2['stochastic']['joint_sat']:.0%}")
    return summary


def aggregate():
    rows = []
    for run, (src, tb, budget) in RUNS.items():
        S = [json.loads(p.read_text()) for p in sorted((OUT / run).glob("seed_*/summary.json"))] \
            if (OUT / run).exists() else []
        if not S:
            continue
        n = len(S)
        ci = lambda xs: (T_CRIT.get(n, 1.96) * np.std(xs, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        q1a = [s["q1"]["argmax"]["cpu"] for s in S]; q1s = [s["q1"]["stochastic"]["cpu"] for s in S]
        q2s = [s["q2"]["stochastic"]["cpu"] for s in S]
        rows.append({
            "run": run, "n": n,
            "published_ckpt_argmax_cpu": f"{np.mean(q1a):.2f}±{ci(q1a):.2f}",
            "published_ckpt_argmax_joint": f"{np.mean([s['q1']['argmax']['joint_sat'] for s in S]):.0%}",
            "published_ckpt_stoch_cpu": f"{np.mean(q1s):.2f}±{ci(q1s):.2f}",
            "published_ckpt_stoch_joint": f"{np.mean([s['q1']['stochastic']['joint_sat'] for s in S]):.0%}",
            "stoch_selected_cpu": f"{np.mean(q2s):.2f}±{ci(q2s):.2f}",
            "stoch_selected_joint": f"{np.mean([s['q2']['stochastic']['joint_sat'] for s in S]):.0%}",
            "stoch_val_satisfied": f"{sum(s['stoch_status']=='satisfied' for s in S)}/{n}",
            "argmax_val_satisfied": f"{sum(s['our_argmax_status']=='satisfied' for s in S)}/{n}",
            "argmax_sel_reproduced": f"{sum(s['argmax_selection_reproduced'] for s in S)}/{n}",
        })
    if rows:
        with open(OUT / "archival_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"\n{'run':16s}{'n':>2} | published ckpt: argmax CPU / joint | stoch CPU / joint | "
              f"stoch-selected CPU / joint | val-sat stoch / argmax | argmax sel reproduced")
        for r in rows:
            print(f"{r['run']:16s}{r['n']:>2} | {r['published_ckpt_argmax_cpu']:>14} {r['published_ckpt_argmax_joint']:>5} | "
                  f"{r['published_ckpt_stoch_cpu']:>14} {r['published_ckpt_stoch_joint']:>5} | "
                  f"{r['stoch_selected_cpu']:>14} {r['stoch_selected_joint']:>5} | "
                  f"{r['stoch_val_satisfied']:>4} / {r['argmax_val_satisfied']:>4} | {r['argmax_sel_reproduced']}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=list(RUNS))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--max-minutes", type=float, default=8.0)
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.aggregate:
        aggregate()
    else:
        reeval(a.run, a.seed, a.max_minutes)
