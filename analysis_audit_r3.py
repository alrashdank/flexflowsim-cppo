#!/usr/bin/env python3
"""Measurements added in revision R3, after an independent audit of the R2 text.

Every figure this script emits appears in the manuscript and was not produced by
any earlier analysis script.  Run from the repository root:

    python3 analysis_audit_r3.py            # all blocks
    python3 analysis_audit_r3.py fill       # one block

Output: r3_numbers.json, plus a readable summary on stdout.

Blocks
------
fill        Section 4.2.  Fill-phase statistics of the cumulative-rate slack
            g_T under ShortestQueue, measured through the published wrapper in
            `cumrate` mode, and the per-episode dual increment they imply
            against the increment needed to reach the cap within budget.
lambda      Section 4.2 and 6.1.  Saturation episode and mean increment of the
            trained control cells, read from the archived lambda histories.
symsat      Section 6.7.  Marginal and joint constraint satisfaction of the
            symmetric cell on the 50 test episodes, and the qualifying- and
            selected-checkpoint counts behind the 16-of-20 rule.
cost        Section 6.7.  Paired hierarchical bootstrap on TOTAL episode cost,
            the objective the CMDP actually minimises, which the R2 analysis
            reported only as point estimates.
ablate      Section 6.1.  Paired hierarchical bootstrap on the 2x2 pilot cells,
            isolating the base reward with the slack signal held fixed.
"""

import json
import sys

import numpy as np

H = 480
WARM = 100
ETA_T = 4000.0
LAM_T0 = 200.0
LAM_T_CAP = 20000.0
FAST = [0, 2]
U_MIN = 0.50
TEST_SEEDS = range(11000, 11050)
BOOT = 10000

TESTBED = {
    "bakery": dict(config="configs/bakery_bk50.json", t_min=18, episodes=3142),
    "electronics": dict(config="configs/electronics_3stage.json", t_min=50,
                        episodes=3352),
}


# ----------------------------------------------------------------- fill phase
def block_fill():
    """Section 4.2: how far below its floor the cumulative rate sits, and what
    that implies for the one-sided dual update, for an oracle dispatching rule."""
    from env import FlexFlowSimEnv
    from lagrangian_slack import SlackLagrangianFlowEnv
    from baselines import ShortestQueuePolicy

    out = {}
    for tb, cfg in TESTBED.items():
        base = FlexFlowSimEnv(config=cfg["config"], weights=(1.0, 0.0, 0.0),
                              seed=11000)
        wrap = SlackLagrangianFlowEnv(
            base, fast_server_indices=FAST, util_floor=U_MIN,
            tp_rate_floor=cfg["t_min"] / H, slack_mode="cumrate",
            lr_tp=ETA_T, lam_tp_init=LAM_T0, lam_tp_max=LAM_T_CAP)
        pol = ShortestQueuePolicy(base)
        pol.env = base

        crossings, viol_frac = [], []
        for s in TEST_SEEDS:
            obs, _ = wrap.reset(seed=s)
            done, t, first, pos, n = False, 0, None, 0, 0
            while not done:
                obs, _, term, trunc, info = wrap.step(pol.predict(obs))
                done = term or trunc
                t += 1
                if t > WARM:
                    dep = float(info.get("total_departed", 0))
                    st = float(info.get("sim_time", t))
                    g = max(0.0, cfg["t_min"] / H - dep / max(st, 1e-9))
                    n += 1
                    if g > 0:
                        pos += 1
                    elif first is None:
                        first = t
            crossings.append(first if first else WARM)
            viol_frac.append(pos / n)
        wrap.reset(seed=11050)          # flush the final dual update

        gaps = np.array([h["tp_gap"] for h in wrap.lambda_history])
        mean_g = float(gaps.mean())
        incr = ETA_T * mean_g
        need = (LAM_T_CAP - LAM_T0) / cfg["episodes"]
        cross = np.array(crossings)
        out[tb] = dict(
            n_episodes=int(gaps.size),
            mean_g_T=mean_g,
            increment_per_episode=incr,
            increment_needed_for_cap=need,
            slack_needed_for_cap=need / ETA_T,
            saturates_within_budget=bool(incr >= need),
            cap_at_episode=int(np.ceil((LAM_T_CAP - LAM_T0) / incr)),
            budget_episodes=cfg["episodes"],
            crossing_median=float(np.median(cross)),
            crossing_min=int(cross.min()),
            crossing_max=int(cross.max()),
            violating_step_fraction=float(np.mean(viol_frac)),
        )
    return out


# --------------------------------------------------------- lambda_T histories
def block_lambda():
    """Sections 4.2 and 6.1: where the trained control cells actually saturate."""
    import csv
    import glob
    import os

    out = {}
    for tb in TESTBED:
        for cell in ("shaped-cumrate", "cost-episode"):
            seeds = {}
            for f in sorted(glob.glob(
                    f"results_r1/{tb}/{cell}/seed_*/lambda_history.csv")):
                with open(f) as fh:
                    rows = list(csv.DictReader(fh))
                if not rows:
                    continue
                lam = [float(r["lam_tp"]) for r in rows]
                gap = [float(r["tp_gap"]) for r in rows]
                capped = next((i for i, v in enumerate(lam, 1)
                               if v >= LAM_T_CAP - 1), None)
                seeds[os.path.basename(os.path.dirname(f))] = dict(
                    episodes=len(lam),
                    final_lam_T=lam[-1],
                    cap_at_episode=capped,
                    cap_at_fraction=(capped / len(lam)) if capped else None,
                    mean_increment_to_cap=(
                        float(np.mean(gap[:capped]) * ETA_T) if capped else None),
                )
            if seeds:
                out[f"{tb}/{cell}"] = seeds
    return out


# ------------------------------------------------------- symmetric cell rates
def block_symsat():
    """Section 6.7: marginal vs joint satisfaction, and the 16-of-20 counts."""
    import glob
    import os

    t_min = TESTBED["electronics"]["t_min"]
    recs = json.load(open("results_r1/r2/episodes_electronics_r2.json"))

    out = {}
    for mode, key in (("stochastic", "stoch"), ("greedy", "argmax")):
        pool = [r for k, v in recs.items() if k.endswith(key) for r in v]
        n = len(pool)
        out[mode] = dict(
            n_episodes=n,
            tp_satisfied=sum(1 for r in pool if r["tp"] >= t_min) / n,
            util_satisfied=sum(
                1 for r in pool if all(r["util"][i] >= U_MIN for i in FAST)) / n,
            joint_satisfied=sum(
                1 for r in pool
                if r["tp"] >= t_min
                and all(r["util"][i] >= U_MIN for i in FAST)) / n,
            mean_throughput=float(np.mean([r["tp"] for r in pool])),
            mean_total_cost=float(np.mean([r["cost"] for r in pool])),
        )

    total = qualifying = 0
    per_seed = {}
    for d in sorted(glob.glob("results_r2/electronics/cost-episode-sym/seed_*")):
        cache = json.load(open(os.path.join(d, "validation_cache.json")))
        total += len(cache)
        qual = []
        for ck, modes in cache.items():
            eps = modes.get("stochastic")
            if not eps:
                continue
            if (sum(e["util_satisfied"] for e in eps) >= 16
                    and sum(e["tp_satisfied"] for e in eps) >= 16):
                qual.append(int(ck))
        qualifying += len(qual)
        per_seed[os.path.basename(d)] = dict(
            n_checkpoints=len(cache), qualifying=sorted(qual))
    out["selection"] = dict(
        checkpoints_validated=total,
        checkpoints_qualifying=qualifying,
        seeds_with_no_qualifying_checkpoint=sum(
            1 for v in per_seed.values() if not v["qualifying"]),
        per_seed=per_seed,
    )
    return out


# ------------------------------------------------------------------ bootstrap
def _hboot(X, Y, rng, n_boot=BOOT):
    """Paired hierarchical bootstrap, identical in form to analysis_r2_stats.py:
    seeds resampled within each cell, one shared vector of episode indices."""
    ns_x, ns_y, ne = X.shape[0], Y.shape[0], X.shape[1]
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        sx = rng.integers(ns_x, size=ns_x)
        sy = rng.integers(ns_y, size=ns_y)
        se = rng.integers(ne, size=ne)
        diffs[b] = X[np.ix_(sx, se)].mean() - Y[np.ix_(sy, se)].mean()
    obs = X.mean() - Y.mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return dict(difference=float(obs), ci_low=float(lo), ci_high=float(hi),
                p=float(p), p_is_at_resolution_limit=bool(p < 2.0 / n_boot))


def block_cost():
    """Section 6.7: the same bootstrap applied to TOTAL episode cost."""
    rng = np.random.default_rng(20260911)
    sym = json.load(open("results_r1/r2/episodes_electronics_r2.json"))
    base = json.load(open("results_r1/r2/episodes_electronics_base.json"))
    rl = json.load(open("results_r1/r2/episodes_electronics_rl.json"))

    def stack(src, prefix):
        seeds = sorted({k.split("/")[1] for k in src if k.startswith(prefix + "/")})
        return np.array([[r["cost"] for r in src[f"{prefix}/{s}/stoch"]]
                         for s in seeds])

    X = stack(sym, "cost-episode-sym")
    out = {}
    for name in ("RoundRobin", "UniformRandom", "ShortestQueue", "LeastUtilised"):
        Y = np.array([[r["cost"] for r in base[f"baseline/{name}"]]])
        out[f"symmetric vs {name}"] = _hboot(X, Y, rng)
    out["symmetric vs corrected"] = _hboot(X, stack(rl, "cost-episode"), rng)
    return out


def block_ablate():
    """Section 6.1: base reward isolated, slack signal held fixed."""
    rng = np.random.default_rng(20260911)
    d = json.load(open(
        "results_r1/r2/episodes_pilot_shaped-cumrate_shaped-episode.json"))
    d.update(json.load(open(
        "results_r1/r2/episodes_pilot_cost-cumrate_cost-episode.json")))
    t_min = TESTBED["electronics"]["t_min"]

    def stack(cell, metric):
        seeds = sorted({k.split("/")[1] for k in d if k.startswith(cell + "/")})
        rows = []
        for s in seeds:
            eps = d[f"{cell}/{s}/stoch"]
            if metric == "cpu":
                rows.append([r["cost"] / r["tp"] for r in eps])
            else:
                rows.append([
                    100.0 * (r["tp"] >= t_min
                             and all(r["util"][i] >= U_MIN for i in FAST))
                    for r in eps])
        return np.array(rows)

    out = {}
    for slack in ("cumrate", "episode"):
        for metric, label in (("cpu", "cost per unit ($)"),
                              ("joint", "joint satisfaction (pp)")):
            out[f"{slack}: shaped minus cost-only, {label}"] = _hboot(
                stack(f"shaped-{slack}", metric),
                stack(f"cost-{slack}", metric), rng)
    return out


BLOCKS = {"fill": block_fill, "lambda": block_lambda, "symsat": block_symsat,
          "cost": block_cost, "ablate": block_ablate}


def main():
    wanted = sys.argv[1:] or list(BLOCKS)
    result = {}
    for name in wanted:
        if name not in BLOCKS:
            raise SystemExit(f"unknown block {name!r}; choose from {list(BLOCKS)}")
        print(f"[{name}] ...", flush=True)
        result[name] = BLOCKS[name]()
    with open("r3_numbers.json", "w") as fh:
        json.dump(result, fh, indent=1, sort_keys=True)
    print(json.dumps(result, indent=1, sort_keys=True))
    print("\nwritten: r3_numbers.json")


if __name__ == "__main__":
    main()
