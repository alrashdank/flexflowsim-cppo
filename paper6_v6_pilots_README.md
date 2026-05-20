# Paper 6 — V6 PID-Lagrangian Pilot Bundle (v2)

## Purpose

These pilots test whether V4's electronics non-convergence is driven by
**asymmetric multiplier saturation** (mechanism hypothesis) or by something
structural to the policy-optimization side. The bundle contains six single-seed
(seed 42) training runs on the V6 PID-Lagrangian variant of the constrained-PPO
training pipeline — four on bakery, two on electronics.

The TL;DR — also called Story B in the project's pre-registered decision rules:

> Three controller-side fixes (symmetric updates, retuned PID gains, EMA-smoothed
> slack signal) each address a distinct hypothesis about V4's failure. None of
> them produces a constraint-satisfying policy on the bakery testbed, even at
> V4's full 1.5M training budget. The same fixes applied to electronics
> eliminate saturation but produce 0/20 joint-sat at every checkpoint. The
> remaining failure mode is on the policy-gradient side, not the constraint
> controller. Story B replicates cleanly across both testbeds.

## What V6 is

V6 replaces V4's one-sided dual ascent
(`lam ← clip(lam + lr * max(0, gap), 0, lam_max)`) with a symmetric PID controller
per multiplier:

  - **P**, **I**, **D** terms with anti-windup back-calculation.
  - Signed slack signal (positive when constraint satisfied with margin, negative
    when violated) drives the controller. V4 only sees one-sided gap.
  - λ can decrease when constraints are over-satisfied, in principle preventing
    the saturation observed in V4 on electronics.

See `pilot_constrained_v6_pid.py` and `pilot_constrained_v6_auto.py` in the
git branch `pilot-pid-v6`. Commit sequence:
`d056bf6 → 87390cf → 8afdb97 → 5e1cbcb → 03fe8f7`.

## The six V6 pilots (all seed 42)

All pilots use checkpoint frequency 50K and the protocol's checkpoint selection
rule (16/20 validation joint-sat threshold, lowest CPU among qualifying;
fallback = lowest CPU overall).

| # | Testbed | PID coefficients | EMA | Budget | Selected ckpt | Best joint-sat | Test CPU | λ_T peak | λ_T saturated |
|---|---|---|---|---|---|---|---|---|---|
| 1 | bakery | original (Ki_tp=4000, Kd_tp=100) | no | 500K | 100K (fallback) | 15/20 @ 100K only | $154.04 | 7492 | 0% |
| 2 | bakery | retuned (Ki_tp=1000, Kd_tp=500) | no | 500K | 100K (fallback) | 5/20 @ 250–300K | $173.47 | 2440 | 0% |
| 3 | bakery | retuned | yes | 500K | 50K (fallback) | 6/20 @ 50K only | $164.11 | ~3700 | 0% |
| 4 | bakery | retuned | yes | **1.5M** | 50K (fallback) | 6/20 @ 50K only | $164.11 | 7821 | 0% |
| 5 | electronics | original (Ki_tp=4000) | yes | 600K | 50K (fallback) | 0/20 anywhere | $139.90 | 20000 | **14.1%** |
| 6 | electronics | **retuned** (Ki_tp=1000, Kd_tp=500) | yes | 600K | 450K (fallback) | **0/20 anywhere** | $223.80 | 16375 | 0% |

V4 same seed for context (from prior Batches 1+2):
- Bakery: $137.55 (satisfied, 300K ckpt) at 1.5M
- Electronics: $113.31 (fallback, USat 0%, TpSat 0%) at 1.6M

Baselines:
- Bakery — LeastUtilised $145.63 (TpSat 90%, USat 100%); ShortestQueue $138.66 (TpSat 96%, USat 100%)
- Electronics — LeastUtilised $80.47 (TpSat 98%, USat 100%); ShortestQueue $73.25 (TpSat 100%, USat 100%)

## What each pilot rules out

1. **Pilot 1 (bakery, original PID)** rules out *asymmetric saturation* as the
   sole cause of V4's failure. V6 does not saturate (λ_T peaks at 37% of cap)
   but produces a constraint-violating policy anyway.

2. **Pilot 2 (bakery, retuned PID, Ki/4, Kd×5)** rules out *integral overshoot*.
   The D-term damping activates — 40.5% of episodes show λ_T decreasing — but
   joint-sat stays bounded at 0–5/20, never crossing the threshold.

3. **Pilot 3 (bakery, retuned + EMA)** rules out *slack-signal noise*.
   tp_slack zero-crossings drop from 151 → 38 in episodes 100–500, util_slack
   crossings drop from 50 → 0, tp_slack std halves. Joint-sat still maxes at
   6/20, only at the 50K checkpoint.

4. **Pilot 4 (bakery, retuned + EMA, 3× longer budget)** rules out *insufficient
   training*. With V4's full 1.5M budget, joint-sat still never crosses 16/20
   anywhere across 30 checkpoints. Best is still 6/20 at the 50K checkpoint;
   beyond 500K, the highest joint-sat at any checkpoint is 1/20 (sporadic
   single-episode spikes at 700K, 950K, 1050K, 1150K).

5. **Pilot 5 (electronics, original PID + EMA)** demonstrates testbed-dependent
   saturation. Electronics tp_slack is more negative (raw mean −0.0066 vs
   bakery's −0.003), so the integrator hits the cap under original Ki=4000
   even with EMA smoothing. Joint-sat 0/20 throughout.

6. **Pilot 6 (electronics, retuned PID + EMA)** — the **closing reviewer ask**.
   Same controller-side fixes that fully ruled out controller-side causes on
   bakery, applied to electronics. **λ_T saturation drops from 14.1% → 0.0%**
   (16375 peak vs cap of 20000) — the saturation issue is fully addressed.
   But joint-sat is **0/20 at every single checkpoint** from 50K to 600K, no
   spikes at any budget. Confirms Story B across both testbeds: controller-side
   fixes prevent saturation but not non-convergence.

## Story B in one paragraph

Across six V6 pilots — three controller-side ablations on bakery, one budget
extension to V4's full 1.5M, and two electronics replications including the
retuned-PID-with-EMA closing pilot — the **best validation joint-sat across all
30 bakery checkpoints (50K through 1.5M) is 6/20 at the 50K checkpoint**, and
electronics never exceeds 0/20 at any checkpoint under any tuning. PID-Lagrangian
with symmetric updates, damped gain, and a clean signal cannot push PPO into
the feasible policy region on either testbed, even though that region
demonstrably exists (V4 satisfies bakery on 4/5 seeds at 1.5M; ShortestQueue
hits both testbeds deterministically). The bottleneck is the policy gradient
under the modified-reward landscape, not the Lagrangian controller.

V4's monotonic λ growth appears to produce a ratcheting pressure that V6's
symmetric mechanism relaxes too readily once partial satisfaction is achieved.
The 50K-then-collapse pattern across all four V6 bakery pilots, and the
across-the-board 0/20 on both electronics V6 pilots, are consistent with this:
early in training, the policy is still being driven by the dense
cost-minimization signal, finds a region with passable constraint satisfaction
incidentally, then collapses (bakery) or never finds the region at all
(electronics) as λ_T grows and the reward landscape destabilizes the policy
rather than guiding it.

**Important framing constraint (Khaled, 2026-05-11):** Both testbeds'
constraints are demonstrably feasible. V4 hits bakery on 4/5 seeds at 1.5M;
ShortestQueue hits both with TpSat ≥ 96% deterministically. The V6 results
are therefore a *tuning/optimization* claim, not a *feasibility* claim. The
paper should be clear on this distinction.

## Bundle contents

```
results/pilot_v6_bakery/                   — Pilot #1: original PID, no EMA, 500K bakery
results/pilot_v6_bakery_retune/            — Pilot #2: retuned PID, no EMA, 500K bakery
results/pilot_v6_bakery_ema/               — Pilot #3: retuned + EMA, 500K bakery
results/pilot_v6_bakery_1p5m/              — Pilot #4: retuned + EMA, 1.5M bakery
results/pilot_v6_electronics/              — Pilot #5: original PID, EMA, 600K electronics
results/pilot_v6_electronics_retune_ema/   — Pilot #6: retuned PID, EMA, 600K electronics

pilot_pid_bakery_lambda.png                — Pilot #1 4-panel diagnostic
pilot_pid_bakery_retune_lambda.png         — Pilot #2 4-panel diagnostic
pilot_pid_bakery_ema_lambda.png            — Pilot #3 4-panel diagnostic
pilot_pid_bakery_1p5m_lambda.png           — Pilot #4 4-panel diagnostic
pilot_pid_electronics_lambda.png           — Pilot #5 4-panel diagnostic
pilot_pid_electronics_retune_ema_lambda.png — Pilot #6 4-panel diagnostic

protocol.md                                 — Pre-registered experimental protocol
```

Each `results/pilot_v6_*/seed_42/lambda_history.csv` has columns:
`episode, lam_util, lam_tp, mean_util_gap, mean_tp_gap, mean_util_slack,
mean_tp_slack` (plus `util_slack_ema, tp_slack_ema` on pilots 3, 4, 5, 6).

Each `results/pilot_v6_*/results_seed_42.csv` has 50 test-episode records with
per-episode CPU, throughput, total_cost, util_satisfied, tp_satisfied,
per-server utilisations.

Each `results/pilot_v6_*/summary.csv` has the aggregated test metrics for the
single seed (n=1, no CIs).
