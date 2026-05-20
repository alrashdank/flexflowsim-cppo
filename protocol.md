# Paper 6 Experimental Protocol — FROZEN

**Protocol committed:** [INSERT DATE WHEN YOU COMMIT THIS FILE]
**Author:** Khaled R. Alrashdan
**Purpose:** Pre-registers the experimental design for Paper 6 ("Violation-Driven Lagrangian PPO for Constrained Flow-Shop Routing") so that result-driven decisions (checkpoint selection, baseline comparison, sensitivity analysis) cannot be accused of post-hoc cherry-picking.

**Status:** This protocol must be committed **before any new training runs are started** for Paper 6 final results. Any deviation from this protocol must be documented in writing, with rationale, in a separate `protocol_deviations.md`.

---

## 1. Methods to evaluate

| ID | Name | Description |
|----|------|-------------|
| M1 | LeastUtilised | Existing baseline in `baselines.py` |
| M2 | ShortestQueue | Existing baseline in `baselines.py` |
| M3 | CostMinimising | Existing baseline in `baselines.py` (= "Cheapest Server") |
| M4 | FastServerFirst | Existing baseline in `baselines.py` (= "Fastest Server") |
| M5 | Unconstrained PPO | Standard PPO with scalarised reward (`w_c=0.8, w_t=0.1, w_w=0.1`) |
| M6 | Violation-Driven Lagrangian PPO (V4) | The method this paper introduces |
| M7 | (optional) PID-Lagrangian PPO | Run only if compute permits, before submission |

## 2. Testbeds

- **Bakery:** `configs/bakery_bk50.json`. 4 actions. Episode horizon 480 minutes.
- **Electronics:** `configs/electronics_3stage.json`. 12 actions. Episode horizon 480 minutes.

## 3. Constraint thresholds

- **Bakery:** T_min = 18 jobs/shift; U_min = 0.50 on `F_fast = {util_0, util_2}` (Kneader B, Oven A).
- **Electronics:** T_min = 50 jobs/shift; U_min = 0.50 on `F_fast = {util_0, util_2}` (Mounter A, Reflow 1).

## 4. Hyperparameters (M6: V4)

| Parameter | Value |
|-----------|-------|
| PPO learning rate | 3 × 10⁻⁴ |
| n_steps | 2048 |
| Batch size | 64 |
| n_epochs | 10 |
| γ | 0.99 |
| GAE λ | 0.95 |
| Clip range | 0.2 |
| Entropy coefficient | 0.01 |
| λ_U init / step / max | 5 / 20 / 500 |
| λ_T init / step / max | 200 / 4000 / 20000 |
| Warm-up steps per episode | 100 |

## 5. Training budgets

- Bakery: **1.5 million timesteps** per seed.
- Electronics: **1.6 million timesteps** per seed.
- Save checkpoint every 100K timesteps.

## 6. Seeds

- **Training seeds:** [42, 7, 2024, 123, 999]. Five seeds, fixed.
- **Validation seeds:** [10000, 10001, ..., 10019]. Used only for checkpoint selection.
- **Test seeds:** [11000, 11001, ..., 11049]. Used only for the final reported number.

These three seed ranges are disjoint by construction.

## 7. Checkpoint selection rule (CRITICAL)

For each (method, training_seed) pair:

1. After training, run a validation pass: 20 episodes per checkpoint, using validation seeds [10000-10019].
2. Among checkpoints with mean satisfaction ≥ 8/10 on **both** throughput and utilisation constraints, select the one with the **lowest mean cost-per-unit**.
3. If no checkpoint satisfies both at ≥ 8/10, select the checkpoint with the **lowest mean cost-per-unit overall** and report the failure explicitly in the results.

This rule must NOT be modified after observing results. If the rule produces unexpected outcomes, document them; do not retune the rule.

## 8. Final test evaluation

- 50 evaluation episodes, deterministic policy, on test seeds [11000-11049].
- Per (method, training_seed): report throughput, cost-per-unit, total cost, per-server utilisation, episode-level satisfaction flags.
- Per method: report mean of seed-means and 95% confidence interval over the seed-mean distribution (n=5).

## 9. Sensitivity analysis (bakery only)

Run V4 with seed 42 only at the following nine (T_min, U_min) combinations:

| Combination | T_min | U_min |
|-------------|-------|-------|
| Low / Low | 16 | 0.40 |
| Low / Mid | 16 | 0.50 |
| Low / High | 16 | 0.60 |
| Mid / Low | 18 | 0.40 |
| **Mid / Mid (paper default)** | **18** | **0.50** |
| Mid / High | 18 | 0.60 |
| High / Low | 19 | 0.40 |
| High / Mid | 19 | 0.50 |
| High / High | 19 | 0.60 |

Same training budget (1.5M), same checkpoint selection rule. Report only test-seed evaluation.

## 10. Reporting

For every CSV produced:
- One row per evaluation episode (not aggregates).
- Columns: `method, training_seed, eval_seed, episode_id, throughput, cost_per_unit, total_cost, util_0, util_1, ..., util_N, util_satisfied, tp_satisfied, checkpoint_steps`.

This raw format makes all aggregates reproducible and allows reviewers to recompute statistics if requested.

## 11. Deviations and failures

If a training run fails to converge, crashes, or produces clearly anomalous output:
1. Do NOT replace the seed.
2. Record the failure in `protocol_deviations.md` with full details (timestamp, error, partial logs).
3. Report the failure in the paper. n=4 with one documented failure is more defensible than n=5 with one silently re-run.

---

## Sign-off

By committing this file to the repository, the author certifies that this protocol was specified before observing any results from the experiments it describes.
