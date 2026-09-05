\# Protocol Deviations and Failures



This document records any deviations from `protocol.md` or training failures

that occurred during the Paper 6 experimental runs. Per protocol §11, failures

are documented rather than silently re-run.



\---



\## Entry 1 — Seed 999 non-convergence (bakery V4)



\- \*\*Date observed:\*\* 2026-05-10

\- \*\*Run:\*\* Batch 1 — bakery V4 multi-seed (5 seeds × 1.5M timesteps)

\- \*\*Stage:\*\* `bakery\_v4`, training seed 999

\- \*\*Status:\*\* TRAINING FAILURE — non-convergence to constrained region

\- \*\*Pipeline log:\*\* `logs/batch1\_20260510\_110752.log`



\### What happened



Per protocol §7 (checkpoint selection rule), validation evaluated all 15

checkpoints (every 100K steps from 100K to 1.5M) on validation seeds \[10000-10019].

For seed 999, no checkpoint met the satisfaction threshold of ≥16/20 episodes

satisfying both throughput and utilisation constraints jointly.



The fallback rule fired: the lowest-CPU checkpoint among non-satisfying

checkpoints was selected. The selected checkpoint was 1.3M steps with:



&#x20; - Validation: CPU $184.06, joint satisfaction 0/20

&#x20; - Test (50 episodes, seeds 11000-11049): CPU $188.60, util\_sat 0%, tp\_sat 4%



Selection status recorded in `results/bakery\_v4/results\_seed\_999.csv` as

`fallback`.



\### What this means



Seed 999 represents a training failure in which Violation-Driven Lagrangian PPO

did not escape the conservative routing attractor across the full 1.5M training

budget on the bakery testbed.



This is consistent with the broader paper findings (V5 oscillation, Section 6.4)

that one-sided multiplier updates exhibit instability under some initial

conditions. It strengthens the motivation for PID-Lagrangian PPO as the natural

follow-up (Section 8).



\### Action taken



Per protocol §11, no replacement seed was added. The original 5 pre-registered

seeds \[42, 7, 2024, 123, 999] are reported in full, with seed 999's

non-convergence transparently documented in the paper:



&#x20; - All-seeds aggregate: $149.30 ± $19.58 (n=5, includes the failure)

&#x20; - Converged-seeds aggregate: $139.48 ± $4.57 (n=4, seed 999 excluded with reason)



Both numbers will be reported in the paper. The 20% (1/5) non-convergence rate

is reported as a robustness finding, not hidden.



\### Why this is preferable to replacing the seed



n=5 with one documented failure is more defensible than n=5 with one silent

re-run. The pre-registration commit (`391eb0d`, "Pre-register Paper 6

experimental protocol") timestamps the seed list before any results were

observed; replacing seed 999 post-hoc would constitute selection-after-observation,

which is the exact form of cherry-picking the protocol exists to prevent.



## Deviation R2-1 (5 September 2026): seeding of stochastic evaluation, and the "joint satisfaction" aggregate

**What Amendment R1 §A2.3 specified.** Action sampling in stochastic evaluation
"uses the run's own seed offset by the episode seed".

**What was implemented.** `run_ablation_2x2.py` and `archival_reeval.py` call
`model.predict(obs, deterministic=False)` without seeding the torch generator.
Stochastic validation and test draws therefore came from the process-global
generator (seeded by Stable-Baselines3 at model construction and advanced by
training). They are single draws and are not reproducible without re-running
training. The same scripts aggregated "joint satisfaction" across seeds as
min(mean util-satisfied, mean tp-satisfied), which is an upper bound on the
per-episode joint rate, not the joint rate itself.

**Consequence and remedy.**
- Checkpoint selection stands as executed. It read the unseeded stochastic
  validation draws, which are archived per checkpoint in `validation_cache.json`.
- Every stochastic *test* figure reported from R2 onwards is a seeded
  re-evaluation of the selected checkpoints (`analysis_r2.py`): before each
  episode the torch generator is seeded with the episode seed, so the action
  stream is reproducible and common across policies evaluated on the same
  episode. Per-episode records (cost, throughput, per-server utilisation) are
  archived under `results_r1/r2/`, and joint satisfaction is computed per
  episode. Argmax figures are unaffected (deterministic) and reproduce the R1
  values to the cent, which is the fidelity check on the re-evaluation.
- The unseeded pipeline draws remain in each run's `summary.json`
  (`stoch_test_*`) and in `results_r1/archival/*/summary.json`. Cell-level
  differences between the two draws: cost-per-unit ≤ $1.2 (bakery corrected,
  $139.95 vs $141.07) and joint satisfaction ≤ 5.6 pp (electronics control,
  75.6% vs 70.0%); no comparison changes direction.
- A2.3's wording "run seed offset by the episode seed" is superseded by
  "episode seed" (common random numbers across policies).
