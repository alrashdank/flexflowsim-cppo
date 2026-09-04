# Protocol Amendment R1 — evaluation and selection under randomised policies

**Status: pre-committed. Written and committed BEFORE any full-budget R1 run was
launched.** It amends `protocol.md` (commit 391eb0d), which governed the original
submission. The original protocol is not retracted: every quantity it defines is
still computed and reported. This amendment adds a second, parallel reporting
track and changes which of the two selects the reported checkpoint.

---

## 1. What is being amended, and why

The original protocol evaluates a trained policy by greedy action selection
(`model.predict(obs, deterministic=True)`), both for checkpoint selection on the
20 validation episodes and for the 50 test episodes.

Constrained MDP optima are, in general, **randomised**. For a CMDP with k
constraints there exists an optimal stationary policy that randomises in at most
k states (Altman, 1999, Thm 4.4); a deterministic optimum is not guaranteed to
exist and generically does not when constraints are active. Lagrangian policy
optimisation converges toward such a mixed policy. Taking the argmax of its
action distribution is therefore not "the same policy, evaluated cleanly" — it
is a different policy, outside the class the algorithm optimised over, and the
randomisation that is discarded is precisely the mechanism carrying constraint
satisfaction.

The 400K pilot (Addendum 2, 20 Aug 2026; `results_ablation/electronics_pilot400k`)
measured the size of this effect on the electronics testbed. Across all four
ablation cells, 20 runs, the same checkpoints evaluated the two ways:

| evaluation | mean CPU | joint constraint satisfaction | between-seed CI |
|---|---|---|---|
| argmax (original protocol) | $122–155 | 0.0–0.8% | ±13 to ±88 |
| stochastic (trained policy class) | $78.6–83.1 | 68–94% | ±1.7 to ±2.9 |

The original protocol's headline finding on electronics — systematic
non-convergence — is therefore not separable from its extraction rule. The
amendment exists so that the R1 results measure the method rather than the
extraction rule, without deleting the original measurement.

## 2. The amendment

**A2.1 — Both evaluation modes are computed and reported, everywhere.** Every
checkpoint validated and every selected checkpoint tested is evaluated twice: in
argmax mode (original protocol) and in stochastic mode (sampling from the policy's
action distribution, the class optimised during training). No table reports one
without the other.

**A2.2 — Selection is stochastic.** The checkpoint-selection rule of
`protocol.md` §7 is unchanged in form — among checkpoints satisfying both
constraints on ≥16 of 20 validation episodes, take the lowest mean cost-per-unit;
if none qualify, take the lowest mean cost-per-unit overall and mark the run
`fallback` — but the validation episodes it reads are now the **stochastic** ones.

Rationale: selection and reporting must use the same criterion, or the reported
number is selected against a different objective than it is judged by. The pilot
mixed them (argmax selection, both-mode reporting), which is why every pilot cell
shows `0/5 validation-satisfied` while the same runs satisfy constraints on ~94%
of stochastic test episodes. That inconsistency is a defect of the pilot and is
corrected here. Argmax selection is still computed and reported alongside, so the
original protocol's selection remains inspectable for every run.

**A2.3 — Stochastic evaluation is seeded and disjoint.** Action sampling uses the
run's own seed offset by the episode seed; validation seeds remain [10000, 10020)
and test seeds [11000, 11050), disjoint from training as before. Stochastic
evaluation is single-draw per episode: 20 validation and 50 test episodes, as in
the original protocol, not averaged over repeated draws. This keeps the episode
budget identical between the two modes and makes them directly comparable.

**A2.4 — Deterministic feasibility is reported as an open problem, not a
footnote.** Greedy extraction is a legitimate deployment requirement in plants
that cannot accept a stochastic controller. R1 reports the argmax result for every
configuration, states plainly that no configuration produces a satisfying
deterministic policy on electronics, and treats deterministic extraction from a
constrained stochastic optimum (e.g. constrained distillation) as future work. The
paper does not claim the constrained agent is deployable under a determinism
requirement.

## 3. Amendment to the training procedure (mechanical, no algorithmic change)

**A3.1 — Segmented training.** R1's full-budget runs are executed in segments of
400,000 timesteps to fit the execution environment's wall-clock limits. Across a
segment boundary the following are carried forward: policy and value-network
weights, PPO optimiser state (both via the SB3 checkpoint), the dual variables
λ_U and λ_T, the dual-update episode counter, and the λ history. The only
quantity not carried is PPO's in-flight rollout buffer, which is flushed at each
boundary: 3 flushes in a 1.6M-step run, i.e. 0.4% of the ~780 rollouts, each
discarding at most 2047 transitions that would otherwise have contributed to one
gradient update. Segment boundaries fall at fixed step counts identical across all
cells and seeds, so no cell is advantaged.

**A3.2 — Unchanged.** PPO hyperparameters, seed sets, checkpoint interval
(100K), episode horizon, constraint thresholds, testbed configurations, dual
learning rates, and multiplier caps are all exactly as in `protocol.md`.

## 4. Experimental design for R1

**Mechanism ablation (2×2), pilot budget 400K, 5 seeds/cell — COMPLETE.**
{shaped, cost} base reward × {cumulative-rate, episode} slack. Fixed budget across
cells makes this a controlled comparison; its purpose is to attribute λ saturation
and the satisfaction gap to the slack operationalisation, which it does
(10/10 cumulative-rate seeds saturate, 0/10 episode seeds do). This experiment is
reported at 400K and is not re-run at full budget: the attribution does not depend
on budget, and re-running it would cost ~12 compute-hours to restate a result the
pilot already establishes at an effect size of 10/10 vs 0/10.

**Headline comparison, full protocol budget, 5 seeds/cell.** Two cells only —
`shaped-cumrate` (the published V4 operationalisation, control) and `cost-episode`
(the CMDP as stated in §3.3, corrected) — at 1.6M steps on electronics and 1.5M
on bakery, against the tuned dispatching-rule baselines. This is what the R1
headline claims are computed from.

**Archival re-evaluation.** The original submission's 1.6M/1.5M checkpoints (V4
and V6, both testbeds, all seeds) are re-evaluated under A2.1 without retraining.
This is the decisive test of whether the published negative result is an artefact
of extraction alone, and it uses the original artefacts unchanged.

## 5. Deviations from this amendment

Any deviation is recorded in `protocol_deviations.md` with date and reason, as
under the original protocol.

---

Recorded 20 August 2026. Implementation: `run_ablation_2x2.py` (flags
`--selection`, `--segment-steps`), `lagrangian_slack.py`.
