# Protocol Amendment R2 — symmetric, reward-scaled dual update

**Status: pre-committed. Written and committed BEFORE the first run of the cell
it defines was launched.** It amends `protocol.md` (391eb0d) and
`protocol_amendment_r1.md`; everything not mentioned here is unchanged.

## 1. Why

The R2 audit (manuscript §6.4; `protocol_deviations.md` R2-1) found that in
every cell run so far, including the corrected `cost-episode` cell, the
constraint terms of the augmented return exceed the cost term by 2–4 orders of
magnitude: the cost-only episode return is about −10 (Eq. 1, κ = 10,
C_norm = 4230), while the Lagrangian term Σ p_t at the inherited initial
multipliers (λ_T, λ_U) = (200, 5) is −950 to −3,300 on the test episodes and
grows under the hinged update, which never decreases a multiplier. Both
constraints are slack under any load-spreading policy, so the equilibrium
multipliers are zero and the trained ones cannot get there. The corrected cell
therefore tested a hinged, heavily weighted Lagrangian, not constrained PPO with
a functioning dual. This amendment defines the cell that tests the latter.

## 2. The cell: `cost-episode-sym`

Base reward (1, 0, 0); episode slack (Eq. 6, unchanged); **symmetric dual
update** at each episode boundary:

    λ_T ← clip(λ_T + η_T (T_min − TP(τ))/H, 0, λ_T,max)
    λ_U ← clip(λ_U + η_U s_U(τ),           0, λ_U,max)
    s_U(τ) = Σ_i max(0, U_min − ū_i(τ))  if any server violates,
           = −min_i (ū_i(τ) − U_min)      otherwise (release at the binding server's margin)

(`lagrangian_slack.py`, `symmetric=True`, implemented 20 Aug 2026 and unchanged.)

**Multiplier scale.** Chosen so that the constraint terms are on the scale of
the reward, by two independent arguments that agree:

- *Shadow price.* λ_T* is the marginal reward per unit of throughput at the
  constraint boundary. From the baseline interpolation of manuscript §3.4 the
  marginal cost is ≈ $40/unit, i.e. κ·40/C_norm ≈ 0.1 reward units per unit.
  λ_U* is the marginal reward per busy-step of a constrained fast server: the
  processing-cost difference between a fast and a slow server is ≈ $0.5–1/min,
  i.e. κ·0.75/C_norm ≈ 0.002 reward units per busy-step.
- *Penalty-to-return ratio.* Dividing the inherited (init, step, cap) triples by
  a common factor of ≈ 2,400 brings the initial-multiplier penalty on a typical
  episode from ≈ 250× the return to ≈ 0.1×. That gives λ_T ≈ 0.08, λ_U ≈ 0.002.

Rounded values, fixed before any run:

| | init | step η | cap |
|---|---|---|---|
| λ_T (reward per unit throughput) | 0.1 | 2.0 | 10 |
| λ_U (reward per busy-step)       | 0.002 | 0.01 | 0.2 |

With η_T = 2.0 a shortfall of one unit moves λ_T by 2/480 ≈ 0.004 per episode,
so λ_T traverses its expected scale (0.1) in ≈ 25 violating episodes and its
cap in ≈ 2,400; a 35-unit collapse to the conservative attractor moves it by
0.15 per episode. With η_U = 0.01 a 0.1 utilisation shortfall on one server
moves λ_U by 0.001 per episode. Caps are 100× the expected equilibria and exist
only as safety rails. Smoke test before commit (seed 42, uniform-random
policy, 4 episodes): Σ p_t = −0.2 to −1.2 against returns of −9.8 to −11.1;
both multipliers released to 0 within seven episodes, as they should be when
the constraints are slack.

## 3. Everything else

Electronics only (the testbed on which every earlier claim is made), 5 seeds
[42, 7, 2024, 123, 999], 1.6M steps in 400K segments (R1 §A3.1), checkpoints
every 100K, PPO hyperparameters unchanged, stochastic selection (R1 §A2.2),
both evaluation modes reported. One change to the pipeline, applied
prospectively as the remedy for deviation R2-1: action sampling in stochastic
validation and test is seeded by the episode seed (`torch.manual_seed`),
per-episode stochastic test records are written (`test_results_stoch.csv`),
and joint satisfaction is computed per episode (`*_joint_sat`).

## 4. What will be reported, whatever the outcome

The cell is added as a row of Table 3 and to the bootstrap family of Table 4
(against ShortestQueue, LeastUtilised, RoundRobin, UniformRandom, the hinged
corrected cell and the original signal: twelve tests, Holm-adjusted), with its
λ trajectories added to Figure 1 (or a companion figure), its policy statistics
to Table 6, and its penalty-to-return ratio to §6.4. The prediction is left
open in the manuscript and here: a releasable multiplier reintroduces the
oscillation between cost minimisation and constraint violation that
PID-Lagrangian methods exist to damp, and the conservative attractor sits at
λ = 0. Possible outcomes and how each will be described:

1. Cost-efficient state-aware routing emerges (CPU below RoundRobin, toward
   ShortestQueue, constraints satisfied): the corrected hinged cell understated
   the method; §7.2 is rewritten accordingly.
2. Oscillation or collapse (CPU above stateless routing, or validation
   `fallback` on most seeds): the method as configured does not solve the
   problem even with a functioning dual; the negative result stands on its
   strongest footing so far.
3. Near-uniform routing again (indistinguishable from RoundRobin): the failure
   to learn is not attributable to the multiplier scale; entropy coefficient
   and credit assignment become the leading hypotheses (§7.2).

No hyperparameter of this cell will be changed after seeing results. If a
second configuration is run, it will be recorded as a further amendment.

Recorded 5 September 2026, before launch. Implementation: `run_ablation_2x2.py`
(cell `cost-episode-sym`), `r2_tick.sh`.
