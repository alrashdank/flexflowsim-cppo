---
title: "Why Constrained Reinforcement Learning Appeared to Fail on Flow-Shop Routing: Two Artefacts in Training and Evaluation, and What Their Correction Reveals"
---

## Abstract

A reported failure of a constrained reinforcement learning method can be a
property of the pipeline rather than of the method. This paper documents two
such properties in a Lagrangian Proximal Policy Optimization (PPO) study of
multi-server flow-shop routing, revisiting an earlier manuscript that reported
systematic constraint failure. First, an episode-level constraint was implemented as a per-step penalty on the
cumulative throughput rate, a signal that is positive while the line fills
regardless of the policy; under a one-sided dual update it saturates the
multiplier for any policy, including an oracle (10 of 10 seeds; 0 of 10 with an
episode-level slack). Second, near-uniform stochastic policies were
evaluated by greedy action selection; evaluated stochastically, the same
checkpoints move from about 0% to 64–94% joint satisfaction, reversing the
original verdict without retraining. A
corrected formulation at the full pre-registered budget satisfies the validation
criterion on every seed, at $78.10 ± 1.64 per unit against $73.25 for
ShortestQueue, but learns near-uniform routers indistinguishable from
round-robin routing on cost-per-unit, because under the one-sided update the
constraint terms outweigh the cost term by two to four orders of magnitude and
cannot be released. A further cell, specified prospectively before its runs, with a
symmetric update and multipliers on the scale of the reward, produces a working
Lagrangian and a state-dependent policy driven toward the throughput floor by
the total-cost objective, as the stated problem asks; it is therefore worse per
unit than random routing and infeasible on a quarter of episodes, because total
cost is nearly flat in routing while cost-per-unit falls with throughput. The
apparent systematic constraint failure disappears once the training signal and
the evaluation mode are corrected; the remaining failure is to learn a
cost-efficient state-aware routing policy, and its reason lies in the gap
between the objective posed and the metric reported. All code and results are
open source.

**Keywords:** constrained reinforcement learning, Lagrangian methods, flow-shop
routing, production scheduling, negative results, reproducibility

## 1. Introduction

Deep reinforcement learning for production scheduling has settled into a
recognisable pattern. A policy-gradient agent, usually Proximal Policy
Optimization (PPO; Schulman et al., 2017), is trained against a
scalarised reward and compared with dispatching rules that receive far less
tuning attention than the agent does. Two recent systematic reviews of the field
(Mayerhoff & Schmidt, 2026; Schneider et al., 2026) identify the same weaknesses:
simulation-only validation, weak baseline practice, and an absence of
standardised benchmarks. An earlier manuscript by the author (Author, 2026a; referred to below as the
original manuscript, and provided as Supplementary Material S1) was a response
to that critique,
with a pre-registered protocol, disjoint train, validation and test seed ranges,
tuned dispatching baselines, and non-convergence reported openly.

That manuscript reported a negative result. Under cost-emphasising reward
weights, PPO converges to what we called the conservative routing attractor: it
idles expensive servers, total cost falls, throughput falls further, and
cost-per-unit (CPU) ends up worse than that of any dispatching rule.
Reformulating the problem as a Constrained Markov Decision Process (CMDP; Altman,
1999) and applying Lagrangian penalties
repaired this partially on a 4-action bakery testbed and not at all on a
12-action electronics testbed, where no seed satisfied the validation criterion
and the multiplier on the throughput constraint saturated against its cap in
every run.

This paper revisits that conclusion, and its subject is more general than the
testbeds. A constrained learning pipeline contains three objects that are easy
to conflate: the constraint stated in the problem formulation, the surrogate the
training loop actually penalises, and the criterion by which the trained policy
is judged. When the three differ, a reported failure can belong to the pipeline
rather than to the method, and aggregate metrics do not reveal which. Here the
reported failure was an artefact of how the constraints were implemented and how
the policies were evaluated, and once both are corrected a different and less
flattering picture of the method emerges. The paper makes four contributions.

The first is the identification and quantitative characterisation of a class of
constraint-implementation error. The stated constraint is on an episode total,
but the implemented training signal penalised the shortfall of a cumulative
average rate at every step. In a flow shop with non-trivial flow time, that
quantity is below its floor for a long prefix of every episode no matter what the
policy does, because the line is still filling. We show that under a monotone
one-sided dual update this guarantees multiplier saturation for any policy,
derive the saturation time from measurable quantities, and confirm that it holds
even when the policy is an oracle dispatching rule that satisfies the true
constraint with a 14–23% margin.

The second is to show that greedy evaluation of the learned policies measured
something arbitrary. The learned action distributions are close to uniform
(§6.5); the argmax of a near-uniform distribution is a deterministic policy
chosen essentially at random, and its score varies wildly across seeds. Evaluated
stochastically, the same checkpoints satisfy their constraints, and the original
pass/fail verdict on electronics reverses on the original artefacts without
retraining.

The third is the corrected formulation and a re-run at the full pre-registered
budget. The corrected per-step penalty is the signed pro-rata decomposition of
the episode Lagrangian, which sums over the episode to the Lagrangian of the
stated expectation constraint, on the same episode-level quantities the protocol
evaluates. Every seed of the corrected formulation
satisfies the validation criterion on both testbeds. On the harder testbed the
correction improves both constraint satisfaction and cost-per-unit over the
original implementation, and both differences survive a Holm adjustment over
the ten tests performed; on the easier testbed it changes nothing measurable.

The fourth follows from the second and is the finding we would least have
predicted. Compared against uniform-random and round-robin routing under the same
stochastic evaluation, with uncertainty in both the learned policies and the
baselines propagated by a paired hierarchical bootstrap, the corrected
constrained agent is indistinguishable on cost-per-unit. Its cost-per-unit is
6.6% above ShortestQueue, a state-aware rule that requires no training. The
audit that follows explains why. In the original and the corrected one-sided
configurations alike, the constraint terms of the augmented objective outweigh
the cost term by two to four orders of magnitude, because the multipliers were
initialised in units inherited from the per-step signal and a one-sided dual
update can never release them once the constraints are slack. The agent was, in
effect, trained to maximise throughput and fast-server utilisation, which it did
by tilting a near-uniform router toward the fast servers. The method did not
fail the protocol's constraint criterion; stateless load-balancing meets it
too. It failed to learn a cost-efficient, state-aware routing policy competitive with a
two-line dispatching rule, and under the objective it was actually given it had
little reason to. A final cell, specified prospectively in a protocol amendment
before its runs, gives it that reason, with a symmetric dual update and
multipliers on the scale of the reward. The dual then works, the policy becomes
state-dependent, and it is driven toward the throughput floor by the total-cost
objective as the stated CMDP directs, which makes it worse on cost-per-unit
than stateless routing and leaves a quarter of episodes infeasible under the
protocol's per-episode criterion. The reason is measurable
across every policy in the study: total episode cost is nearly flat in routing,
throughput is not, and the objective posed is therefore not the metric
reported.

The scope of the result should be stated plainly. The two artefacts are
properties of the pipeline, not of constrained reinforcement learning, and the
first in particular is easy to introduce and difficult to detect from aggregate
metrics. Correcting them does not make constrained PPO competitive on this
problem, and it does not make deterministic deployment feasible. What it does is
replace a wrong negative result with a correct and narrower one, run the
experiment that tests the method proper, and locate the remaining failure in
the formulation rather than in the optimiser.

## 2. Related work

### 2.1 RL for flow-shop and production scheduling

Two recent systematic reviews map the field. Mayerhoff and Schmidt (2026)
catalogue 196 RL studies in production planning and control between 2018 and
2024, reporting a shift from value-based to policy-gradient methods, with PPO
dominant. Schneider et al. (2026) reach a similar headline. Both flag the same
gaps: simulation-only validation, weak baseline practice, and lack of
standardised benchmarks. Recent flow-shop RL has converged on graph-based
architectures with PPO (Li et al., 2025; Liu et al., 2025; Shen et al., 2026;
Wang et al., 2025), with a consistent pattern of impressive performance against
weak baselines and ambiguous performance against well-tuned heuristics. An
alternative to soft penalties is action masking, which zeroes out infeasible
actions in the policy distribution (Ali & Tirel, 2023; Tang et al., 2020; Zhang
et al., 2025). Masking presupposes that some actions are infeasible,
which does not apply in the fully feasible routing problem we study.

Doherty et al. (2025) is the most directly relevant single paper on the
benchmarking problem. They reproduce five landmark RL studies in optical resource
allocation, apply properly tuned heuristic baselines, and find that simple
heuristics consistently match or outperform the published RL results. On the
two testbeds used here, Alrashdan (2026) benchmarked dispatching rules,
Thompson-sampling bandits and unconstrained PPO under machine breakdowns of
varying severity and found ShortestQueue ahead of PPO on cost-per-unit by
38–153% depending on the disruption level, with a corrected training protocol
(longer discount horizon, held-out checkpoint selection, a per-departure
throughput bonus) narrowing the gap without closing it; that study did not
consider constraints or evaluation mode, which are this paper's subject.
Rinciog and Meyer (2021) introduced FabricatioRL, the closest comparator
framework to our simulator. A parallel line bypasses RL entirely and learns interpretable
dispatching rules via genetic programming or hyper-heuristics (Ferreira et al.,
2022; Huang et al., 2025; Marques et al., 2025); we acknowledge it as an
alternative but do not benchmark against it.

### 2.2 Constrained MDPs and Lagrangian policy optimisation

The CMDP formulation is due to Altman (1999), who shows that for a finite CMDP
with K constraints under discounted cost there exists an optimal stationary
policy requiring at most K randomisations (Theorem 3.5), and notes more generally
that optimal policies for constrained problems require randomisation or
time-sharing between deterministic policies (Sect. 1.2). These results are stated
for a setting different from ours, which is finite-horizon and uses function
approximation; we use them as motivation for reporting stochastic as well as
greedy evaluation, not as an explanation of the learned policies' behaviour. The
modern policy-gradient pipeline begins with Constrained Policy Optimization
(Achiam et al., 2017). Reward Constrained Policy Optimization (Tessler et al.,
2019) introduces the multi-timescale Lagrangian approach we build on, and
Paternain et al. (2019) prove a zero duality gap that underwrites primal-dual
methods despite the non-convexity of policy optimisation. Stooke et al. (2020) is
the closest methodological precedent: the standard Lagrangian update behaves as
integral control on the violation signal, producing oscillation and overshoot,
and a PID controller on the multiplier damps it.

### 2.3 Evaluation practice in deep RL

Henderson et al. (2018) showed that reported deep-RL results are sensitive to
evaluation choices that are rarely stated, including the number of seeds and the
evaluation protocol, and Agarwal et al. (2021) showed that point estimates over a
handful of runs routinely misstate both the level and the uncertainty of an
algorithm's performance. Whether a stochastic policy is evaluated by sampling or
by its mode is one such choice; it is often left to a library default and seldom
reported. For a policy that is close to uniform the two modes measure different
objects, and this paper is an extended example of the consequence.

### 2.4 Positioning

The literature above analyses dual dynamics given a constraint signal. It has
comparatively little to say about whether the signal fed to the dual update is
the constraint the paper claims to impose. Our finding is that in a queueing
system with non-negligible flow time, the natural per-step implementation of an
episode-level throughput constraint is systematically biased, and that this bias
interacts with a one-sided dual update to produce a failure which, from aggregate
metrics alone, cannot be distinguished from the method not working. We are not
aware of a treatment of this issue in the constrained-RL-for-scheduling
literature.

## 3. Problem formulation

### 3.1 Multi-server flow shop

Consider a flow shop with N stages indexed s = 1, …, N. Stage s contains m_s
parallel servers, each with its own service-time distribution and processing
cost. Jobs arrive at stage 1 according to a Poisson process and traverse the
stages in order. At each stage, a routing decision selects which of the m_s
servers handles the job. Service times are exponentially distributed with stage-
and server-specific rates. Processing costs accrue per unit of busy time and
differ across servers; idle time and waiting time also incur cost. An episode is
one shift of H = 480 one-minute steps.

A routing policy maps the system state (queue lengths, in-service indicators,
accumulated costs) to a routing action a = (a₁, …, a_N), a joint tuple specifying
the full downstream route for an arriving job rather than a sequence of local
stage-by-stage decisions. The action space has size ∏_s m_s: 4 for the bakery
testbed (N = 2, m = (2, 2)) and 12 for electronics (N = 3, m = (2, 3, 2)). This
centralised full-route formulation simplifies credit assignment relative to a
per-stage MDP, at the cost of an action space that grows multiplicatively with
stage count. The observation vector concatenates stage queue lengths, per-server
in-service indicators and accumulated cost signals. The original manuscript
stated that it also contained the current dual multipliers. It does not; the
Lagrangian wrapper modifies the reward and leaves the observation unchanged, and
we correct the description here.

### 3.2 The conservative routing attractor

Let c_t denote the dollar cost accrued in step t: processing cost on busy
servers, idle cost on idle servers and waiting cost per queued job, summed over
servers, with C(τ) = Σ_t c_t the episode total. The environment reward is the
per-step scalarised signal of Eq. (1), transcribed from the implementation,

$$r_t = \kappa\left(-\,w_c\,\frac{c_t}{C_{\mathrm{norm}}} + w_t\,\frac{\dot n_t\,\Delta t}{N_{\mathrm{norm}}} - w_w\,\frac{W_t\,\Delta t}{W_{\mathrm{norm}}}\right)\qquad(1)$$

with (w_c, w_t, w_w) summing to one, κ = 10, Δt = 1 min, W_t the work in process
(queued plus in service) and (C_norm, N_norm, W_norm) fixed normalisation
constants, (2740, 20, 3220) on bakery and (4230, 55, 5100) on electronics, set
to typical episode totals so that each term is of order one per episode. Here ṅ_t
is the cumulative average throughput rate d_t / t, with d_t the departures up to
step t, not an instantaneous rate. The original manuscript described it as
instantaneous; this was incorrect, and the distinction matters in §4.2. For the
cost-only weights (1, 0, 0) used in the corrected cells, the episode return is
−κ C(τ)/C_norm, proportional to total dollar cost, so the objective of Eq. (2)
is optimised without approximation. For w_c near unity PPO solves the problem it was given and
shuts down expensive servers. Cost rate falls, throughput falls more, and
cost-per-unit rises. Sweeping w_c does not escape the attractor: down to
w_c = 0.5, cost-per-unit never falls below that of the LeastUtilised rule.

### 3.3 Constrained formulation

Let H = 480. Write TP(τ) for the number of departures in episode τ and u_i(τ) for
the realised utilisation of server i. The agent is posed the constrained problem
of Eq. (2),

$$\max_{\pi}\; \mathbb{E}_\pi\!\left[-C(\tau)\right] \quad \text{s.t.}\quad \mathbb{E}_\pi[\mathrm{TP}(\tau)] \ge T_{\min},\quad \mathbb{E}_\pi[u_i(\tau)] \ge U_{\min}\;\; \forall\, i \in F_{\mathrm{fast}}\qquad(2)$$

with T_min = 18 (bakery) or 50 (electronics) and U_min = 0.50. Both constraints
are on episode-level quantities. We state this more carefully than the original manuscript did, where the constraint was written as an episode count while the
implementation used a per-step rate; §4.2 describes the consequences.

Three distinct constraint objects appear in this paper, and much of what follows
turns on the differences between them. The first is the expectation constraint
of Eq. (2), the problem stated. The second is the training surrogate: the primal
update penalises the exact per-episode Lagrangian of Eq. (2), but the dual
update reads a hinged per-episode shortfall, max(0, T_min − TP(τ)), rather than
the signed expected slack (§4.3), so the multipliers respond to the frequency
and depth of violations, not to the expected constraint value. The third is the
selection and reporting criterion: a checkpoint qualifies if each constraint
holds on at least 16 of 20 validation episodes, and results report the fraction
of test episodes on which both hold. Selection therefore imposes two per-episode
chance constraints, P(TP(τ) ≥ T_min) ≥ 0.8 and P(min_i u_i(τ) ≥ U_min) ≥ 0.8,
and reporting measures the joint event. Both are stricter than Eq. (2) for any
policy whose throughput varies between episodes (§3.4 quantifies the gap), and
§6.7 shows a configuration that satisfies Eq. (2) while failing them. The original manuscript treated the three as interchangeable. They are
not, and §4.2 shows that the version implemented in training was a fourth object
that coincides with none of them.

### 3.4 The objective differs from the reported metric

The headline metric throughout this literature, and in our own work, is
cost-per-unit (CPU), computed here as the per-episode ratio C(τ)/TP(τ) averaged
over episodes, E[C(τ)/TP(τ)], whereas the CMDP objective above minimises
expected total cost subject to a throughput floor. The two differ, and on our
testbeds they appear to diverge in a direction that matters. Interpolating
linearly between the mean cost and mean throughput of the CostMinimising and
ShortestQueue baselines on bakery gives a marginal cost of about $40 per
additional unit against an average of about $139, which would make cost-per-unit
decreasing in throughput across the operating range. If so, a total-cost
minimiser has every incentive to sit at the throughput floor, where the
interpolated cost-per-unit (approximately $151) is worse than ShortestQueue's
$138.66, and the constrained optimum as posed would be dominated on the metric
we report. This is a plausibility argument from two points, not a measurement,
and we present it as motivation.

Its consequence, if it holds, is that any apparent success of the constrained
agent on cost-per-unit must be delivered by something other than the objective.
In the original manuscript a candidate is the checkpoint-selection rule.
Requiring at least 16 of 20 validation episodes to satisfy TP ≥ 18, with a
per-episode standard deviation of about 2.0 units, implies a mean throughput of
about 19.7, an effective floor above the nominal 18 but less than one standard
deviation above it. We note this here and return to it in §6.6 and §8. The ratio-objective formulation it motivates is future work
rather than a claim of this paper.

## 4. Method

### 4.1 Lagrangian PPO and the augmented reward

The constrained problem is solved with PPO as the inner loop and a Gymnasium
wrapper that augments the per-step reward with penalties. The augmented reward
used is given by Eq. (3),

$$\tilde r_t = r_t - \lambda_U\, g_{U,t} - \lambda_T\, g_{T,t}\qquad(3)$$

where r_t is the environment reward of Eq. (1) rather than the bare cost term. In
the original manuscript both the abstract and the formulation section stated
that the scalarised reward was replaced by single-objective cost minimisation.
That was a misdescription of the implementation: every constrained variant
augments the shaped reward with weights (0.8, 0.1, 0.1) active. We correct the
description here, and §6.1 reports an ablation isolating the effect of the
shaping terms. Within confidence intervals there is none, but the description
had to be corrected regardless.

### 4.2 The slack signal

The dual variables are updated from a slack signal g. The original
implementation used, at every step after a 100-step warm-up, the signal of
Eq. (4),

$$g_{T,t} = \max\!\left(0,\; \frac{T_{\min}}{H} - \frac{d_t}{t}\right)\qquad\text{(cumulative-rate slack)}\qquad(4)$$

with d_t the cumulative departures up to step t. This differs from the
constraint of Eq. (2). It compares a cumulative average rate against the floor
at every instant. In a flow shop whose flow time is a sizeable
fraction of the horizon, the cumulative rate is below the floor for a long prefix
of every episode, whatever the policy does, because the line is filling.
Measured crossing times under ShortestQueue are t = 102–207 on bakery and
t = 129–227 on electronics. A policy achieving TP = 50 on electronics, which
meets the stated constraint, does not cross until t ≈ 480, so it never satisfies
the training signal within the horizon. The utilisation slack was defined
analogously, g_{U,t} = Σ_{i∈F_fast} max(0, U_min − ū_{i,t}), with ū_{i,t} the
cumulative utilisation of server i up to step t; it carries the same fill-phase
bias, although servers become busy quickly and the effect is milder.

The one-sided dual update of Eq. (5), applied once per episode to each
multiplier with the mean taken over the post-warm-up steps of that episode,

$$\lambda \leftarrow \operatorname{clip}\!\left(\lambda + \eta\,\operatorname{mean}_t\, g_t,\; 0,\; \lambda_{\max}\right)\qquad(5)$$

which the original manuscript described as "a violation-driven penalty
accumulator with monotone growth, not true primal-dual ascent", turns this bias
into a structural outcome. The measured mean per-step g_T for ShortestQueue is
0.0003–0.0029 on bakery and 0.0010–0.0064 on electronics; at η_T = 4000 this is a
per-episode increment of 1–26 even for an oracle. Over 3,333 training episodes,
λ_T reaches its cap of 20,000 for any policy. For realistic electronics policies
(TP ≈ 35–46), the increment is 40–80 per episode and the cap is reached after
250–500 episodes, or 8–15% of training, consistent with where the original manuscript observed it. Saturation therefore reflects the signal, not the policy.

### 4.3 Corrected slack

We replace the per-step signal with the signed pro-rata decomposition of the
episode Lagrangian, Eq. (6),

$$p_t = \lambda_T\left(\frac{T_{\min}}{H} - \delta_t\right) + \lambda_U \sum_{i\in F_{\mathrm{fast}}}\left(U_{\min} - b_{i,t}\right)\qquad(6)$$

where δ_t is the number of departures in step t and b_{i,t} ∈ {0, 1} indicates
whether server i is in service at step t. The augmented reward is r̃_t = r_t − p_t
at every step, without warm-up. Summed over the episode, Eq. (6) telescopes to
Eq. (7),

$$\sum_{t=0}^{H-1} p_t = \lambda_T\left(T_{\min} - \mathrm{TP}(\tau)\right) + \lambda_U\, H \sum_{i\in F_{\mathrm{fast}}}\left(U_{\min} - \bar u_i(\tau)\right)\qquad(7)$$

the Lagrangian penalty corresponding to the expectation constraint of Eq. (2),
expressed in the same episode-level quantities, TP(τ) and ū_i(τ), that
validation and test evaluate under the per-episode criterion of §3.3; here
ū_i(τ) is the realised utilisation of server i over the episode.
There is no fill-phase bias; an action's contribution depends only on the
departures and busy time it causes. The multipliers are updated once per
episode, at the episode boundary, from the hinged episode-level slack of
Eqs. (8) and (9),

$$\lambda_T \leftarrow \operatorname{clip}\!\left(\lambda_T + \eta_T\,\frac{\max\!\left(0,\; T_{\min} - \mathrm{TP}(\tau)\right)}{H},\; 0,\; \lambda_{T,\max}\right)\qquad(8)$$

$$\lambda_U \leftarrow \operatorname{clip}\!\left(\lambda_U + \eta_U \sum_{i\in F_{\mathrm{fast}}} \max\!\left(0,\; U_{\min} - \bar u_i(\tau)\right),\; 0,\; \lambda_{U,\max}\right)\qquad(9)$$

The division by H keeps the throughput slack in the per-step rate units of
Eq. (4), so that η_T, λ_T's initial value and its cap carry over unchanged from
the original manuscript; the utilisation slack is already in per-step units.
Under Eqs. (8) and (9) the ratchet advances only on episodes that violate, so
saturation is no longer guaranteed for an oracle, which is the property the
mechanism ablation of §6.1 tests. Two consequences of retaining the hinge
should be stated at the outset. The update is still monotone: a multiplier never
decreases, so it cannot return toward zero once the constraint is slack, and
under a stochastic policy that violates on a persistent fraction of episodes it
continues to grow, at a rate set by the violation frequency rather than by the
expected slack. Eqs. (8) and (9) are therefore not a convergent dual ascent
either; they remove the guaranteed divergence of Eq. (4), not the monotonicity.
A symmetric variant uses the signed slack (T_min − TP(τ))/H in Eq. (8) and, in
Eq. (9), the summed shortfall when any constrained server violates and the
negative margin of the binding server otherwise, so that a multiplier can fall
back toward zero once its constraint is slack. §6.4 explains why the scale of
the multipliers matters as much as the hinge, and §6.7 reports the symmetric
variant at full budget with multipliers on the scale of the reward
(Protocol Amendment R2).

### 4.4 Evaluation of a stochastic policy

The original protocol evaluated with `deterministic=True`, that is, greedy action
selection. PPO optimises a stochastic policy, and for a constrained problem the
optimum may itself be randomised (§2.2). More immediately, if the learned
distribution is close to uniform, its argmax is a deterministic policy selected
by small and unstable differences in logits, and evaluating it measures those
differences rather than the policy. We therefore report both evaluation modes
throughout, select checkpoints under the same mode we report, and in §6.5 measure
directly how far the learned policies are from uniform and how stable their
argmax is to seed and to small perturbations of the logits. Protocol Amendment
R1, committed before any full-budget run of this paper, records the change and
its rationale. We call the resulting discrepancy an evaluation-mode artefact
rather than an error: greedy evaluation is a legitimate choice when the
deployment target is a deterministic controller (§7.3), and the problem is
reporting it for a near-uniform policy without saying so.

## 5. Experimental protocol

### 5.1 Testbeds

**Bakery (BK50).** Two stages, two servers per stage, 4 actions. Stage 1 has
Kneader B (fast, mean service 14.2 min, $1.5/min) and Kneader A (slow, 16.7 min,
$1.0/min); stage 2 has Oven A convection (fast, 36.6 min, $1.5/min) and Oven C
deck (slow, 47.9 min, $1.0/min). Service times are calibrated to the BK50 subset
of the small and medium-sized bakery production dataset (Babor & Hitzmann, 2022).
Constrained fast servers F_fast = {Kneader B, Oven A}; T_min = 18.

**Electronics (3-stage).** Three stages with 2, 3 and 2 servers, 12 actions; a
synthetic instance with asymmetric server capacities and a tighter capacity
floor. Stage 1: Mounter A (fast, expensive) and Mounter B. Stage 2: Reflow 1
(fast, expensive), Reflow 2 (medium), Wave (slow, cheap). Stage 3: AOI Scanner
(fast, structurally over-capacitated) and Manual Inspection (slow). F_fast =
{Mounter A, Reflow 1}; the AOI Scanner is excluded because LeastUtilised itself
reaches only 42% utilisation there and a 0.50 floor would be infeasible.
T_min = 50. Both testbeds are those of the FlexFlowSim simulator described in
Alrashdan (2026), without that study's breakdown extensions; service-time
distributions and costs are in the FlexFlowSim-CPPO configuration files.

### 5.2 Baselines

Six dispatching rules are evaluated on the 50 test episodes (Table 1). Four are
state-aware. ShortestQueue minimises summed instantaneous load (queue length plus
in-service indicator) across the full route. LeastUtilised weights each server's
load by its mean service time. CostMinimising minimises summed processing cost
per unit time, ignoring queue state. FastServerFirst always selects the fastest
server at every stage. Two are stateless and are included because the learned
policies turn out to be close to them (§6.5): UniformRandom draws a route
uniformly at each arrival, and RoundRobin cycles through the routes in order.
ShortestQueue dominates on cost-per-unit while satisfying both constraints and is
the primary comparator.

**Table 1.** Dispatching baselines, mean ± 95% confidence interval (CI) over 50
test episodes. Joint sat. is the fraction of episodes satisfying both constraints.

| Testbed / rule | Throughput | Cost/unit | Util. sat. | TP sat. | Joint sat. |
|---|---|---|---|---|---|
| Bakery, LeastUtilised | 19.64 ± 0.49 | $145.63 ± 4.05 | 100% | 90% | 90% |
| Bakery, ShortestQueue | 20.48 ± 0.55 | $138.66 ± 4.26 | 100% | 96% | 96% |
| Bakery, CostMinimising | 9.00 ± 0.22 | $264.14 ± 9.12 | 0% | 0% | 0% |
| Bakery, FastServerFirst | 12.06 ± 0.44 | $230.05 ± 8.71 | 100% | 0% | 0% |
| Bakery, RoundRobin | 20.38 ± 0.54 | $137.55 ± 5.06 | 94% | 94% | 88% |
| Bakery, UniformRandom | 19.88 ± 0.43 | $141.05 ± 5.05 | 98% | 92% | 90% |
| Electronics, LeastUtilised | 56.46 ± 0.85 | $80.47 ± 1.60 | 100% | 98% | 98% |
| Electronics, ShortestQueue | 61.42 ± 0.99 | $73.25 ± 1.55 | 100% | 100% | 100% |
| Electronics, CostMinimising | 15.58 ± 0.27 | $253.07 ± 7.05 | 0% | 0% | 0% |
| Electronics, FastServerFirst | 30.16 ± 0.48 | $157.62 ± 3.96 | 100% | 0% | 0% |
| Electronics, RoundRobin | 55.10 ± 1.08 | $78.90 ± 1.54 | 92% | 88% | 86% |
| Electronics, UniformRandom | 54.40 ± 1.22 | $79.91 ± 2.20 | 84% | 88% | 78% |

F_fast and U_min = 0.50 are identified from a one-shot LeastUtilised run on each
testbed. This anchors the feasible region on the heuristic family the agent is
compared against, and Table 1 shows the consequence: at the thresholds used,
stateless routing satisfies both constraints on 78–90% of episodes, so the
constraints are not demanding. §6.6 reports how satisfaction changes as the
thresholds are tightened, and §8 discusses the limitation. The original manuscript stated
that random routing was dominated by LeastUtilised and ShortestQueue without
reporting it; Table 1 shows that this was true but uninformative, since the gap
between LeastUtilised and UniformRandom on electronics is $0.56 per unit.

### 5.3 Training and evaluation protocol

Five pre-committed seeds [42, 7, 2024, 123, 999]; 1.6M timesteps on electronics
and 1.5M on bakery; checkpoints every 100K. Validation on seeds [10000, 10020):
among checkpoints satisfying both constraints on at least 16 of 20 episodes, the
one with the lowest mean cost-per-unit is selected; if none qualifies, the lowest
mean cost-per-unit overall is selected and the run is marked `fallback`. Test on
seeds [11000, 11050). Cost-per-unit is computed per episode and averaged (mean of
ratios), and joint satisfaction is the fraction of episodes on which both
constraints hold. Confidence intervals in Tables 2–4 are 95% two-sided
t-intervals across seeds, and in Table 1 across episodes, so the two are not
directly comparable. PPO is the Stable-Baselines3 implementation (Raffin et al., 2021) with
the hyperparameters of the original manuscript (learning rate 3×10⁻⁴,
n_steps 2048, batch 64, 10 epochs, γ = 0.99, GAE λ = 0.95, clip 0.2, entropy
coefficient 0.01). Dual variables: λ_U initialised at 5 with step 20 and cap 500;
λ_T initialised at 200 with step 4000 and cap 20,000.

Two protocol changes are recorded in `protocol_amendment_r1.md`, committed
before the full-budget runs it governs. First, both evaluation modes are computed
and reported for every checkpoint and every test, and checkpoint selection reads
the stochastic validation episodes so that selection and reporting share a
criterion. The original manuscript's argmax selection is still computed and
reported alongside. Second, full-budget training is executed in 400K-step
segments to fit the execution environment. Policy weights, optimiser state, dual
variables and the dual episode counter are carried across segment boundaries;
only PPO's in-flight rollout buffer is flushed, three times in 1.6M steps, or
about 0.4% of rollouts.

One deviation from the amendment is recorded in `protocol_deviations.md`
(R2-1). The amendment specified seeded action sampling for stochastic
evaluation; the training pipeline sampled from the unseeded process generator
instead, so its validation and test draws are single draws that cannot be
regenerated. Checkpoint selection stands as executed on the archived validation
draws. Every stochastic test figure in this paper is a seeded re-evaluation of
the selected checkpoints, with the generator seeded by the episode seed before
each episode so that the action stream is reproducible and common across
policies, and with per-episode records archived. Argmax figures are unaffected
and reproduce the pipeline's values to the cent, the fidelity check on the
re-evaluation. The pipeline's own stochastic draws differ from the
seeded ones by at most $1.2 per unit and 5.6 points of satisfaction at the cell
level, and no comparison changes direction.

A third protocol document, `protocol_amendment_r2.md`, was committed before the
runs of §6.7 were launched. It defines one further cell on electronics,
`cost-episode-sym`: the cost reward and episode slack of the corrected cell with
the symmetric dual update of §4.3 and multipliers initialised, stepped and
capped on the scale of the reward (λ_T: 0.1, η_T = 2.0, cap 10; λ_U: 0.002,
η_U = 0.01, cap 0.2). The values were derived before any run by two independent
arguments that agree: the shadow price of a unit of throughput implied by the
baseline interpolation of §3.4 (about 0.1 reward units) and of a busy-step of a
fast server (about 0.002), and the factor of about 2,400 by which the inherited
initial multipliers had to be divided to bring the penalty on a typical episode
to a tenth of the return. The amendment also fixes the pipeline remedy for
deviation R2-1 (seeded evaluation, per-episode records), lists the comparisons
to be reported, and states three possible outcomes with the wording each would
receive. No hyperparameter of the cell was changed after results were seen.

The mechanism ablation of §6.1 predates the amendment and motivated it. It was
run at a reduced budget of 400K steps with argmax checkpoint selection, as in the
original protocol, and both evaluation modes recorded. It is not repeated at full
budget: its purpose is attribution, not performance measurement, the effect it
attributes is categorical (10 of 10 seeds versus 0 of 10), and repeating it would
cost approximately twelve compute-hours to restate a result already established.
The full-budget comparison of §6.2 runs only the two diagonal cells of the
ablation, the original implementation and the fully corrected one; the ablation
shows the base reward to be immaterial within confidence intervals, so the
off-diagonal cells were not run at full budget.

Comparisons in §6.2 use a paired hierarchical bootstrap on the per-episode test
records (Appendix A.5). Both sources of uncertainty are resampled: training
seeds, with replacement, within each learned cell; and test episodes, with
replacement, jointly across the policies being compared, so that a learned
policy and a baseline are always compared on the same resampled episodes. This
replaces the one-sample t-tests of an earlier draft, which treated the baseline
means as known constants and overstated the evidence. Ten tests form the
pre-specified family on each testbed, the corrected cell against ShortestQueue,
LeastUtilised, RoundRobin, UniformRandom and the original signal on
cost-per-unit and on joint satisfaction, and are adjusted by the Holm procedure
at α = 0.05. Comparisons involving the control cell alone are reported without
adjustment as descriptive. The cell of §6.7 forms its own family of twelve
tests, against the same four rules and both full-budget cells.

## 6. Results

### 6.1 Mechanism ablation

A 2×2 design crosses the base reward {shaped (0.8, 0.1, 0.1), cost (1, 0, 0)}
with the slack signal {cumulative-rate, episode-level}, five seeds per cell, 400K
steps, on electronics, with argmax checkpoint selection and both evaluation modes
recorded. Table 2 reports the four cells.

**Table 2.** Mechanism ablation, electronics, 400K steps, 5 seeds per cell,
argmax selection. Reference: ShortestQueue $73.25 per unit at TP 61.4.

| Cell | argmax CPU ± CI | argmax joint sat. | stochastic CPU ± CI | stochastic joint sat. | λ_T at cap |
|---|---|---|---|---|---|
| shaped + cumulative-rate (original) | $152.22 ± 84.67 | 0.8% | $82.37 ± 2.80 | 63.6% | 5/5 |
| shaped + episode | $122.26 ± 13.46 | 0.0% | $78.50 ± 1.27 | 94.4% | 0/5 |
| cost + cumulative-rate | $155.18 ± 88.17 | 0.0% | $80.44 ± 1.01 | 75.2% | 5/5 |
| cost + episode (the Lagrangian of §4.3) | $155.35 ± 80.54 | 0.0% | $78.73 ± 1.57 | 93.6% | 0/5 |

Four observations follow from Table 2.

Saturation tracks the slack signal. All ten seeds trained with the
cumulative-rate signal end at the λ_T cap of 20,000; all ten trained with the
episode-level signal end between 908 and 2,833, an order of magnitude lower.
There is no overlap and no exception at the seed level, and the pattern holds at
full budget on both testbeds (Figure 1 and §6.2: 10 of 10 control seeds at the
cap, 0 of 10 corrected seeds within a factor of 2.5 of it). The prediction of §4.2 is
confirmed as a property of the design rather than a tendency. One qualification
comes from the full-budget runs. At 400K the corrected multiplier appears to
plateau, with training slack identically zero for hundreds of consecutive
episodes; over 1.5–1.6M steps it continues to climb slowly, to 2,525–7,858, as
the policy occasionally violates and the hinged update ratchets. It does not
saturate, although it has not converged either.

Evaluation mode accounts for the reported pass/fail verdict. Every cell,
including the original implementation, reproduces the original manuscript's
failure pattern under argmax evaluation, with $122–155 per unit and essentially
zero joint satisfaction, and every cell passes on the same checkpoints under
stochastic evaluation. The between-seed confidence intervals also collapse, from
±13–88 to ±1.0–2.8. §6.5 shows why: the learned distributions are close to
uniform, and the argmax of a near-uniform distribution varies arbitrarily
between seeds.

The slack bias damages the learned policy as well as the dual variables, on the
harder testbed. Even under stochastic evaluation, the cumulative-rate cells
remain $1.7–3.9 per unit worse and 18–31 percentage points lower in joint
satisfaction than their episode-slack counterparts at 400K, and both gaps
survive at full budget on electronics (§6.2). Once λ_T is pinned at its cap
the policy gradient is dominated by penalty avoidance, and the resulting policy
does not recover. On bakery the same saturation produces no measurable damage
(§6.2), so the claim is testbed-dependent.

The base reward has little effect. Shaped and cost-only cells are within
confidence intervals of each other in both evaluation modes. The misdescription
identified in §4.1 is a documentation error rather than a driver of the result.

### 6.2 Full-budget comparison

Two cells were run at the full pre-registered budget: the original
implementation (shaped reward, cumulative-rate slack) as control, and the
corrected formulation of §4.3 (cost reward, episode slack). Five seeds each,
1.6M steps on electronics and 1.5M on bakery, checkpoint selection on stochastic
validation per Amendment R1, and both evaluation modes on the 50 test episodes.
Table 3 reports the results and Table 4 the paired comparisons.

**Table 3.** Full-budget comparison. CI is the 95% two-sided t-interval across
seeds. "Sat." is joint constraint satisfaction over test episodes. References
(Table 1): ShortestQueue $73.25 / 100% and RoundRobin $78.90 / 86% on
electronics; ShortestQueue $138.66 / 96% and RoundRobin $137.55 / 88% on bakery.

| Testbed | Cell | stochastic CPU ± CI | stoch. sat. | val.-satisfied | λ_T at cap | argmax CPU ± CI | argmax sat. |
|---|---|---|---|---|---|---|---|
| electronics | corrected | $78.10 ± 1.64 | 95.2% | 5/5 | 0/5 | $162.94 ± 75.56 | 0.0% |
| electronics | original signal | $82.12 ± 1.64 | 70.0% | 5/5 | 5/5 | $196.04 ± 42.87 | 0.0% |
| bakery | corrected | $141.07 ± 0.68 | 94.4% | 5/5 | 0/5 | $165.66 ± 46.86 | 54.4% |
| bakery | original signal | $139.61 ± 2.04 | 86.4% | 5/5 | 5/5 | $180.37 ± 40.63 | 13.2% |

**Table 4.** Paired hierarchical bootstrap (10,000 resamples of seeds and
episodes) of the corrected cell, evaluated stochastically, against each
comparator. Entries are the difference in means (corrected minus comparator)
with 95% percentile interval; Δsat. is in percentage points. p is two-sided;
asterisks mark the comparisons that survive the Holm adjustment over the ten
tests on each testbed, or over the twelve tests of the symmetric cell's own
family.

| Testbed | Comparator | ΔCPU ($) | p | Δsat. (pp) | p |
|--------|-----------|------------------|-------|------------------|-------|
| electronics | ShortestQueue | +4.85 [+2.94, +6.88] | < 0.001* | −4.8 [−9.2, −1.2] | 0.005* |
| electronics | LeastUtilised | −2.37 [−4.11, −0.42] | 0.017 | −2.8 [−7.2, +2.0] | 0.24 |
| electronics | RoundRobin | −0.79 [−2.74, +1.37] | 0.43 | +9.2 [−0.8, +19.6] | 0.073 |
| electronics | UniformRandom | −1.81 [−3.96, +0.41] | 0.10 | +17.2 [+6.0, +29.2] | 0.002* |
| electronics | original signal | −4.01 [−5.87, −1.99] | < 0.001* | +25.2 [+12.0, +38.8] | < 0.001* |
| bakery | ShortestQueue | +2.41 [−1.69, +6.55] | 0.25 | −1.6 [−7.6, +5.2] | 0.64 |
| bakery | LeastUtilised | −4.56 [−9.45, +0.50] | 0.079 | +4.4 [−5.2, +14.8] | 0.40 |
| bakery | RoundRobin | +3.52 [−1.02, +8.06] | 0.13 | +6.4 [−2.4, +16.4] | 0.18 |
| bakery | UniformRandom | +0.02 [−3.93, +3.88] | 0.99 | +4.4 [−4.0, +14.0] | 0.35 |
| bakery | original signal | +1.46 [−1.79, +4.78] | 0.38 | +8.0 [0.0, +17.2] | 0.063 |
| electronics, symmetric cell (§6.7) | ShortestQueue | +8.65 [+5.86, +11.39] | < 0.001* | −25.2 [−34.0, −16.8] | < 0.001* |
| electronics, symmetric cell (§6.7) | LeastUtilised | +1.44 [−1.39, +4.14] | 0.32 | −23.2 [−32.4, −14.8] | < 0.001* |
| electronics, symmetric cell (§6.7) | RoundRobin | +3.01 [+0.17, +5.86] | 0.036 | −11.2 [−22.4, +0.8] | 0.073 |
| electronics, symmetric cell (§6.7) | UniformRandom | +1.99 [−1.05, +4.90] | 0.20 | −3.2 [−16.4, +10.4] | 0.65 |
| electronics, symmetric cell (§6.7) | corrected (hinged) | +3.81 [+0.92, +6.61] | 0.012 | −20.4 [−30.0, −11.2] | < 0.001* |
| electronics, symmetric cell (§6.7) | original signal | −0.21 [−3.22, +2.80] | 0.89 | +4.8 [−9.6, +19.2] | 0.54 |

Four results follow. First, every seed in every cell satisfies the validation
criterion, including the unmodified original signal, which under the original
protocol satisfied it on 0 of 5 electronics seeds and 4 of 5 bakery seeds.
Evaluation mode alone reverses the original pass/fail verdict on electronics and
completes the set on bakery. Second, on electronics the slack correction is worth
25 points of joint satisfaction and $4 per unit against the original signal,
and both differences survive adjustment. Third, on bakery the slack correction
makes no measurable difference in either metric. The multiplier still saturates
on all five control seeds, so the mechanism of §4.2 is present, but on the
easier problem the saturated penalty does not damage the policy enough to show.
We report this as a null result, not as a partial success.

Fourth, and most important for interpretation, the corrected agent does not
separate from stateless routing on cost-per-unit. On electronics the paired
difference against RoundRobin is −$0.79 [−2.74, +1.37] and against UniformRandom
−$1.81 [−3.96, +0.41]; the intervals are wide enough to contain differences of a
few dollars in either direction, so this is a failure to detect, not a
demonstration of equality (§8). On joint satisfaction the corrected agent is
ahead of UniformRandom by 17 points, which survives adjustment, and ahead of
RoundRobin by 9 points, which does not. An earlier draft of this paper reported
both satisfaction advantages as significant on one-sample t-tests against the
baseline means; once the baselines' own episode-to-episode variance is
propagated, the RoundRobin comparison does not hold up. On bakery no comparison
against any rule is significant. ShortestQueue, a state-aware rule, remains
ahead of every learned policy: 6.6% cheaper per unit on electronics with 100%
constraint satisfaction, and the only comparator on that testbed the corrected
agent is significantly worse than on both metrics.

A second feature of Table 3 is the between-seed dispersion. Under argmax
evaluation the electronics confidence intervals are ±43 to ±76; under stochastic
evaluation of the same checkpoints they are ±1.6. The high seed-to-seed
instability that the original manuscript reported on electronics was, to a first
approximation, the variance of an argmax taken over near-uniform action
distributions (§6.5).

### 6.3 Archival re-evaluation of the original manuscript's checkpoints

The original manuscript's checkpoints were re-evaluated under Amendment R1
§A2.1 without retraining: V4, the one-sided Lagrangian of §4.2, and V6, a
PID-Lagrangian comparator with exponentially smoothed slack (Stooke et al.,
2020), on both testbeds and every archived seed (five for V4, four of five for
V6). We ask two questions. The first (Q1)
concerns the exact checkpoint the original manuscript selected and reported,
evaluated stochastically. The second (Q2) concerns the checkpoint that stochastic
selection would have chosen, and what it scores.

Applying the original argmax selection rule to the archived validation sweep
reproduces the originally selected checkpoint in 18 of 18 runs and the reported
test figures to the cent: V4 bakery $149.30 ± 27.73 and V4 electronics
$112.71 ± 13.46 are Tables 3 and 5 of the original manuscript. The pipeline
evaluated here is the one that produced those figures. Table 5 reports both
questions for all four archived runs.

**Table 5.** Archival re-evaluation. Q1 columns evaluate the originally selected
checkpoint in both modes; Q2 applies stochastic selection. n = 4 for V6 because
one seed's checkpoints were not archived.

| Run | n | Q1 argmax CPU ± CI (original) | Q1 sat. | Q1 stochastic CPU ± CI | Q1 sat. | Q2 stochastic CPU ± CI | Q2 sat. | Q2 val.-satisfied |
|---|---|---|---|---|---|---|---|---|
| V4 electronics | 5 | $112.71 ± 13.46 | 6% | $81.35 ± 2.25 | 71% | $81.59 ± 2.68 | 74% | 5/5 (original 0/5) |
| V6 electronics | 4 | $106.75 ± 23.87 | 22% | $83.94 ± 9.92 | 58% | $80.83 ± 3.40 | 72% | 3/4 |
| V4 bakery | 5 | $149.30 ± 27.73 | 69% | $140.56 ± 3.15 | 91% | $139.60 ± 2.49 | 80% | 5/5 |
| V6 bakery | 4 | $141.57 ± 22.71 | 74% | $140.46 ± 1.60 | 84% | $139.94 ± 3.72 | 83% | 4/4 |

The five electronics V4 checkpoints that the original manuscript reported as
failing systematically, at $113.31, $112.08, $95.38, $118.35 and $124.44, score
$83.78, $81.23, $78.68, $81.73 and $81.33 as stochastic policies, with joint
constraint satisfaction of 64–88%. The checkpoint files are unchanged. Had the
original protocol selected on stochastic validation, all five would have been
reported as `satisfied` at $81.59 ± 2.68. The original negative result on
electronics is therefore reproduced and then reversed on the original artefacts;
the retrained runs of §6.2 confirm the same result under a corrected training
signal. Note that these stochastic scores are, once again, at the level of
RoundRobin and UniformRandom in Table 1.

The bakery rows show the complementary pattern. Argmax and stochastic scores are
within a few dollars of each other on most seeds, and the original argmax
selection already satisfied validation on 4 of 5 V4 seeds. Greedy extraction from
the learned policies finds a feasible deterministic policy often enough on
bakery and almost never on electronics. This is not because no feasible
deterministic policy exists on electronics; ShortestQueue is one, and a better
one than anything learned. It is because the learned distributions on electronics
are close enough to uniform that their argmax is arbitrary (§6.5).

### 6.4 Multiplier dynamics

Figure 1 shows the throughput multiplier over training for every full-budget
run under both slack signals.

![](fig_lambda_T.png){width=16cm}\

**Figure 1.** λ_T over training, five seeds per cell, both testbeds, log scale.
Under the original cumulative-rate signal the multiplier ramps monotonically to
its 20,000 cap on every seed, at 0.19–0.24M steps on electronics (12–15% of the
budget, matching the original manuscript's Figure 3) and at 0.59–0.65M on bakery
(39–43%), and is flat thereafter. The difference in saturation time between
testbeds is what §4.2 predicts: the fill-phase gap is larger where flow time is a
larger fraction of the horizon and the floor sits closer to the achievable rate.
Under the corrected episode signal the multiplier rises slowly, advancing only on
episodes that violate, and ends between 2,525 and 7,858, never within a factor of
2.5 of the cap.

Selected checkpoints do not cluster: under stochastic selection they range from
100K to 1.6M in both cells on both testbeds, so we make no claim that the
original budget was too long or too short.

The magnitude argument of the original manuscript also needs correcting, and
the correction applies to the corrected cell as much as to the control. The
original manuscript compared the integrated penalty at saturation with "the
unconstrained per-episode cost reward (−$2,000 to −$4,000)" and concluded a
ratio of 35–70×. That comparison mixed units. Dollars never enter the reward: by
Eq. (1) the per-episode base return in the units the agent optimises is
−κ C(τ)/C_norm, which is −10.1 to −10.6 on both testbeds. Against this, the
penalty of Eq. (7) evaluated on the test episodes of the corrected runs is
already 190–300 times larger in magnitude at the initial multipliers
(λ_T, λ_U) = (200, 5), and 1,200–10,000 times larger at the multipliers reached
by the end of training (λ_T = 2,525–7,858, λ_U = 18–255). For the control cell
at saturation the ratio is of order 10⁴. Two features of the corrected cell's
term matter for what follows. Its sign is negative on almost every episode, because
both constraints are over-satisfied by any load-spreading policy (mean
throughput 57.0 against a floor of 50 on electronics and 20.2 against 18 on
bakery; mean fast-server utilisation 0.79–0.91 against 0.50), so it acts as a
reward for surplus throughput and surplus fast-server
busy time rather than as a penalty. And it cannot shrink, because Eqs. (8) and
(9) never decrease a multiplier; the equilibrium value of a multiplier on a
slack constraint is zero, and the trained values are two to four orders of
magnitude above the scale of the cost term. The comparison of returns, if
anything, understates the imbalance seen by the policy gradient: the cost term
of Eq. (1) differs between the routing actions available at a decision by about
10⁻³ per step, whereas a single departure moves the per-step term of Eq. (6) by
λ_T, between 200 and 7,858. In every one-sided configuration of this study,
therefore, the original and the corrected alike, the objective the agent
actually optimised was, to within a fraction of a percent, throughput and
fast-server utilisation, and the cost term of Eq. (2) was numerically
irrelevant.

This is the reason the corrected agent could not be expected to learn
cost-efficient routing, and it is independent of the two artefacts. It follows
from two design choices inherited unchanged from the original manuscript: the
multiplier initial values and step sizes, which were set in the units of the
per-step signal of Eq. (4) and are large relative to a reward whose episode
return is of order ten; and the hinge in the dual update. The experiment that tests
the method proper is the symmetric variant of §4.3 with the multipliers
initialised and stepped on the scale of the reward, so that they can settle at
zero when the constraints are slack and the cost term can be felt. §6.7 reports
it.

### 6.5 Structure of the learned policies

For the selected checkpoint of every full-budget run we rolled out the stochastic
policy on ten test episodes and recorded, at every decision, the entropy of the
action distribution and its largest probability. Table 6 reports the entropy
normalised by ln(number of routes), the effective number of routes exp(H), and
the mean top-1 probability. A uniform router has normalised entropy 1.0, effective
routes equal to the number of routes, and top-1 probability 1/12 = 0.083 on
electronics or 0.25 on bakery. The table also reports three measures of how
much the argmax of these distributions means, all computed on a common set of
2,395 states visited under uniform-random routing on the first five test
episodes: the mean gap between the largest and second-largest action
probability; the rate at which two seeds' argmax policies agree on the same
state, against a chance level of 1/12 or 1/4; and the fraction of states at
which adding Gaussian noise of standard deviation 0.1 to the logits changes the
argmax.

**Table 6.** Action-distribution statistics of the selected checkpoints, ranges
or means across five seeds. Uniform reference: normalised entropy 1.00;
effective routes 12 (electronics) or 4 (bakery); top-1 probability 0.083 or
0.25; cross-seed argmax agreement 8.3% or 25%. The last three columns are
computed on 2,395 common states; the gap is a mean with the per-seed range in
parentheses.

| Testbed | Cell | norm. entropy | eff. routes | top-1 prob. | top-1 − top-2 gap | argmax agreement | flip rate (σ = 0.1) |
|--------|--------|--------|----------|--------|-------------|--------|--------|
| electronics | corrected | 0.94–0.97 | 10.4–11.3 of 12 | 0.13–0.18 | 0.021 (0.008–0.029) | 18.1% | 25.5% |
| electronics | original signal | 0.96–0.98 | 10.9–11.5 of 12 | 0.13–0.18 | 0.019 (0.014–0.026) | 9.2% | 24.3% |
| bakery | corrected | 0.91–0.98 | 3.5–3.9 of 4 | 0.31–0.44 | 0.154 (0.021–0.281) | 39.4% | 11.4% |
| bakery | original signal | 0.95–0.99 | 3.7–3.9 of 4 | 0.29–0.40 | 0.113 (0.027–0.214) | 33.4% | 15.3% |
| electronics | symmetric, rescaled (§6.7) | 0.63–0.83 | 4.8–7.8 of 12 | 0.28–0.48 | 0.179 (0.070–0.287) | 9.7% | 10.2% |

The learned policies are near-uniform routers with a modest tilt. On electronics
the most probable route at a typical decision carries 13–18% of the mass against
8.3% for uniform, and the marginal distribution over routes across an episode
has normalised entropy 0.94–0.99. The tilt is not random: the corrected policies
load the two constrained fast servers to 0.82 and 0.79 mean utilisation on
electronics, against 0.65 and 0.69 under uniform-random routing and 0.81 and 0.90
under ShortestQueue, and to 0.82 and 0.91 on bakery against 0.74 and 0.90. This
is the response one would expect to the objective identified in §6.4, in which
surplus fast-server busy time is rewarded. It buys constraint reliability, 17
points of joint satisfaction over uniform-random routing on electronics, and no
measurable cost-per-unit. Whether it is state-aware in any further sense the
present data cannot say; §6.6 shows that it is worth more than half of the gap
between stateless routing and ShortestQueue when the utilisation floor is
raised, and little of the gap when the throughput floor is.

The argmax columns show why greedy evaluation of these policies measured little.
On electronics the top action leads the runner-up by two percentage points of
probability on average; two seeds' greedy policies agree on 18% of states in the
corrected cell and 9% in the control, against 8.3% by chance; and logit noise of
standard deviation 0.1 changes the greedy action at a quarter of visited states.
Greedy evaluation therefore selects a route determined by differences that
neither replicate across seeds nor survive small perturbations, and it has no
reason to be feasible; the wide argmax confidence intervals of Table 3 follow.
On bakery, with four routes, the gaps are larger and the agreement higher, and
the argmax feasibility rate of 54% in Table 3 is correspondingly better,
although the per-seed range of the gap (0.02 to 0.28) shows that some seeds
learned a decided preference and others none. The last row of Table 6, the
symmetric cell of §6.7, is the one configuration whose policies are not
near-uniform; it is discussed there.

Two robustness checks. First, single-draw stochastic evaluation introduces
sampling variance of its own. Re-evaluating the electronics corrected seed 42
checkpoint five times with different sampling seeds gives cost-per-unit
79.11–81.63 (SD 0.96) and joint satisfaction 88–96% (SD 0.036); the seed-level
confidence intervals of Table 3 are wider than this, and the bootstrap of
Table 4 resamples episodes as well as seeds, so the comparisons do not rest on a
single draw. Second, the control cell is, if anything, closer to uniform than
the corrected cell, consistent with §6.1: a saturated penalty term four orders
of magnitude larger than the base reward leaves the policy gradient with almost
nothing to say about routing.

### 6.6 Constraint severity

The thresholds T_min and U_min were set from a one-shot LeastUtilised run
(§5.2), and Table 1 shows that at those values stateless routing satisfies both
constraints on most episodes. To see how the comparison depends on this choice
we recomputed joint satisfaction from the archived per-episode records at
stricter thresholds, for the dispatching rules and for the stochastic policies
of the two full-budget cells, without retraining (Table 7). This is a post-hoc
re-scoring of policies trained for the original thresholds, not an experiment
on policies trained for the stricter ones, and it is reported as such.

**Table 7.** Joint constraint satisfaction on the 50 test episodes at
alternative thresholds, without retraining. SQ ShortestQueue, LU LeastUtilised,
RR RoundRobin, Random UniformRandom; "corrected" and "original" are the two
full-budget cells and "symmetric" the cell of §6.7, evaluated stochastically and
pooled over five seeds (the symmetric cell was run on electronics only). The
first row of each testbed is the protocol threshold.

| Testbed | T_min | U_min | SQ | LU | RR | Random | corrected | original | symmetric |
|----------|------|------|------|------|------|--------|-----------|----------|----------|
| electronics | 50 | 0.50 | 100% | 98% | 86% | 78% | 95% | 70% | 75% |
| electronics | 50 | 0.70 | 96% | 96% | 28% | 16% | 71% | 19% | 26% |
| electronics | 54 | 0.50 | 96% | 82% | 66% | 54% | 80% | 47% | 52% |
| electronics | 56 | 0.50 | 92% | 64% | 40% | 34% | 64% | 30% | 36% |
| electronics | 56 | 0.70 | 90% | 64% | 18% | 12% | 58% | 14% | 15% |
| electronics | 58 | 0.50 | 86% | 34% | 30% | 24% | 46% | 19% | 20% |
| electronics | 60 | 0.50 | 74% | 16% | 14% | 14% | 30% | 10% | 11% |
| bakery | 18 | 0.50 | 96% | 90% | 88% | 90% | 94% | 86% | – |
| bakery | 18 | 0.70 | 88% | 88% | 44% | 54% | 84% | 55% | – |
| bakery | 19 | 0.50 | 86% | 72% | 78% | 82% | 85% | 75% | – |
| bakery | 20 | 0.50 | 68% | 46% | 60% | 60% | 64% | 58% | – |
| bakery | 20 | 0.70 | 62% | 46% | 22% | 34% | 57% | 40% | – |
| bakery | 21 | 0.50 | 44% | 32% | 46% | 32% | 40% | 41% | – |

Two patterns emerge. Raising the utilisation floor separates the policies that
route preferentially to the fast servers from those that do not: at U_min = 0.70
on electronics, ShortestQueue and LeastUtilised still satisfy on 96% of
episodes, uniform-random routing on 16%, and the corrected agent on 71%, so the
utilisation tilt of §6.5 recovers roughly two-thirds of the gap between
stateless routing and the state-aware rules on electronics, and nearly all of it
on bakery (84% against 54% and 88%). Raising the throughput floor does the
opposite. At T_min = 58 on electronics, where ShortestQueue still satisfies on
86% of episodes, the corrected agent falls to 46%, closer to RoundRobin (30%)
than to ShortestQueue, and at T_min = 60 the ordering is unchanged with every
learned or stateless policy at or below 30%. The control cell tracks uniform-random
routing throughout. Meeting a demanding throughput floor requires the
queue-sensitive routing that ShortestQueue performs and the learned policies do
not; the constraint the agent was trained against did not require it, and the
selection rule (§3.3) enforced an effective floor less than one throughput
standard deviation above T_min (§3.4). Whether an agent trained against the stricter thresholds
would learn it is a retraining experiment, not a re-scoring, and is outside this
paper.

### 6.7 A functioning dual: the symmetric, reward-scaled cell

The cell defined by Protocol Amendment R2 (§5.3) removes the two features that
§6.4 identified as keeping the corrected cell's multipliers far from
equilibrium: the hinge, replaced by the signed update, and the inherited scale,
replaced by multipliers on the scale of the reward. Five seeds, 1.6M steps,
electronics, stochastic selection, both evaluation modes. Table 8 reports the
per-seed results and Figure 2 the dynamics.

**Table 8.** Symmetric, reward-scaled cell (Amendment R2), electronics, per
seed. λ_T statistics are over the 3,352 training episodes; the penalty ratio is
|Σ p_t| / |Σ r_t| on the test episodes at the training-mean multipliers. CIs
are 95% t-intervals across seeds.

| Seed | Selected checkpoint | Stochastic CPU | Stoch. TP | Stoch. joint sat. | Argmax CPU | Argmax TP | λ_T mean / max | Penalty ratio |
|------|-----------|----------|--------|--------|----------|--------|-------------|--------|
| 42 | 601K | $81.89 | 54.6 | 80% | $107.78 | 43.4 | 0.09 / 0.73 | 0.12 |
| 7 | 300K | $84.87 | 52.2 | 78% | $106.10 | 43.5 | 0.06 / 0.50 | 0.08 |
| 2024 | 200K | $80.92 | 53.9 | 76% | $135.70 | 34.4 | 0.06 / 0.46 | 0.09 |
| 123 | 300K | $77.80 | 57.0 | 72% | $92.76 | 48.1 | 0.13 / 1.48 | 0.14 |
| 999 | 1,103K | $84.05 | 52.6 | 68% | $109.88 | 39.6 | 0.07 / 0.55 | 0.07 |
| mean ± CI | | $81.91 ± 3.47 | 54.1 ± 2.4 | 74.8% | $110.45 ± 19.40 | 41.8 ± 6.3 | | |

![](fig_symmetric.png){width=16cm}\

**Figure 2.** The symmetric, reward-scaled cell during training, five seeds.
(a) λ_T per episode on a linear scale; the dashed line is the shadow-price
estimate of 0.1 from which the initial value was set. (b) Throughput slack
TP − T_min per training episode, 50-episode moving average; the dashed line is
the mean slack of uniform-random routing on the test episodes (+4.4).

The dual now behaves as a dual. λ_T is released to zero within a few episodes
whenever the constraint is slack, rises when it is violated, and spends training
oscillating between 0 and about 0.5, with a training mean of 0.06–0.13 per seed
that brackets the a priori shadow-price estimate; it is positive on 60–72% of
episodes and never approaches its cap (one excursion to 1.5 on seed 123). λ_U
behaves the same way at its own scale (mean 0.002–0.003). The constraint terms
are consequently on the scale of the cost term throughout: 7–14% of the return
at the training-mean multipliers and 0.6–1.5 times the return at the
multipliers' peaks, against 190–10,000 times in the corrected cell.

The policy responds in the direction the total-cost objective pushes it.
Within the first 400K
steps mean throughput per training episode falls from the level of random
routing, 4.4 units above T_min, to 1–3 units above it, and stays there for the
rest of training (Figure 2b); 24–39% of training episodes fall below T_min in
every quarter of every seed. The expectation constraint of Eq. (2) is satisfied,
with a margin of one to three units, and the per-episode chance constraint of
the selection rule is not: of the 95 checkpoints validated, 14 qualify under
the 16-of-20 rule, eight of them at or before 600K steps, and joint satisfaction
on stochastic validation fluctuates between 2 and 18 of 20 episodes across
checkpoints with no trend. Four of the five selected checkpoints are early
(200K–600K; the fifth is at 1.1M), and on test the cell satisfies both
constraints jointly on 74.8% of episodes, 20 points
below the corrected cell (Table 4: −20.4 [−30.0, −11.2], significant after
adjustment) and 25 below ShortestQueue.

On cost-per-unit the cell is worse, not better. Its $81.91 ± 3.47 is
significantly above ShortestQueue (+$8.65) and not distinguishable from
RoundRobin, UniformRandom, LeastUtilised or the original signal; against the
corrected cell it is $3.81 [+0.92, +6.61] more expensive, a difference that does
not survive adjustment (p = 0.012 against a Holm threshold of 0.006). Nor is it
a better total-cost minimiser than the rules it was measured against. Mean
episode cost is $4,414 for this cell against $4,342 for RoundRobin, $4,347 for
UniformRandom, $4,445 for the corrected cell and $4,493 for ShortestQueue: the
range of total cost across every load-spreading policy in this study is 4%,
because the work to be processed is fixed by the arrival process and routing
changes only which server does it and how long jobs wait, while throughput
across the same policies spans 13% (54–61 units). An objective that is flat in
routing and indifferent to throughput above the floor has little to teach a
policy gradient about routing, and what it does teach, that throughput above
the floor is not worth paying for, is the opposite of what cost-per-unit
rewards.

What the cell did learn is a different kind of policy from every other cell in
this paper. Its action distributions are concentrated: normalised entropy
0.63–0.83 against 0.94–0.97, 4.8–7.8 effective routes against 10.4–11.3, mean
top-1 probability 0.28–0.48 against 0.13–0.18, and a top-1 gap eight times
larger (Table 6). Yet its marginal route frequencies over an episode remain
near-uniform (normalised entropy 0.89–0.96), so the concentration is
state-dependent: the policy routes differently in different states, which the
near-uniform cells did not. It is also a randomised policy in the sense of §2.2
rather than a near-uniform one. Its argmax is a coherent conservative router,
throughput 41.8 and $110 per unit, feasible on 10% of episodes, while the
stochastic policy from which it is extracted reaches 54 units at $82. The
randomisation carries a quarter of the throughput. Greedy extraction from this
cell fails for the reason the CMDP literature predicts, not because the logits
are flat: two seeds' greedy policies still agree on only 10% of states, but
logit noise of 0.1 now moves the greedy action at 10% of states rather than 25%.

The outcome is the second of the three the amendment listed. A functioning dual
on the stated CMDP does not produce a policy competitive with ShortestQueue on
either metric; it produces a state-dependent, randomised policy that the
total-cost objective drives to the throughput floor, which satisfies the
expectation constraint, fails the per-episode criterion on a quarter of
episodes, and costs more per unit than stateless routing. Whether it is close
to the total-cost optimum we cannot say; its total cost is not below that of
stateless routing, so the objective was pursued but not demonstrably solved. The failure to learn cost-efficient routing in the corrected
cell was attributable to the multiplier scale in one respect only: with the
constraint terms on the reward scale, PPO does move away from the uniform
policy. Where it moves is governed by the objective it was given, and that
objective is not cost-per-unit.

## 7. Discussion

### 7.1 Two artefacts

The original manuscript's electronics result decomposes into two artefacts,
neither of which is a property of constrained reinforcement learning. The first
is a constraint-implementation error with a specific and generalisable form. An
episode-level constraint was translated into a per-step signal by way of a
cumulative average, and in any system with non-trivial transit time that
translation is biased during the fill phase of every episode. The bias is small,
a per-step throughput gap of order 10⁻³, and would be harmless under a symmetric
dual update, which would release the multiplier once the line filled. Under a
monotone one-sided update it integrates without bound. Each design choice is
defensible on its own; the combination is not, which is what makes this class of
error difficult to detect. Its consequence, as distinct from its presence,
depends on the problem: the multiplier saturates on every seed of every control
run on both testbeds, but the resulting damage to the policy is measurable only
on electronics.

The second artefact is an evaluation-mode mismatch: reporting the argmax of a
policy that is close to uniform, selected and trained as a stochastic policy.
The argmax of such a policy is an essentially arbitrary deterministic router
(§6.5: two seeds agree on 9–18% of states, and logit noise of 0.1 moves the
greedy action at a quarter of them), and the original manuscript's finding of
severe seed-to-seed instability on electronics was largely a measurement of
that arbitrariness. Of the two artefacts this is the one that reverses the
reported verdict, and §6.3 shows that it does so on the original checkpoints
without any retraining. It would have gone unnoticed had the first artefact not
prompted a re-examination of the pipeline.

### 7.2 What the method actually learned, and what it was asked to learn

Correcting both artefacts does not produce a competitive agent. It produces a
near-uniform router with a tilt toward the fast servers. On electronics the
corrected policies are indistinguishable from RoundRobin and UniformRandom on
cost-per-unit and ahead of UniformRandom on constraint reliability by 17 points
of joint satisfaction; the 9-point margin over RoundRobin is not significant. On bakery they are indistinguishable from both on every metric.
ShortestQueue, which reads the queue state and requires no training, is 6.6%
cheaper per unit than the best learned policy on electronics with 100%
constraint satisfaction, and is indistinguishable from it on bakery.

§6.4 supplies the reason for the near-uniform policies, and it is not the one
the original manuscript gave. The agent was not defeated by a penalty that
overwhelmed its objective; its objective was, for all practical purposes, the
constraint terms. With the multipliers two to four orders of magnitude above the
scale of the cost term, never released by the hinged update, and with both
constraints slack under any load-spreading policy, the augmented return
rewarded surplus throughput and surplus fast-server busy time and registered
cost only in the third or fourth significant figure. A near-uniform router
tilted toward the fast servers is a reasonable response to that objective, and
it is what every hinged cell produced. The entropy coefficient of 0.01,
retained from the original manuscript for comparability, works in the same
direction.

§6.7 then closes the question that §6.4 opens. With the multipliers on the scale
of the reward and free to fall, PPO does leave the uniform policy: the
symmetric cell's action distributions are concentrated and state-dependent, and
its argmax is a coherent conservative router rather than an arbitrary one. What
it learns is a policy driven toward the throughput floor by the total-cost
objective, which is the direction the CMDP of Eq. (2) points, and that policy
is worse on cost-per-unit than stateless routing and satisfies the per-episode
criterion on only three quarters of episodes. The mechanism is the one §3.4 anticipated from two data
points and §6.7 measures across every policy in the study: total episode cost
is nearly flat in routing, throughput is not, and cost-per-unit therefore falls
with throughput across the whole operating range. An agent minimising total
cost has no reason to buy throughput above the floor, so a working Lagrangian
moves the policy away from the region where cost-per-unit is lowest, and the chance
constraint that the protocol actually checks is stricter than the expectation
constraint the Lagrangian actually enforces. The phrase "failed to learn"
therefore needs two qualifications. Constrained PPO failed to learn a
cost-efficient, state-aware routing policy competitive with ShortestQueue. In
the hinged cells it was never seriously asked to, because cost had negligible
weight in the objective; in the symmetric cell it was asked to minimise a
quantity that is not the metric, and moved in that direction.

Three consequences follow for anyone posing this problem to a constrained
learner. If cost-per-unit is the criterion by which routing performance is
judged, the learning objective should be aligned with it: a ratio objective
(§3.4), or a throughput term whose weight is set by the reported cost-per-unit
rather than by a constraint; on these testbeds a per-departure bonus of that
kind narrowed but did not close the gap to ShortestQueue for unconstrained PPO
(Alrashdan, 2026). A CMDP that minimises total cost is a legitimate
formulation; it should then be judged on total cost, and here it was not. The
constraint must be the one the protocol
checks: a per-episode chance constraint needs a per-episode formulation, for
instance a penalty on the indicator of violation or a conditional-value-at-risk
constraint, rather than an expectation constraint whose satisfaction with a
one-unit margin leaves a quarter of episodes infeasible. And a functioning dual
is a necessary condition, not a sufficient one; here it changed what was
learned without making it competitive. Until those changes are made and tested,
the practical recommendation of the original manuscript stands, for a
different reason. Tuned dispatching rules remain the baseline to beat on
problems of this size and structure. The constrained agent no longer shows a
systematic constraint failure; it meets the protocol's criterion either the way
a fast-server-biased random router does, or, under a functioning dual, meets
the expectation constraint while failing the per-episode criterion on a quarter
of episodes, and in both cases it is more expensive per unit than a rule
that looks at the queues.

### 7.3 Deterministic deployment

Feasible deterministic policies exist on both testbeds; ShortestQueue is one.
What we did not obtain is a feasible deterministic policy *from the learned
agent* on electronics, for two different reasons in the two kinds of cell.
Greedy extraction from the near-uniform distributions of the hinged cells yields
an arbitrary route. Greedy extraction from the concentrated distributions of
the symmetric cell yields a coherent but conservative router, feasible on 10% of
episodes, because the randomisation carries a quarter of the throughput (§6.7);
this is the randomised-optimum structure of §2.2 observed directly, and no
configuration we tested produced a learned policy whose argmax was reliable. In plants that cannot deploy a
stochastic controller, and there are good reasons for such a requirement,
including auditability and operator trust, the learned policies of this paper
are not deployable, and the correction described here does not change that. The
argmax column is reported throughout so that this remains visible.

### 7.4 Implications for evaluation practice

Two systematic reviews have criticised this field for weak baselines and
simulation-only validation. This paper suggests four further items for the
checklist. Where a constrained method is reported to fail its constraints, the
reader should be able to verify that the constraint optimised is the constraint
stated: the exact augmented per-step reward, the exact dual update, the units
of the slack signal, and its measured value under a known-good reference policy
should be reported. The magnitude of the constraint terms relative to the
objective, at initialisation and at the end of training, should be reported
with them; a single ratio would have revealed the problem of §6.4 in the
original manuscript. Where a stochastic policy is evaluated, the evaluation
mode should be stated and both modes reported, together with a measure of how
far the learned distribution is from uniform. And where a learned router is
compared against dispatching rules, uniform-random and round-robin routing
should be among them, evaluated under the same protocol with uncertainty in the
baselines propagated; without that control this paper would have reported a
recovery where there was only a near-random policy satisfying easy constraints.

## 8. Threats to validity

**Two testbeds.** Bakery and electronics differ simultaneously in action-space
size, stage count, capacity ratios and reward scale. We can attribute the
original failure to the implementation, because that is manipulated directly,
but we cannot attribute the difference in difficulty between testbeds to any
single structural feature. A controlled feature ablation remains outstanding.

**Five seeds.** Stochastic-mode confidence intervals are tight (±0.7–2.0 on
cost-per-unit at full budget), which makes n = 5 more defensible than it was
under argmax evaluation. It remains a small sample, and the bootstrap of Table 4
resamples five seed-level means, so its intervals for between-cell differences
should be read as approximate. The bakery null in §6.2 and the null comparisons
against stateless routing are failures to detect a difference at this n, not
evidence of equality: the electronics interval against RoundRobin, for
instance, spans −$2.74 to +$1.37 per unit. Ten tests per testbed form the
adjusted family; on electronics five survive Holm adjustment and on bakery
none. One V6 seed was not archived, so the V6 rows of Table 5 are n = 4.

**Constraint thresholds.** T_min and U_min were set from a one-shot
LeastUtilised run, and Table 1 shows that at those values stateless routing
satisfies both constraints on 78–90% of episodes. The comparison between the
learned policies and stateless routing on constraint reliability is therefore a
comparison at easy thresholds, and the selection rule of §3.3 adds only a small
effective margin (§3.4). §6.6 shows that the ordering of policies changes when
the thresholds are tightened, in a direction that favours the queue-sensitive
rule, but it re-scores policies trained for the original thresholds; agents
trained against stricter thresholds were not run. The claim that the constraints
"were satisfied" should be read with the thresholds in view.

**Single-draw stochastic evaluation.** Each validation and test episode samples
one action trajectory. The repeated-draw check in §6.5 bounds the resulting
variance at roughly a third of the between-seed interval on one checkpoint; the
bootstrap resamples episodes, which absorbs part of this variance, but the
validation draws that selected the checkpoints were single and unseeded (§5.3).

**Selection bias in the archival comparison.** Q2 in §6.3 re-selects checkpoints
on the same stochastic validation episodes now used for reporting. This is the
protocol the amendment specifies, applied identically to every run, but it is a
different selection from the one the original manuscript used. The Q1 columns,
which evaluate the originally selected checkpoint, are the cleaner like-for-like
comparison, and the principal claim rests on them.

**The amendment was informed by the pilot.** The change to stochastic selection
was motivated by the 400K ablation, which was run under argmax selection. The
amendment was committed before the full-budget runs it governs, but it is not
independent of data from the same study.

**Reduced-budget ablation and omitted cells.** §6.1 is at 400K steps and the
two off-diagonal cells were not run at full budget; see §5.3 for the
justification. The attribution should be treated as established at 400K only.

**One symmetric configuration.** §6.4 shows that the corrected cell optimises
the exact Lagrangian of Eq. (2) at multiplier values that never approach their
equilibrium, so its results characterise a hinged, heavily weighted Lagrangian.
The symmetric cell of §6.7 tests constrained PPO with a functioning dual, but in
one configuration only: multiplier scales fixed by Amendment R2 before its runs
but after the results of §6.1–6.6 had been inspected, so the cell is prospective
rather than pre-registered in the sense of the original protocol; plain dual
ascent without PID damping or averaging; the original entropy coefficient; and
electronics only. Its λ_T oscillates rather than converging (Figure 2a), and
a damped or averaged dual, or a different step size, might select different
checkpoints. We regard the mechanism it exposes, a total-cost objective that is
flat in routing and indifferent to throughput above the floor, as robust to
those choices, because it is a property of the objective measured across every
policy in the study (§6.7), but the cell's numbers are those of one
configuration.

**Segmented training.** Full-budget runs flush the rollout buffer three times per
run. We regard the effect as negligible but have not measured it directly.

**Simulation only.** As in the original manuscript, there is no physical
validation.

## 9. Conclusion

A negative result about constrained reinforcement learning for flow-shop
routing, reported in an earlier manuscript by the author, reflected the
instrumentation used to obtain it. A per-step cumulative-rate slack signal
guaranteed multiplier saturation for any policy, including an oracle, and greedy
evaluation of a near-uniform policy measured an arbitrary deterministic router
rather than the policy that was trained. Both artefacts are reproduced and then
removed on the original checkpoints. With both artefacts corrected, every seed
satisfies the protocol's validation criterion on both testbeds. It does so,
however, in the way that a fast-server-biased random router does: the learned policies
are near-uniform, indistinguishable from round-robin on cost-per-unit under a
paired bootstrap, and 6.6% more expensive than ShortestQueue on the harder
testbed. The audit also shows why. In the original and the corrected one-sided
configurations the constraint terms outweighed the cost term by two to four
orders of magnitude and the dual update could not release them, so the agent was in effect
trained to maximise throughput and fast-server utilisation, and cost-efficient
routing was never effectively part of its objective. The corrected negative
result is narrower than the original and, we think, more useful. The method did
not fail the protocol's constraint criterion; it failed to learn a
cost-efficient state-aware routing policy, under an objective that gave it
little reason to.
Giving it that reason, with a symmetric dual update and multipliers on the scale
of the reward, produced a working Lagrangian and a different policy: a
state-dependent, randomised policy driven to the throughput floor by the
total-cost objective, which satisfies the expectation constraint it was given, fails the per-episode
criterion the protocol checks on a quarter of episodes, and is more expensive
per unit than stateless routing, because total cost is nearly flat in routing
while cost-per-unit falls with throughput. Neither the artefacts nor their
correction make constrained PPO competitive with a two-line rule on this
problem; the reason is now located in the formulation, in the gap between the
objective posed and the metric reported, and between the constraint enforced
and the constraint checked. Separating those claims required reporting the
constraint actually implemented, the scale of the penalty against the objective,
the evaluation mode actually used, and a random-routing control with its own
uncertainty, and we would encourage all four as routine practice.

## Broader impact statement {-}

This is a simulation study of production routing with no human subjects and no
personal data. The bakery testbed is calibrated to the processing-time records of
a published production dataset (Babor & Hitzmann, 2022); that dataset also
contains anonymised employee shift records, which we did not use, and the
simulator models machines only, including the "manual inspection" station of
the electronics testbed, which is represented by a service-time distribution
and a cost rate rather than by any person. The paper's practical conclusion is
negative: it recommends tuned dispatching rules over the learned controllers it
studies, so it does not encourage the deployment of a reinforcement learning
system in a plant. Its intended effect on practice is on how such systems are
evaluated and reported, which we consider a benefit. Two remarks on possible
misuse. Utilisation-floor constraints of the kind studied here, if applied to
human-operated stations rather than machines, would impose a minimum workload on
people and should be set with occupational limits in mind, not only cost; and a
learned router that meets a constraint on average while violating it on a
quarter of shifts (§6.7) would be unsuitable wherever the per-shift target
matters, which the paper's per-episode reporting is designed to make visible.
The study's compute footprint is small (about 20 CPU-hours).

## Data availability

FlexFlowSim-CPPO (Author, 2026b), the pre-registered protocol (`protocol.md`,
commit 391eb0d),
Protocol Amendment R1 (`protocol_amendment_r1.md`), the corrected wrapper
(`lagrangian_slack.py`), Protocol Amendment R2 (`protocol_amendment_r2.md`),
the ablation and archival runners, the policy-entropy and R2 analysis scripts,
and all per-seed results, λ histories and per-episode test records are
available at [URL withheld for review].

## Funding

This research did not receive any specific grant from funding agencies in the
public, commercial, or not-for-profit sectors.

## Declaration of competing interest

The author declares no known competing financial interests or personal
relationships that could have appeared to influence the work reported in this
paper.

## Declaration of generative AI use

During the preparation of this work the corresponding author used a large
language model assistant (Claude, Anthropic) to assist with simulation code,
analysis scripting, drafting and editing. The author verified all results and
takes full responsibility for the content of the article.

## CRediT authorship contribution statement

**Khaled R. Alrashdan:** Conceptualization, Methodology, Software, Investigation,
Formal analysis, Data curation, Visualization, Writing – original draft, Writing
– review & editing.


## References

Achiam, J., Held, D., Tamar, A., & Abbeel, P. (2017). Constrained Policy
Optimization. *Proceedings of the 34th International Conference on Machine
Learning*, PMLR 70, 22–31.

Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A., & Bellemare, M. G.
(2021). Deep Reinforcement Learning at the Edge of the Statistical Precipice.
*Advances in Neural Information Processing Systems 34*, 29304–29320.

Ali, A. M., & Tirel, L. (2023). Action Masked Deep Reinforcement Learning for
Controlling Industrial Assembly Lines. *2023 IEEE World AI IoT Congress (AIIoT)*.
https://doi.org/10.1109/AIIoT58121.2023.10174426

Alrashdan, K. R. (2026). Routing Under Machine Breakdowns: A Benchmark of
Dispatching Rules, Bandits, and Reinforcement Learning for Multi-Server Flow
Shops. *Journal of King Saud University – Engineering Sciences*, 38, 55.
https://doi.org/10.1007/s44444-026-00128-9

Altman, E. (1999). *Constrained Markov Decision Processes*. Chapman & Hall/CRC.

Author. (2026a). Failure Modes of Constrained PPO in Multi-Server Flow-Shop
Routing: A Benchmark Against Dispatching Rules. Manuscript. [Details withheld for
review.]

Author. (2026b). FlexFlowSim-CPPO: Companion code. [Computer software]. GitHub.
[URL withheld for review.]

Babor, M., & Hitzmann, B. (2022). *Small and medium-sized bakery production data
for scheduling* (Version 2) [Dataset]. Mendeley Data.
https://doi.org/10.17632/dhgbssb8ns.2

Doherty, M., Matzner, R., Sadeghi, R., Bayvel, P., & Beghelli, A. (2025).
Reinforcement Learning for Dynamic Resource Allocation in Optical Networks: Hype
or Hope? *Journal of Optical Communications and Networking*, 17(9), D1–D17.
https://doi.org/10.1364/JOCN.559990

Ferreira, C., Figueira, G., & Amorim, P. (2022). Effective and Interpretable
Dispatching Rules for Dynamic Job Shops via Guided Empirical Learning. *Omega*,
111, 102643. https://doi.org/10.1016/j.omega.2022.102643

Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., & Meger, D.
(2018). Deep Reinforcement Learning That Matters. *Proceedings of the
Thirty-Second AAAI Conference on Artificial Intelligence*, 3207–3214.
https://doi.org/10.1609/aaai.v32i1.11694

Huang, Z., Mei, Y., Zhang, F., & Zhang, M. (2025). Toward Evolving Dispatching
Rules With Flow Control Operations by Grammar-Guided Linear Genetic Programming.
*IEEE Transactions on Evolutionary Computation*, 29(1), 217–231.
https://doi.org/10.1109/TEVC.2024.3353207

Li, C., Zhao, X., Lin, L., Zhang, W., Gen, M., & Zhang, Q. (2025). An
Evolutionary Knowledge Training-Based Proximal Policy Optimization Algorithm for
Job Shop Scheduling in Flexible Intelligent Manufacturing. *Computers &
Industrial Engineering*, 210, 111533. https://doi.org/10.1016/j.cie.2025.111533

Liu, Y., Fan, J., & Shen, W. (2025). A Deep Reinforcement Learning Approach With
Graph Attention Network and Multi-Signal Differential Reward for Dynamic Hybrid
Flow Shop Scheduling Problem. *Journal of Manufacturing Systems*, 80, 643–661.
https://doi.org/10.1016/j.jmsy.2025.03.028

Marques, N., Figueira, G., & Guimarães, L. (2025). Dynamic Dispatching Rule
Selection for the Job Shop Scheduling Problem. *Computers & Industrial
Engineering*, 210, 111471. https://doi.org/10.1016/j.cie.2025.111471

Mayerhoff, J., & Schmidt, M. (2026). Reinforcement Learning for Autonomous
Production Planning and Control: A Systematic Literature Review. *Journal of
Manufacturing Systems*, 86, 546–568. https://doi.org/10.1016/j.jmsy.2026.03.023

Paternain, S., Chamon, L. F. O., Calvo-Fullana, M., & Ribeiro, A. (2019).
Constrained Reinforcement Learning Has Zero Duality Gap. *Advances in Neural
Information Processing Systems 32*.

Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., & Dormann, N.
(2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations.
*Journal of Machine Learning Research*, 22(268), 1–8.

Rinciog, A., & Meyer, A. (2021). Fabricatio-RL: A Reinforcement Learning
Simulation Framework for Production Scheduling. *Proceedings of the 2021 Winter
Simulation Conference*. https://doi.org/10.1109/WSC52266.2021.9715366

Schneider, J., Pfannschmidt, C., Nyhuis, P., & Schmidt, M. (2026). The Role of
Reinforcement Learning in Production Control: A Systematic Literature Review.
*IEEE Access*, 14, 34375–34389. https://doi.org/10.1109/ACCESS.2026.3668903

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
Proximal Policy Optimization Algorithms. *arXiv preprint arXiv:1707.06347*.
https://doi.org/10.48550/arXiv.1707.06347

Shen, Y., Zhang, X., & Jin, T. (2026). Transformer-Based Multi-Agent
Reinforcement Learning for Flexible Job Shop Scheduling With AGVs. *Applied Soft
Computing*, 193, 114899. https://doi.org/10.1016/j.asoc.2026.114899

Stooke, A., Achiam, J., & Abbeel, P. (2020). Responsive Safety in Reinforcement
Learning by PID Lagrangian Methods. *Proceedings of the 37th International
Conference on Machine Learning*, PMLR 119, 9133–9143.

Tang, C.-Y., Liu, C.-H., Chen, W.-K., & You, S.-D. (2020). Implementing Action
Mask in Proximal Policy Optimization (PPO) Algorithm. *ICT Express*, 6(3),
200–203. https://doi.org/10.1016/j.icte.2020.05.003

Tessler, C., Mankowitz, D. J., & Mannor, S. (2019). Reward Constrained Policy
Optimization. *7th International Conference on Learning Representations*.

Wang, R., Jing, Y., Gu, C., He, S., & Chen, J. (2025). End-to-End Multitarget
Flexible Job Shop Scheduling With Deep Reinforcement Learning. *IEEE Internet of
Things Journal*, 12(4), 4420–4434. https://doi.org/10.1109/JIOT.2024.3485748

Zhang, N., Liu, B., & Zhang, J. (2025). Dual Resource Scheduling Method of
Production Equipment and Rail-Guided Vehicles Based on Proximal Policy
Optimization Algorithm. *Technologies*, 13(12), 573.
https://doi.org/10.3390/technologies13120573


## Appendix A. Reproducibility

### A.1 Code and configurations

All experiments use FlexFlowSim-CPPO (Author, 2026b). The two testbeds use
configs/bakery_bk50.json and configs/electronics_3stage.json. The original
Lagrangian wrapper is pilot_constrained_v4_auto.py; the corrected wrapper of
§4.3, which also reproduces the original signal as a control mode, is
lagrangian_slack.py. The ablation and full-budget runs of §6.1–6.2 are driven
by run_ablation_2x2.py (training in resumable segments, dual-mode validation
with per-checkpoint caching, selection, and test); the archival re-evaluation of
§6.3 by archival_reeval.py; the action-distribution statistics and
repeated-draw check of §6.5 by analysis_stateless_and_entropy.py; and the seeded
per-episode re-evaluation, argmax-stability metrics, paired hierarchical
bootstrap and severity re-scoring of §5.2, §6.2, §6.5 and §6.6 by
analysis_r2.py, analysis_r2_stats.py and analysis_r2_tables.py; Figure 2 by
make_fig_r2.py. The pre-committed protocol is protocol.md (commit 391eb0d); its
amendments are protocol_amendment_r1.md and protocol_amendment_r2.md;
deviations are in protocol_deviations.md. All per-seed summaries, validation
caches, λ histories and per-episode test records are under results_r1/ and
results_r2/, with the seeded records under results_r1/r2/.

### A.2 Hyperparameters

PPO inner loop (Stable-Baselines3, MlpPolicy with two hidden layers of 64 tanh
units; defaults unless noted): learning rate 3 × 10⁻⁴; n_steps 2048; batch size
64; n_epochs 10; γ = 0.99; GAE λ = 0.95; clip range 0.2; entropy coefficient
0.01; value-function coefficient 0.5; max gradient norm 0.5. Lagrangian outer
loop, identical in both slack modes: U_min = 0.50; T_min = 18 (bakery) or 50
(electronics); λ_U initial 5, step η_U = 20, cap 500; λ_T initial 200, step
η_T = 4000, cap 20,000. The cumulative-rate mode applies a 100-step warm-up; the
episode mode applies none (§4.3). Multipliers are updated once per episode in
both modes. The symmetric cell of §6.7 (Amendment R2) uses the signed update
with λ_T initial 0.1, η_T = 2.0, cap 10 and λ_U initial 0.002, η_U = 0.01,
cap 0.2; all PPO hyperparameters unchanged.

### A.3 Budgets and seeds

Training seeds [42, 7, 2024, 123, 999] throughout. Mechanism ablation (§6.1):
400K timesteps per run, four cells, electronics only. Full-budget comparison
(§6.2): 1.6M timesteps on electronics and 1.5M on bakery, two cells, executed in
400K-step segments (§5.3); symmetric cell (§6.7): 1.6M timesteps, electronics,
same segments and seeds. Checkpoints every 100K timesteps in all runs; because
PPO completes whole 2,048-step rollouts, checkpoint labels are nominal and
actual step counts exceed them by up to 4,224 steps (0.3%), identically in
every cell.
Validation seeds [10000, 10020); test seeds [11000, 11050); both disjoint from
training. All dispatching rules (Table 1) are evaluated on the 50 test seeds,
with UniformRandom's action sampling seeded by the episode seed. The
action-distribution statistics of Table 6 use the first ten test seeds; its
argmax-stability columns use the states visited under uniform-random routing on
the first five.

### A.4 Evaluation protocol

Every checkpoint is evaluated on the 20 validation episodes in both modes:
argmax (`deterministic=True`) and stochastic (one action sample per step from
the policy distribution, one trajectory per episode). Selection follows §5.3 on
the stochastic validation results; argmax selection is also computed and stored.
The selected checkpoint is evaluated on the 50 test episodes in both modes; the
stochastic test figures reported are the seeded re-evaluation described in §5.3
(deviation R2-1), with `torch.manual_seed(episode seed)` before each episode.
Cost-per-unit is total episode cost divided by episode departures, averaged over
episodes. Joint satisfaction is the fraction of episodes in which every
constrained server's realised utilisation is at least U_min and departures are
at least T_min, computed per episode.

### A.5 Statistical tests

All comparisons in Table 4 use a paired hierarchical bootstrap on the
per-episode test records. Let X be the seeds × episodes matrix of a learned
cell's per-episode metric (5 × 50) and Y the comparator's (5 × 50 for another
learned cell, 1 × 50 for a dispatching rule). Each of 10,000 replicates draws
seed indices with replacement, independently for X and Y, and a single vector of
50 episode indices with replacement applied to both, so the two policies are
compared on the same resampled episodes; the statistic is the difference of
grand means. Intervals are 2.5th–97.5th percentiles; p is twice the smaller
tail proportion at zero, floored at 1/10,000. The generator seed is 20260905.
The pre-specified family on each testbed is the corrected cell against the four
rules of Table 1 and the original signal, on cost-per-unit and joint
satisfaction (ten tests), adjusted by Holm's step-down procedure at α = 0.05.
Confidence intervals in the tables use the t-distribution with n − 1 degrees of
freedom across seeds and the normal approximation across episodes. Welch and
one-sample t-tests on seed-level means, used in an earlier draft, are retained
in the analysis scripts for comparison; they agree in direction with the
bootstrap everywhere and overstate significance for the comparisons against
dispatching rules.

### A.6 Hardware and runtime

R1 training and evaluation ran on a two-vCPU Linux container at approximately
800–1,000 environment steps per second. A 400K-step segment takes 7–8 minutes;
a full 1.6M-step run about 30 minutes plus 10–15 minutes of dual-mode
validation over 16 checkpoints. The full R1 matrix (20 pilot runs, 20 full-budget
runs, 18 archival re-evaluations of 279 checkpoints, baselines and analyses)
consumed approximately 16 compute-hours; the seeded per-episode re-evaluations
and analyses of R2 (about 5,400 further evaluation episodes) roughly one more,
and the five symmetric-cell runs of §6.7 about three and a half. The original manuscript's runs were
executed on a consumer-grade Windows machine at roughly 20–22 minutes per
1.5–1.6M-step run.

## Appendix B. Nomenclature

| Symbol / abbreviation | Meaning |
|---|---|
| PPO | Proximal Policy Optimization (Schulman et al., 2017) |
| CMDP | Constrained Markov Decision Process |
| CPO, RCPO | Constrained Policy Optimization; Reward Constrained Policy Optimization |
| GAE | Generalised Advantage Estimation |
| PID | Proportional–integral–derivative (controller form of the dual update in V6) |
| V4, V6 | Original manuscript's one-sided Lagrangian variant; its PID-Lagrangian comparator |
| CPU | Cost-per-unit ($ per completed job; primary metric) |
| TP(τ) | Throughput: jobs completed in episode τ |
| Joint sat. | Fraction of episodes satisfying both constraints |
| CI | 95% confidence interval |
| LU, SQ | LeastUtilised; ShortestQueue (dispatching rules) |
| F_fast | Set of fast/expensive servers subject to the utilisation constraint |
| T_min, U_min | Throughput floor (episode departures); per-server utilisation floor |
| λ_T, λ_U | Lagrange multipliers for the throughput and utilisation constraints |
| η_T, η_U | Multiplier step sizes |
| g_T,t, g_U,t | Per-step slack signals of the original implementation, Eq. (4) |
| p_t | Corrected per-step penalty, Eq. (6) |
| ū_{i,t}, ū_i(τ) | Cumulative utilisation of server i up to step t; realised utilisation over episode τ |
| b_{i,t} | In-service indicator of server i at step t |
| d_t, δ_t | Cumulative departures up to step t; departures in step t |
| c_t, C(τ) | Dollar cost accrued in step t; episode total |
| W_t | Work in process at step t (queued plus in service) |
| C_norm, N_norm, W_norm | Reward normalisation constants of Eq. (1) |
| H, Δt | Episode horizon, 480 one-minute steps; step length, 1 min |
| κ, w_c, w_t, w_w | Reward scale and shaping weights of Eq. (1) |
| π | Policy |
