---
title: "Objective, constraint and evaluation alignment in constrained reinforcement learning for multi-server flow-shop routing"
---


## Abstract

Constrained reinforcement learning is increasingly applied to production
routing, but a reported result rests on a chain of choices seldom reported
together: the constraint stated, the surrogate the training loop penalises, the
scale of the multipliers, the evaluation mode, and the metric reported. This
simulation study measures what happens when links in that chain disagree, using
Lagrangian PPO on two flow-shop testbeds with 4 and 12 routes. Implementing an
episode-level throughput constraint as a per-step penalty on the cumulative rate
drives the multiplier to its cap for an oracle dispatching rule as well as for
every trained policy, because that rate sits below its floor while the line
fills: 10 of 10 seeds saturate with this signal, 0 of 10 with an episode-level
slack. Evaluating near-uniform stochastic policies greedily measures a router
selected by unstable logit differences; sampling instead raises joint constraint
satisfaction on the archived checkpoints from 6% to 71% without retraining, and
correcting the slack signal raises it from 70% to 95%. The agent is nonetheless
not distinguishable from round-robin routing on cost per unit under a paired
bootstrap, and is 6.6% more expensive than ShortestQueue. Giving the dual the
scale of the reward makes it behave as a dual and moves the policy to the
throughput floor, where it is no better on the reported metric, because total
cost is nearly flat in routing while cost per unit falls with throughput.

**Keywords:** discrete-event simulation, constrained reinforcement learning,
flow-shop routing, dispatching rules, benchmarking, production control

## 1. Introduction

Reinforcement learning (RL) is now routinely proposed for routing and sequencing
decisions in production systems. Two recent systematic reviews of the field
[1,2] report the same weaknesses across
roughly two hundred studies: validation in simulation only, dispatching-rule
baselines that receive far less tuning attention than the learned agent, and an
absence of standardised benchmarks. Work that responds to that critique tends to
concentrate on the baselines and the protocol. This paper argues that a second
class of problem sits upstream of both, in the instrumentation of the
constrained formulation itself, and that it is capable of producing a reported
result that is an artefact of the pipeline rather than a property of the method.

A constrained learning pipeline contains a chain of five objects that are easy
to conflate. There is the constraint written in the formulation, normally an
expectation over an episode. There is the surrogate the training loop actually
penalises, which must be expressed per step for a policy-gradient method to use
it. There is the scale of the multipliers that weight that surrogate against the
objective. There is the mode in which the trained stochastic policy is
evaluated, by sampling or by its mode. And there is the metric finally reported,
which in this literature is almost always a ratio such as cost per unit. Each
link is a defensible engineering choice in isolation. When two of them disagree,
aggregate performance figures do not reveal which link is at fault, and the
natural reading of a poor result is that the method does not work.

This study measures the consequences of three such breaks, using Lagrangian
Proximal Policy Optimization (PPO) [3] on two multi-server
flow-shop testbeds, and it does so on a system where the answer can be checked
against tuned dispatching rules. The vehicle is an earlier study by the present
author, run under a protocol committed in advance, which reported that no seed
satisfied throughput and utilisation constraints on a 12-route electronics
testbed and that the throughput multiplier saturated against its cap in every
run. That study is referred to below as the initial study; its protocol,
archived checkpoints and result files are in the companion repository
[4] and are the objects re-examined here.

The first break is between the stated constraint and the training surrogate. The
constraint is on an episode total, but the implemented signal penalised the
shortfall of a cumulative average rate at every step. In a flow shop whose flow
time is a sizeable fraction of the shift, that quantity is below its floor for a
long prefix of every episode whatever the policy does, because the line is
filling. Section 4.2 shows that under a monotone one-sided dual update this
drives the multiplier upward for any policy, derives from measurable quantities
the slack above which it reaches its cap within the training budget, and confirms
that an oracle dispatching rule exceeds that threshold by a factor of two while
satisfying the true constraint with a 14–23% margin.

The second break is between the policy trained and the policy evaluated. The
learned action distributions are close to uniform, and the greedy action of a
near-uniform distribution is a deterministic router selected by differences that
neither replicate across seeds nor survive small perturbations of the logits
(Section 6.5). Evaluated by sampling, the archived checkpoints satisfy their
constraints and the original pass/fail verdict reverses without any retraining
(Section 6.3).

The third break is between the objective optimised and the metric reported, and
it survives the correction of the other two. With both artefacts removed, the
agent satisfies the constraint criterion on every seed, but it is
indistinguishable from round-robin routing on cost per unit under a paired
hierarchical bootstrap and 6.6% more expensive than a two-line queue-length
rule. Section 6.4 explains why: in every one-sided configuration studied the
constraint terms of the augmented objective exceed the cost term by two to four
orders of magnitude and cannot be released, so cost was never effectively part
of what was optimised. A further cell, specified in a protocol amendment before
its runs, gives the dual multipliers the scale of the reward and a symmetric
update. The dual then works as a dual, the learned policy becomes concentrated
and state-dependent, and it moves toward the throughput floor exactly as a
total-cost objective directs, leaving it no better on the reported metric than
stateless routing and significantly worse than a queue-length rule (Section 6.7).
The mechanism is measurable across the seven load-spreading policies of this
study: total episode cost spans 4.6% across them while throughput spans 15.5%,
so cost per unit falls with throughput and an agent minimising total cost has no
reason to buy throughput above its floor.

The scope of these results should be stated plainly. The first two breaks are
properties of the instrumentation, not of constrained RL, and the first in
particular is easy to introduce and hard to detect from aggregate metrics.
Correcting them does not make the method competitive on this problem, and it
does not make deterministic deployment feasible. What the study offers is a
measured account of where a constrained pipeline can come apart, the
diagnostics that expose each break, and a demonstration that the residual
failure lies in the formulation rather than in the optimiser.

## 2. Related work

### 2.1 Reinforcement learning and dispatching rules in production control

The two systematic reviews cited above map a field that has shifted from
value-based methods to policy gradients, with PPO dominant, and that repeatedly
reports strong performance against weak baselines and ambiguous performance
against well-tuned heuristics. Recent flow-shop work has converged on
graph-based architectures trained with PPO [5,6,7,8]. An alternative to soft penalties is action
masking, which zeroes infeasible actions in the policy distribution
[9,10,11]; masking presupposes that some
actions are infeasible, which does not apply in the fully feasible routing
problem studied here. A parallel line bypasses RL and evolves interpretable
dispatching rules directly [12,13,14].

On the benchmarking problem itself, Doherty and colleagues [15]
reproduce five landmark RL studies in optical resource allocation, apply properly
tuned heuristic baselines, and find that simple heuristics consistently match or
outperform the published results. On the two testbeds used here, Alrashdan
[16] benchmarked dispatching rules, Thompson-sampling
bandits and unconstrained PPO under machine breakdowns of varying severity, and
found ShortestQueue ahead of PPO on cost per unit by 38–153% depending on the
disruption level, with a corrected training protocol narrowing the gap without
closing it; that study addressed neither constraints nor evaluation mode, which
are the subject here. Rinciog and Meyer [17] introduced
FabricatioRL, the closest comparator to the simulator used in this work.

### 2.2 Constrained MDPs, Lagrangian methods and evaluation practice

The constrained Markov decision process (CMDP) formulation is due to Altman
[18], who shows that for a finite CMDP with K constraints under
discounted cost there is an optimal stationary policy requiring at most K
randomisations, and notes more generally that optimal constrained policies
require randomisation or time-sharing between deterministic policies. Those
results are stated for a setting different from ours, which is finite-horizon
and uses function approximation; they are used here to motivate reporting
stochastic as well as greedy evaluation, not to explain the learned policies.
The modern policy-gradient pipeline begins with Constrained Policy Optimization
[19]; Reward Constrained Policy Optimization [20]
introduces the multi-timescale Lagrangian approach used here, and Paternain and
colleagues [21] prove a zero duality gap that underwrites
primal-dual methods despite the non-convexity of policy optimisation. Stooke and
colleagues [22] are the closest methodological precedent: the
standard Lagrangian update behaves as integral control on the violation signal,
producing oscillation and overshoot, and a PID controller on the multiplier damps
it.

That literature analyses dual dynamics given a constraint signal. It has
comparatively little to say about whether the signal fed to the dual update is
the constraint the paper claims to impose, which is the gap this study occupies.
On the evaluation side, Henderson and colleagues [23] showed
that reported deep-RL results are sensitive to protocol choices that are rarely
stated, and Agarwal and colleagues [24] showed that point
estimates over a handful of runs routinely misstate both the level and the
uncertainty of performance. Whether a stochastic policy is evaluated by sampling
or by its mode is one such choice, usually left to a library default; for a
near-uniform policy the two modes measure different objects, and this paper is an
extended example of the consequence.

## 3. Problem formulation and testbeds

### 3.1 Flow-shop model

The simulator represents a flow shop of N stages, stage s holding m_s parallel
servers that differ in service-time distribution and cost rate. Jobs arrive at
the first stage as a Poisson process and pass through the stages in order;
service times are exponential with stage- and server-specific rates. Cost accrues
on three counts: a processing rate while a server is busy, an idle rate while it
is not, and a waiting rate per queued job. An episode represents one shift of
H = 480 one-minute steps.

A routing decision assigns an arriving job its complete downstream route rather
than one stage at a time, so an action is a tuple a = (a₁, …, a_N) drawn from a
space of size ∏_s m_s. This centralised formulation simplifies credit assignment
relative to a per-stage MDP at the cost of an action space that grows
multiplicatively with stage count. The observation concatenates stage queue
lengths, per-server in-service indicators and accumulated cost signals; it does
not contain the dual multipliers, since the Lagrangian wrapper modifies the
reward and leaves the observation untouched.

Two instances are used. The bakery testbed has two stages of two servers, giving
4 routes, with each stage pairing a fast expensive machine against a slow cheap
one and service times calibrated to the BK50 subset of a published bakery
production dataset [25]. The electronics testbed has three stages
of 2, 3 and 2 servers, giving 12 routes, with asymmetric capacities and one
structurally over-provisioned station. Full machine-level specifications are in
the simulator configuration files of the companion repository
[4] and in the study that introduced these instances
[16]; the present work uses them without that study's
breakdown extensions. The constrained fast servers are the fast machine of each
of the first two stages, written F_fast, with throughput floor T_min = 18
(bakery) or 50 (electronics) and utilisation floor U_min = 0.50. The
over-provisioned electronics station is excluded from F_fast because the
LeastUtilised rule itself reaches only 42% utilisation there, so a 0.50 floor
would be infeasible for any policy.

### 3.2 Reward and the conservative routing attractor

Let c_t be the cost accrued in step t, summed over servers, and C(τ) = Σ_t c_t
the episode total. The environment reward is the scalarised per-step signal of
Eq. (1), transcribed from the implementation,

$$r_t = \kappa\left(-\,w_c\,\frac{c_t}{C_{\mathrm{norm}}} + w_t\,\frac{\dot n_t\,\Delta t}{N_{\mathrm{norm}}} - w_w\,\frac{W_t\,\Delta t}{W_{\mathrm{norm}}}\right)\qquad(1)$$

with weights (w_c, w_t, w_w) summing to one, κ = 10, Δt = 1 min, W_t the work in
process and (C_norm, N_norm, W_norm) fixed normalisation constants, (2740, 20,
3220) on bakery and (4230, 55, 5100) on electronics, chosen as typical episode
totals so each term is of order one per episode. Here ṅ_t is the cumulative
average throughput rate d_t / t, with d_t the departures up to step t, not an
instantaneous rate; the distinction matters in Section 4.2. Under the cost-only
weights (1, 0, 0) the episode return is −κ C(τ)/C_norm, proportional to total
cost, so the objective of Eq. (2) is optimised without approximation.

For w_c near unity, PPO solves the problem it was given and idles expensive
servers. Cost falls, throughput falls further, and cost per unit rises above that
of any dispatching rule. Sweeping w_c down to 0.5 does not escape this
attractor. Constraining the problem is the natural response, and is what the
remainder of the paper examines.

### 3.3 Constrained formulation and three constraint objects

Write TP(τ) for departures in episode τ and u_i(τ) for the realised utilisation
of server i. The agent is posed the problem of Eq. (2),

$$\max_{\pi}\; \mathbb{E}_\pi\!\left[-C(\tau)\right] \quad \text{s.t.}\quad \mathbb{E}_\pi[\mathrm{TP}(\tau)] \ge T_{\min},\quad \mathbb{E}_\pi[u_i(\tau)] \ge U_{\min}\;\; \forall\, i \in F_{\mathrm{fast}}\qquad(2)$$

Three distinct constraint objects appear in this study, and much of what follows
turns on the differences between them. The first is the expectation constraint of
Eq. (2), the problem stated. The second is the training surrogate: the primal
update penalises the exact per-episode Lagrangian of Eq. (2), but the dual update
reads a hinged per-episode shortfall, max(0, T_min − TP(τ)), rather than the
signed slack, so the multipliers respond to the frequency and depth of violations
rather than to the expected constraint value. The third is the selection and
reporting criterion: a checkpoint qualifies if each constraint holds on at least
16 of 20 validation episodes, and results report the fraction of test episodes on
which both hold. Selection therefore imposes two per-episode chance constraints,
P(TP(τ) ≥ T_min) ≥ 0.8 and P(min_i u_i(τ) ≥ U_min) ≥ 0.8, while reporting measures
the joint event. For per-episode distributions as nearly symmetric as those
observed here both are stricter than Eq. (2), and the joint event is stricter
again; Section 6.7 exhibits a configuration that satisfies Eq. (2) and both
chance constraints while failing the joint criterion. The initial study treated the three as
interchangeable; Section 4.2 shows that what it implemented in training was in
fact a fourth object coinciding with none of them.

### 3.4 The objective is not the reported metric

The headline metric in this literature is cost per unit, computed here as the
per-episode ratio C(τ)/TP(τ) averaged over episodes, whereas the CMDP objective
of Eq. (2) minimises expected total cost subject to a throughput floor. The two
differ, and on these testbeds they diverge in a direction that matters.
Interpolating between the mean cost and mean throughput of the CostMinimising and
ShortestQueue baselines on bakery gives a marginal cost of about $40 per
additional unit against an average of about $139, which makes cost per unit
decreasing in throughput across the operating range. A total-cost minimiser then
has every incentive to sit at the throughput floor, where the interpolated cost
per unit of approximately $151 is worse than ShortestQueue's $138.66. This is a
two-point argument, presented here as motivation; Section 6.7 measures the same
divergence directly across the seven load-spreading policies of the study, by
which is meant the four dispatching rules of Table 1 that distribute work across
routes and the three learned cells, excluding CostMinimising and FastServerFirst,
which concentrate it.

Its consequence is that any apparent success of a constrained agent on cost per
unit must be delivered by something other than the objective. In the initial
study the candidate is the checkpoint-selection rule: requiring at least 16 of 20
validation episodes to satisfy TP ≥ 18, with a per-episode standard deviation of
about 2.0 units, implies a mean throughput near 19.7, an effective floor above
the nominal 18 though less than one standard deviation above it.

## 4. Constrained method and its implementation

### 4.1 Lagrangian PPO

The constrained problem is solved with PPO as the inner loop and a Gymnasium
wrapper that augments the per-step reward with penalties, as in Eq. (3),

$$\tilde r_t = r_t - \lambda_U\, g_{U,t} - \lambda_T\, g_{T,t}\qquad(3)$$

where r_t is the environment reward of Eq. (1) rather than the bare cost term.
The initial study posed the problem as single-objective cost minimisation, but
every constrained variant it ran augmented the shaped reward with weights
(0.8, 0.1, 0.1) active. An ablation isolating the shaping terms (Appendix B,
Table B1) finds no resolvable effect under the bootstrap of Section 6.1, but the discrepancy
between the problem posed and the reward implemented is recorded here as the
first instance of the chain breaking.

### 4.2 The cumulative-rate slack and why it saturates any multiplier

The dual variables are updated from a slack signal g. The initial implementation
used, at every step after a 100-step warm-up, the signal of Eq. (4),

$$g_{T,t} = \max\!\left(0,\; \frac{T_{\min}}{H} - \frac{d_t}{t}\right)\qquad(4)$$

with d_t the cumulative departures up to step t. This is not the constraint of
Eq. (2): it compares a cumulative average rate against the floor at every
instant. In a flow shop whose flow time is a sizeable fraction of the horizon the
cumulative rate lies below the floor for a long prefix of every episode whatever
the policy does, because the line is filling. Measured under ShortestQueue over
the 50 test episodes, the cumulative rate first reaches its floor at a median
step of 184 on bakery and 194 on electronics, ranging from 100 to 434 and from
101 to 399 across episodes, and it remains below the floor on 45% of post-warm-up
steps on bakery and 31% on electronics. A policy achieving exactly TP = 50 on
electronics, which meets the stated constraint, does not cross until t ≈ 480 and
so never satisfies the training signal within the horizon. The utilisation slack
was defined analogously, with the same fill-phase bias in milder form.

The one-sided dual update of Eq. (5), applied once per episode with the mean
taken over the post-warm-up steps,

$$\lambda \leftarrow \operatorname{clip}\!\left(\lambda + \eta\,\operatorname{mean}_t\, g_t,\; 0,\; \lambda_{\max}\right)\qquad(5)$$

is a violation-driven ratchet with monotone growth rather than a dual ascent, and
it turns that bias into a structural outcome. Averaged over the 50 test episodes,
the mean per-step g_T under ShortestQueue is 0.0031 on bakery and 0.0054 on
electronics; at η_T = 4000 these are per-episode increments of 12.5 and 21.6,
against the 6.30 and 5.91 needed to carry λ_T from its initial value of 200 to
its cap of 20,000 within the 3,142 and 3,352 training episodes of the two
budgets. Any policy whose mean slack exceeds 0.0016 per step therefore saturates
the multiplier within the budget, and the oracle's exceeds it by a factor of two,
reaching the cap at episode 1,581 of 3,142 on bakery and 915 of 3,352 on
electronics. The policies actually trained saturate sooner, at increments of
15–16 and 39–49 per episode and at episodes 1,223–1,359 (39–43% of training) and
402–504 (12–15%), which is what Section 6.1 observes. Saturation therefore
reflects the signal, not the policy, and reporting it as evidence about the
method is a category error.

### 4.3 Episode-level slack and the corrected dual updates

The per-step signal is replaced by the signed pro-rata decomposition of the
episode Lagrangian, Eq. (6),

$$p_t = \lambda_T\left(\frac{T_{\min}}{H} - \delta_t\right) + \lambda_U \sum_{i\in F_{\mathrm{fast}}}\left(U_{\min} - b_{i,t}\right)\qquad(6)$$

where δ_t is the number of departures in step t and b_{i,t} ∈ {0, 1} indicates
whether server i is in service at step t. The augmented reward is r̃_t = r_t − p_t
at every step, with no warm-up. Summed over the episode, Eq. (6) telescopes to
Eq. (7),

$$\sum_{t=0}^{H-1} p_t = \lambda_T\left(T_{\min} - \mathrm{TP}(\tau)\right) + \lambda_U\, H \sum_{i\in F_{\mathrm{fast}}}\left(U_{\min} - \bar u_i(\tau)\right)\qquad(7)$$

the Lagrangian penalty corresponding to the expectation constraint of Eq. (2),
expressed in the same episode-level quantities that validation and test evaluate.
There is no fill-phase bias: an action's contribution depends only on the
departures and busy time it causes. Multipliers are updated once per episode from
the hinged episode-level slack of Eqs. (8) and (9),

$$\lambda_T \leftarrow \operatorname{clip}\!\left(\lambda_T + \eta_T\,\frac{\max\!\left(0,\; T_{\min} - \mathrm{TP}(\tau)\right)}{H},\; 0,\; \lambda_{T,\max}\right)\qquad(8)$$

$$\lambda_U \leftarrow \operatorname{clip}\!\left(\lambda_U + \eta_U \sum_{i\in F_{\mathrm{fast}}} \max\!\left(0,\; U_{\min} - \bar u_i(\tau)\right),\; 0,\; \lambda_{U,\max}\right)\qquad(9)$$

The division by H keeps the throughput slack in the per-step rate units of
Eq. (4), so η_T, λ_T's initial value and its cap carry over unchanged and the
comparison with the original signal is like for like. Under Eqs. (8) and (9) the
ratchet advances only on episodes that violate, so saturation is no longer
guaranteed for an oracle, which is the property tested in Section 6.1.

Two consequences of retaining the hinge should be stated at the outset. The
update remains monotone, so a multiplier never decreases and cannot return toward
zero once its constraint is slack; and under a stochastic policy that violates on
a persistent fraction of episodes it continues to grow at a rate set by the
violation frequency rather than by the expected slack. Eqs. (8) and (9) remove
the guaranteed divergence of Eq. (4), not the monotonicity. A symmetric variant
uses the signed slack (T_min − TP(τ))/H in Eq. (8) and, in Eq. (9), the summed
shortfall when any constrained server violates and the negative margin of the
binding server otherwise, so that a multiplier can fall back toward zero. Section
6.4 shows why the scale of the multipliers matters as much as the hinge, and
Section 6.7 reports the symmetric variant at full budget with multipliers on the
scale of the reward.

### 4.4 Evaluation of a stochastic policy

The initial protocol evaluated policies greedily. PPO optimises a stochastic
policy, and for a constrained problem the optimum may itself be randomised
(Section 2.2); more immediately, if the learned distribution is close to uniform
then its greedy action is selected by small and unstable differences in logits,
and evaluating it measures those differences rather than the policy. Both
evaluation modes are therefore computed and reported throughout, checkpoints are
selected under the same mode in which results are reported, and Section 6.5
measures directly how far the learned distributions are from uniform and how
stable their greedy action is to seed and to perturbation. This is described as
an evaluation-mode mismatch rather than an error: greedy evaluation is a
legitimate choice when the deployment target is a deterministic controller
(Section 7.3), and the problem is reporting it for a near-uniform policy without
saying so.

## 5. Experimental protocol

### 5.1 Baselines

Six dispatching rules are evaluated on the 50 test episodes (Table 1). Four are
state-aware: ShortestQueue minimises summed instantaneous load across the full
route; LeastUtilised weights each server's load by its mean service time;
CostMinimising minimises summed processing cost per unit time, ignoring queue
state; FastServerFirst always takes the fastest server at each stage. Two are
stateless and are included because the learned policies turn out to resemble
them (Section 6.5): UniformRandom draws a route uniformly at each arrival, and
RoundRobin cycles through routes in order. ShortestQueue has the lowest cost per
unit on electronics while satisfying both constraints on every test episode, and
is the primary comparator; on bakery it is not separable from RoundRobin, whose
mean is $1.11 lower with heavily overlapping intervals.

[Table 1 near here]


F_fast and U_min are identified from a one-shot LeastUtilised run on each
testbed. This anchors the feasible region on the heuristic family the agent is
compared against, and Table 1 shows the consequence: at these thresholds even
stateless routing satisfies both constraints on 78–90% of episodes, so the
constraints are not demanding. Section 6.6 reports how the ordering changes when
they are tightened.

### 5.2 Training, selection and statistical treatment

Five seeds committed in advance [42, 7, 2024, 123, 999]; 1.6M timesteps on
electronics and 1.5M on bakery; checkpoints every 100K. Validation uses seeds
[10000, 10020) and test uses seeds [11000, 11050), both disjoint from training.
Among checkpoints satisfying each constraint on at least 16 of 20 validation
episodes, the one with the lowest mean cost per unit is selected; if none
qualifies, the lowest mean cost per unit overall is selected and the run is
marked as a fallback. Cost per unit is computed per episode and averaged.
Confidence intervals across seeds are 95% two-sided t-intervals; those in Table 1
are across episodes, so the two are not directly comparable. PPO is the
Stable-Baselines3 implementation [26] with the hyperparameters of the
initial study, listed with the dual parameters in Appendix A.

Three protocol documents govern the runs and are in the repository. The first is
the protocol committed before the initial study. The second, committed before any
full-budget run reported here, requires both evaluation modes to be computed and
reported for every checkpoint and every test, moves checkpoint selection onto the
stochastic validation episodes so that selection and reporting share a criterion,
and records the segmentation of full-budget training into 400K-step blocks
carrying policy, optimiser and dual state across boundaries. The third, committed
before the runs of Section 6.7, defines the symmetric reward-scaled cell, fixes
its multiplier scales a priori, lists the comparisons to be reported and states
the three possible outcomes with the wording each would receive; no
hyperparameter of that cell was changed after results were seen. Two defects in
the pipeline are recorded as a single deviation. First, the second document
specified seeded action sampling and the training pipeline sampled from an
unseeded generator, so every stochastic test figure reported here is a seeded
re-evaluation of the selected checkpoints with per-episode records archived.
Second, the pipeline aggregated joint satisfaction across seeds as the smaller of
the two mean marginal rates, which is an upper bound on the per-episode joint
rate rather than that rate, so every joint-satisfaction figure reported here is
recomputed per episode from the archived records. Greedy cost per unit is
unaffected by either defect and reproduces the pipeline's values to the cent,
which serves as the fidelity check; greedy joint satisfaction moves by at most
1.2 percentage points under the second correction, the two stochastic draws
differ by at most $1.2 per unit and 5.6 percentage points at cell level, and no
comparison changes direction. Where the seeded and unseeded draws differ most,
the correction of Section 6.2 is worth 19 rather than 25 percentage points of
joint satisfaction.

Comparisons in Section 6.2 use a paired hierarchical bootstrap on the per-episode
test records, resampling training seeds with replacement within each learned cell
and test episodes with replacement jointly across the policies being compared, so
that a learned policy and a baseline are always scored on the same resampled
episodes. This replaces one-sample tests against baseline means, which treat
those means as known constants and overstate the evidence. Ten tests form the
pre-specified family on each testbed and are adjusted by the Holm procedure at
α = 0.05; the cell of Section 6.7 forms its own family of twelve.

## 6. Results

### 6.1 Multiplier saturation is a property of the slack signal

A 2×2 design crossing the base reward {shaped, cost-only} with the slack signal
{cumulative-rate, episode-level}, five seeds per cell at a reduced budget of 400K
steps on electronics, attributes saturation unambiguously (Appendix B, Table B1).
All ten seeds trained with the cumulative-rate signal end at the λ_T cap of
20,000; all ten trained with the episode-level signal end between 908 and 2,833,
an order of magnitude lower. There is no overlap and no exception at the seed
level. The base reward has no resolvable effect in either arm: under the paired
bootstrap of Section 5.2, holding the slack signal fixed, replacing the shaped
reward with the cost-only reward moves stochastic cost per unit by +$1.93
[−0.42, +4.06] under the cumulative-rate signal and −$0.22 [−2.07, +1.47] under
the episode-level signal, and joint satisfaction by −11.6 [−32.4, +8.0] and
+0.8 [−6.4, +8.0] percentage points. The discrepancy noted in Section 4.1 is
therefore a documentation matter rather than a driver of the result, although the
first of those intervals is wide enough that a moderate effect under the
saturated signal cannot be excluded at five seeds.

The pattern holds at full budget on both testbeds (Figure 1). Under the
cumulative-rate signal the multiplier ramps monotonically to its cap on every
seed, at 0.19–0.24M steps on electronics and 0.59–0.65M on bakery, and is flat
thereafter; the difference in saturation time is what Section 4.2 predicts, since
the fill-phase gap is larger where flow time is a larger fraction of the horizon
and the floor sits closer to the achievable rate. Under the episode-level signal
the multiplier rises slowly, advancing only on episodes that violate, and ends
between 2,525 and 7,858, never within a factor of 2.5 of the cap. It does not
saturate, but neither has it converged, which is the monotonicity anticipated in
Section 4.3.

[Figure 1 near here]


### 6.2 Full-budget comparison

Two cells were run at the full budget: the original implementation (shaped
reward, cumulative-rate slack) as control, and the corrected formulation of
Section 4.3 (cost-only reward, episode-level slack). Five seeds each, selection on
stochastic validation, both evaluation modes on the 50 test episodes. Table 2
reports the outcomes and Table 3 the paired comparisons.

[Table 2 near here]


[Table 3 near here]


Four results follow. First, every seed in every cell now satisfies the validation
criterion, including the unmodified original signal, which under the initial
protocol satisfied it on 0 of 5 electronics seeds. Evaluation mode alone reverses
the pass/fail verdict on the harder testbed. Second, on electronics the slack
correction is worth 25 percentage points of joint satisfaction and about $4 per
unit against the original signal, and both differences survive adjustment. Third,
on bakery it makes no measurable difference in either metric; the multiplier
still saturates on all five control seeds, so the mechanism of Section 4.2 is
present, but on the easier problem the saturated penalty does not damage the
policy enough to show. That is reported as a null result, not a partial success.

Fourth, and most consequential, the corrected agent does not separate from
stateless routing on cost per unit. The paired difference against RoundRobin is
−$0.79 [−2.74, +1.37] and against UniformRandom −$1.81 [−3.96, +0.41]; these
intervals are wide enough to contain differences of a few dollars in either
direction, so this is a failure to detect rather than a demonstration of
equality. On joint satisfaction the agent is ahead of UniformRandom by 17 points,
which survives adjustment, and ahead of RoundRobin by 9 points, which does not.
On bakery no comparison against any rule is significant. ShortestQueue remains
ahead of every learned policy: 6.2% cheaper per unit on electronics with 100%
constraint satisfaction, and the only comparator on that testbed against which
the corrected agent is significantly worse on both metrics.

A further feature of Table 2 is the dispersion between seeds. Under greedy
evaluation the electronics intervals are ±43 to ±76; under stochastic evaluation
of the same checkpoints they are ±1.6. The severe seed-to-seed instability that
the initial study reported was, to a first approximation, the variance of a
greedy action taken over near-uniform distributions.

### 6.3 Re-evaluation of the archived checkpoints

The initial study's checkpoints were re-evaluated without retraining: the
one-sided Lagrangian variant of Section 4.2, and a PID-Lagrangian comparator with
exponentially smoothed slack in the manner of Stooke and colleagues
[22], on both testbeds and every archived seed. Applying the original
greedy selection rule to the archived validation sweep reproduces the originally
selected checkpoint in 18 of 18 runs and the recorded test figures to the cent,
so the pipeline evaluated here is the one that produced them. Two questions are
asked of each run: what the originally selected checkpoint scores when evaluated
stochastically (Q1), and which checkpoint stochastic selection would have chosen
and what it scores (Q2). Table 4 reports both.

[Table 4 near here]


The five electronics checkpoints recorded as failing systematically, at $113.31,
$112.08, $95.38, $118.35 and $124.44, score $83.78, $81.23, $78.68, $81.73 and
$81.33 as stochastic policies, with joint satisfaction of 64–88%. The files are
unchanged. Had the original protocol selected on stochastic validation, all five
would have been reported as satisfying the criterion at $81.59 ± 2.68. The
negative result is thus reproduced and then reversed on its own artefacts,
without retraining. These scores are, once again, at the level of RoundRobin and
UniformRandom in Table 1.

The bakery rows show the complementary pattern: greedy and stochastic scores lie
within a few dollars of each other on most seeds, and greedy selection already
satisfied validation on 4 of 5 seeds. Greedy extraction finds a feasible
deterministic policy often on bakery and almost never on electronics. This is not
because no feasible deterministic policy exists there, since ShortestQueue is one
and a better one than anything learned, but because the learned distributions on
electronics are close enough to uniform that their greedy action is arbitrary.

### 6.4 The scale of the constraint terms against the objective

The magnitude of the penalty relative to the objective is the third link in the
chain, and it needs stating in the units the agent actually optimises. By Eq. (1)
the per-episode base return under cost-only weights is −κ C(τ)/C_norm, which is
−10.1 to −10.6 on both testbeds; the initial study's analysis compared the
integrated penalty against a per-episode cost expressed in dollars and thereby
mixed units. Evaluated correctly, the penalty of Eq. (7) on the test episodes of
the corrected runs is already 190–300 times larger in magnitude at the initial
multipliers (λ_T, λ_U) = (200, 5), and 1,200–10,000 times larger at the
multipliers reached by the end of training. For the control cell at saturation
the ratio is of order 10⁴.

Two features of that term govern what follows. Its sign is negative on almost
every episode, because both constraints are over-satisfied by any load-spreading
policy: mean throughput is 57.0 against a floor of 50 on electronics and 20.2
against 18 on bakery, and mean fast-server utilisation is 0.79–0.91 against a
floor of 0.50. The term therefore acts as a reward for surplus throughput and
surplus fast-server busy time rather than as a penalty. And it cannot shrink,
because Eqs. (8) and (9) never decrease a multiplier, whereas the equilibrium
value of a multiplier on a slack constraint is zero. The comparison of returns
understates the imbalance seen by the policy gradient: the cost term of Eq. (1)
differs between the routing actions available at a decision by about 10⁻³ per
step, whereas a single departure moves the per-step term of Eq. (6) by λ_T,
between 200 and 7,858.

In every one-sided configuration of this study, therefore, the original and the
corrected alike, the objective actually optimised was to within a fraction of a
percent the throughput and fast-server utilisation terms, and the cost term of
Eq. (2) was numerically irrelevant. This is independent of the two artefacts and
follows from two choices inherited unchanged from the initial study: multiplier
initial values and step sizes set in the units of the per-step signal of Eq. (4),
which are large relative to a reward whose episode return is of order ten, and
the hinge in the dual update. Section 6.7 reports the experiment that removes
both.

### 6.5 Structure of the learned policies

For the selected checkpoint of every full-budget run, the stochastic policy was
rolled out on ten test episodes and the entropy and largest probability of the
action distribution recorded at every decision, together with three measures of
how much the greedy action means, computed on a common set of 2,395 states
visited under uniform-random routing (Appendix B, Table B2).

The hinged cells learn near-uniform routers with a modest tilt. On electronics
the most probable route at a typical decision carries 13–18% of the mass against
8.3% for uniform, normalised entropy is 0.94–0.98 and the effective number of
routes is 10.4–11.5 of 12. The tilt is not random: the corrected policies load
the two constrained fast servers to 0.82 and 0.79 mean utilisation on
electronics, against 0.65 and 0.69 under uniform-random routing and 0.81 and 0.90
under ShortestQueue. That is the response one would expect to the objective
identified in Section 6.4, in which surplus fast-server busy time is rewarded,
and it buys constraint reliability without buying cost efficiency.

The greedy-action diagnostics show why evaluating these policies by their mode
measured so little. On electronics the top action leads the runner-up by two
percentage points of probability on average; two seeds' greedy policies agree on
18% of states in the corrected cell and 9% in the control, against 8.3% by
chance; and logit noise of standard deviation 0.1 changes the greedy action at a
quarter of visited states. Greedy evaluation therefore selects a route determined
by differences that neither replicate across seeds nor survive small
perturbations, and it has no reason to be feasible, which is the origin of the
wide intervals in Table 2. On bakery, with four routes, the gaps are larger and
agreement higher, and the greedy feasibility rate of 54% is correspondingly
better.

Two robustness checks. Single-draw stochastic evaluation introduces sampling
variance of its own: re-evaluating one electronics checkpoint five times with
different sampling seeds gives cost per unit 79.11–81.63 (SD 0.96) and joint
satisfaction 88–96% (SD 0.036), narrower than the between-seed intervals of
Table 2, and the bootstrap of Table 3 resamples episodes as well as seeds. And
the control cell is, if anything, closer to uniform than the corrected cell,
consistent with Section 6.1: a saturated penalty four orders of magnitude larger
than the base reward leaves the policy gradient with almost nothing to say about
routing.

### 6.6 Sensitivity to the constraint thresholds

The thresholds were set from a one-shot LeastUtilised run, and Table 1 shows
that at those values stateless routing satisfies both constraints on most
episodes. Joint satisfaction was therefore recomputed from the archived
per-episode records at stricter thresholds, for the dispatching rules and for the
stochastic policies of the learned cells, without retraining (Appendix B,
Table B3). This is a post-hoc re-scoring of policies trained for the original
thresholds, not an experiment on policies trained for the stricter ones.

Two patterns emerge. Raising the utilisation floor separates policies that route
preferentially to the fast servers from those that do not: at U_min = 0.70 on
electronics, ShortestQueue and LeastUtilised still satisfy on 96% of episodes,
uniform-random routing on 16%, and the corrected agent on 71%, so the utilisation
tilt of Section 6.5 recovers roughly two-thirds of the gap between stateless
routing and the state-aware rules. Raising the throughput floor does the
opposite. At T_min = 58 on electronics, where ShortestQueue still satisfies on 86%
of episodes, the corrected agent falls to 46%, closer to RoundRobin at 30% than
to ShortestQueue, and at T_min = 60 every learned or stateless policy is at or
below 30%. Meeting a demanding throughput floor requires the queue-sensitive
routing that ShortestQueue performs and the learned policies do not; the
constraint the agent was trained against did not require it.

### 6.7 A functioning dual, and what it reveals about the objective

The final cell removes the two features identified in Section 6.4 as holding the
multipliers away from equilibrium: the hinge, replaced by the signed update of
Section 4.3, and the inherited scale, replaced by multipliers initialised,
stepped and capped on the scale of the reward (λ_T: 0.1, step 2.0, cap 10;
λ_U: 0.002, step 0.01, cap 0.2). Those values were fixed before any run by two
independent arguments that agree: the shadow price of a unit of throughput
implied by the interpolation of Section 3.4, about 0.1 reward units, and of a
busy-step of a fast server, about 0.002; and the factor of roughly 2,400 by which
the inherited multipliers must be divided to bring the penalty on a typical
episode to a tenth of the return. Five seeds, full budget, electronics.

[Table 5 near here]


[Figure 2 near here]


The dual now behaves as a dual. λ_T is released to zero within a few episodes
whenever the constraint is slack, rises when it is violated, and spends training
oscillating between 0 and about 0.5, with a per-seed training mean of 0.06–0.13
that brackets the a priori shadow-price estimate; it is positive on 60–72% of
episodes and never approaches its cap. The constraint terms are consequently on
the scale of the cost term throughout, at 7–14% of the return at the
training-mean multipliers, against 190 to 10,000 times the return in the hinged
cells.

The policy responds in the direction the total-cost objective pushes it. Within
the first 400K steps, mean throughput per training episode falls from the level
of random routing, 4.4 units above T_min, to 1–3 units above it and stays there
(Figure 2b), with 24–39% of training episodes below T_min in every quarter of
every seed. The expectation constraint of Eq. (2) is thus satisfied during
training with a margin of one to three units, and the selected checkpoints
satisfy it on test with a margin of 4.1 units. The chance constraints are met
only narrowly and the joint criterion is not met at all. Of the 95 checkpoints
validated across the five seeds only 14 qualify under the 16-of-20 rule, although
every seed had at least one and none fell back; on test the selected checkpoints
satisfy the throughput constraint on 84.8% of episodes and the utilisation
constraint on 82.0%, both just above the 80% the rule demands, but they satisfy
the two jointly on only 74.8%, 20 points below the corrected cell and 25 below
ShortestQueue. This is the configuration promised in Section 3.3: it satisfies
Eq. (2) and both chance constraints while failing the joint criterion that
results report.

On the reported metric the cell is no better. Its $81.91 ± 3.47 is significantly
above ShortestQueue and indistinguishable from RoundRobin, UniformRandom,
LeastUtilised and the original signal (Table 3). On total cost, the quantity it
actually minimises, it is not the cheapest either: mean episode cost is $4,414
for this cell against $4,342 for RoundRobin, $4,346 for UniformRandom, $4,445 for
the corrected cell, $4,493 for ShortestQueue and $4,542 for LeastUtilised. Under
the paired bootstrap of Section 5.2 applied to total cost the cell sits $72 above
RoundRobin [−$42, +$183] and $67 above UniformRandom [−$53, +$187], neither
resolvable, and $129 below LeastUtilised [−$229, −$30], which is. A functioning
dual therefore buys a real total-cost improvement over one state-aware rule and
none over stateless routing. That
comparison is the direct measurement promised in Section 3.4. Across the seven
load-spreading policies in this study, mean episode cost spans 4.6%, from $4,342
for RoundRobin to $4,542 for LeastUtilised, because the work to be processed is
fixed by the arrival process and routing changes only which server performs it
and how long jobs wait; throughput across the same policies spans 15.5%, from
53.2 to 61.4 units. An objective that is nearly flat in routing and indifferent to
throughput above the floor has little to teach a policy gradient about routing,
and what it does teach, that throughput above the floor is not worth paying for,
is the opposite of what cost per unit rewards.

What the cell did learn is a different kind of policy from every other cell here.
Its action distributions are concentrated: normalised entropy 0.63–0.83 against
0.94–0.97 for the hinged cells, 4.8–7.8 effective routes against 10.4–11.3, and a
mean top-1 probability of 0.28–0.48 against 0.13–0.18. Yet its marginal route
frequencies across an episode remain near-uniform at 0.89–0.96, so the
concentration is state-dependent: this policy routes differently in different
states, which the near-uniform cells did not. It is also randomised in the sense
of Section 2.2 rather than merely flat. Its greedy action is a coherent
conservative router, throughput 41.8 at $110 per unit and feasible on 10% of
episodes, while the stochastic policy from which it is extracted reaches 54 units
at $82; the randomisation carries a quarter of the throughput. Greedy extraction
fails here for the reason the CMDP literature predicts rather than because the
logits are flat, and logit noise of 0.1 now moves the greedy action at 10% of
states rather than 25%.

A functioning dual on the stated CMDP therefore does not produce a policy
competitive with ShortestQueue on either metric. It produces a state-dependent,
randomised policy that the total-cost objective drives to the throughput floor,
satisfying the expectation constraint and both chance constraints, failing the
joint criterion on a quarter of episodes, and no cheaper per unit than stateless
routing. Whether it approaches the total-cost optimum cannot be established here:
its total cost is not below that of stateless routing, so the objective was
pursued but not demonstrably solved.

## 7. Discussion

### 7.1 Where the chain broke, and what each break cost

The three breaks examined here differ in kind and in consequence. The first, a
per-step surrogate for an episode-level constraint, is a modelling error with a
specific and generalisable form: in any system with non-trivial transit time, a
cumulative-average translation is biased during the fill phase of every episode.
The bias is small, of order 10⁻³ per step, and would be harmless under a
symmetric dual update that released the multiplier once the line filled. Under a
monotone one-sided update it integrates without bound. Each choice is defensible
alone and the combination is not, which is what makes this class of error hard to
detect. Its consequence is problem-dependent: the multiplier saturates on every
control seed of both testbeds, but the damage to the policy is measurable only on
the 12-route instance.

The second, evaluating a near-uniform stochastic policy by its mode, is what
reverses the reported verdict, and Section 6.3 shows it doing so on the archived
checkpoints without retraining. It would have gone unnoticed had the first break
not prompted a re-examination of the pipeline.

The third, the scale of the multipliers, is the one that survives correction of
the other two and explains the residual result. It is also the least visible: no
diagnostic in the original pipeline would have surfaced it, whereas a single
ratio of penalty magnitude to base return, reported at initialisation and at the
end of training, exposes it immediately.

### 7.2 What the method learned, and what it was asked to learn

Correcting the first two breaks does not produce a competitive agent; it produces
a near-uniform router with a tilt toward the fast servers, indistinguishable from
RoundRobin and UniformRandom on cost per unit and ahead of UniformRandom on
constraint reliability. Section 6.4 supplies the reason, and it is not the one
originally offered. The agent was not defeated by a penalty that overwhelmed its
objective; its objective was, for practical purposes, the constraint terms. With
the multipliers two to four orders of magnitude above the scale of the cost term,
never released, and with both constraints slack under any load-spreading policy,
the augmented return rewarded surplus throughput and surplus fast-server busy
time and registered cost only in the third or fourth significant figure. A
near-uniform router tilted toward the fast servers is a reasonable response to
that objective, and it is what every hinged cell produced.

Section 6.7 closes the question. With multipliers on the scale of the reward and
free to fall, PPO does leave the uniform policy and learns a state-dependent
router, and that router moves toward the throughput floor as the stated CMDP
directs. The mechanism, measured across the seven load-spreading policies of the
study, is that total episode cost is nearly flat in routing while throughput is
not, so cost per unit falls with throughput across that range. An agent
minimising total cost has no reason to buy throughput above the floor, so a
working Lagrangian moves the policy away from the region where the reported
metric is lowest, and the joint criterion the protocol reports is stricter than
the expectation constraint the Lagrangian enforces. The phrase "failed to learn" therefore needs
qualification twice over. Constrained PPO failed to learn a cost-efficient,
state-aware routing policy competitive with ShortestQueue. In the hinged cells it
was never seriously asked to, because cost carried negligible weight in the
objective; in the symmetric cell it was asked to minimise a quantity that is not
the metric, and moved in that direction.

Three consequences follow for anyone posing this problem to a constrained
learner. If cost per unit is the criterion by which routing performance is
judged, the learning objective should be aligned with it, through a ratio
objective or a throughput term whose weight is set by the reported metric rather
than by a constraint; on these testbeds a per-departure bonus of that kind
narrowed but did not close the gap to ShortestQueue for unconstrained PPO
[16]. A CMDP that minimises total cost is a legitimate
formulation, but it should then be judged on total cost, and here it was not. The
constraint enforced should be the constraint checked: a joint per-episode
criterion needs a per-episode formulation, such as a penalty on the violation
indicator or a conditional-value-at-risk constraint on the joint event, rather
than an expectation constraint whose satisfaction, with both marginal chance
constraints met, still leaves a quarter of episodes jointly infeasible. And a functioning dual is necessary but not sufficient; here
it changed what was learned without making it competitive.

Until those changes are made and tested, the practical recommendation is
unchanged, though for a different reason than originally given. Tuned dispatching
rules remain the baseline to beat on problems of this size and structure. The
constrained agent no longer exhibits systematic constraint failure; it meets the
protocol's criterion either as a fast-server-biased random router does or, under
a functioning dual, by satisfying the expectation constraint and both marginal
chance constraints while failing the joint criterion on a quarter of episodes,
and in both cases it costs more per unit than a rule that reads the queues.

### 7.3 Deterministic deployment

Feasible deterministic policies exist on both testbeds; ShortestQueue is one.
What was not obtained is a feasible deterministic policy extracted from a learned
agent on the harder testbed, for two different reasons in the two kinds of cell.
Greedy extraction from the near-uniform distributions of the hinged cells yields
an arbitrary route. Greedy extraction from the concentrated distributions of the
symmetric cell yields a coherent but conservative router, feasible on 10% of
episodes, because the randomisation carries a quarter of the throughput; this is
the randomised-optimum structure of Section 2.2 observed directly. In plants that
cannot deploy a stochastic controller, and auditability and operator trust are
good reasons for such a requirement, the learned policies studied here are not
deployable, and the corrections described do not change that. The greedy column
is reported throughout so that this remains visible.

### 7.4 Reporting practice

Two systematic reviews have criticised this field for weak baselines and
simulation-only validation. This study suggests four further items. Where a
constrained method is reported to fail its constraints, a reader should be able to
verify that the constraint optimised is the constraint stated: the exact
augmented per-step reward, the exact dual update, the units of the slack signal
and its measured value under a known-good reference policy should all be
reported. The magnitude of the constraint terms relative to the objective, at
initialisation and at the end of training, should be reported with them; a single
ratio would have exposed the third break immediately. Where a stochastic policy
is evaluated, the mode should be stated and both modes reported, together with a
measure of how far the learned distribution is from uniform. And where a learned
router is compared against dispatching rules, uniform-random and round-robin
routing should be among them, evaluated under the same protocol with their own
uncertainty propagated; without that control this study would have reported a
recovery where there was only a near-random policy meeting undemanding
constraints.

## 8. Limitations

The two instances differ simultaneously in action-space size, stage count,
capacity ratios and reward scale, so while the failure can be attributed to the
implementation, which is manipulated directly, the difference in difficulty
between instances cannot be attributed to any single structural feature; a
controlled feature ablation remains outstanding. Five seeds per cell is a small
sample, and although stochastic-mode intervals are tight (±0.7 to ±2.0 on cost
per unit for the four cells of Table 2, and ±3.5 for the symmetric cell), the
bootstrap resamples five seed-level means, so it has 126 distinct resamples in
the seed dimension and is anti-conservative at this cluster count; its
between-cell intervals should be read as approximate and its p-values between
0.01 and 0.05 as no better than indicative. The null comparisons
against stateless routing are failures to detect at this sample size rather than
evidence of equality: the electronics interval against RoundRobin spans −$2.74 to
+$1.37 per unit.

The constraint thresholds were set from a one-shot LeastUtilised run, and Table 1
shows that stateless routing satisfies them on 78–90% of episodes, so the
reliability comparison is a comparison at undemanding thresholds. Section 6.6
shows the ordering changing when they are tightened, but it re-scores policies
trained for the original thresholds; agents trained against stricter ones were
not run. Each validation and test episode samples a single action trajectory,
bounded in Section 6.5 at roughly a third of the between-seed interval, and the
validation draws that selected the checkpoints were single and unseeded. The
re-selection in Q2 of Section 6.3 uses the same stochastic validation episodes
now used for reporting, so the Q1 columns are the cleaner like-for-like
comparison and the principal claim rests on them. The amendment that introduced
stochastic selection was motivated by the reduced-budget ablation and is
therefore not independent of data from the same study, although it was committed
before the runs it governs.

Finally, the symmetric cell tests a functioning dual in one configuration only:
multiplier scales fixed a priori but after the earlier results had been
inspected, plain dual ascent without damping or averaging, the original entropy
coefficient, and the electronics instance alone. Its λ_T oscillates rather than
converging (Figure 2a), and a damped or averaged dual might select different
checkpoints. The mechanism it exposes, an objective flat in routing and
indifferent to throughput above the floor, is a property of the objective
measured across the seven load-spreading policies of the study and is robust to
those choices, but the cell's numbers are those of one configuration. All results are from simulation;
there is no physical validation.

## 9. Conclusions

A reported failure of constrained reinforcement learning on a multi-server
flow-shop routing problem was traced to three breaks in the chain running from
the constraint stated to the metric reported. A per-step cumulative-rate
surrogate for an episode-level constraint drove the multiplier upward for any
policy and to its cap within the training budget for an oracle dispatching rule.
Greedy evaluation of near-uniform stochastic policies measured a deterministic
router selected by unstable logit differences, and correcting it alone reverses
the reported verdict on the archived checkpoints without retraining. With both
corrected, every seed satisfies the constraint criterion, but the agent is not
distinguishable from round-robin routing on cost per unit at this sample size and
is 6.6% more expensive than ShortestQueue, while the multipliers outweigh the
cost term by two to four orders of magnitude and cannot be released. Giving the
dual the scale of the reward and a symmetric update makes it behave as a dual and
produces a state-dependent policy, which then moves to the throughput floor as a
total-cost objective directs, and it is no better on the reported metric than the
stateless rules and remains significantly worse than ShortestQueue.

The engineering conclusion is that the objective posed, the constraint enforced,
the surrogate trained against, the evaluation mode and the reported metric form a
chain that must be coherent, and that on this problem it was not. Total cost is
nearly flat in routing while cost per unit falls with throughput, so a total-cost
CMDP is the wrong instrument for a cost-per-unit objective however well its dual
behaves. Separating these claims required reporting the constraint actually
implemented, the scale of the penalty against the objective, the evaluation mode
actually used, and a random-routing control with its own uncertainty; all four
are recommended as routine practice for simulation studies of learned production
control.

## Author contributions

The author conceived and designed the work, implemented the simulator, wrappers
and analysis, conducted the experiments, analysed and interpreted the data,
drafted the paper and revised it critically for intellectual content, approved
the version to be published, and agrees to be accountable for all aspects of the
work.

## Funding

This research received no specific grant from funding agencies in the public,
commercial, or not-for-profit sectors.

## Disclosure of interest

The author reports there are no competing interests to declare.

## Declaration of generative AI use

During the preparation of this work the corresponding author used a large
language model assistant (Claude, Anthropic) to assist with simulation code,
analysis scripting, drafting and editing. The author verified all results and
takes full responsibility for the content of the article.

## Data availability statement

The simulator, the Lagrangian wrappers, the three protocol documents, the
deviation record, all runner and analysis scripts, and every per-seed summary,
multiplier history and per-episode test record supporting the results reported
here are openly available in the FlexFlowSim-CPPO repository at
https://github.com/alrashdank/flexflowsim-cppo (branch `ablation-slack-fix`).
The bakery service-time data are from the openly available dataset of Babor and
Hitzmann [25].

## References

1. Mayerhoff J, Schmidt M. Reinforcement learning for autonomous production planning and control: a systematic literature review. J Manuf Syst. 2026;86:546–568. doi:10.1016/j.jmsy.2026.03.023.

2. Schneider J, Pfannschmidt C, Nyhuis P, Schmidt M. The role of reinforcement learning in production control: a systematic literature review. IEEE Access. 2026;14:34375–34389. doi:10.1109/ACCESS.2026.3668903.

3. Schulman J, Wolski F, Dhariwal P, Radford A, Klimov O. Proximal policy optimization algorithms. arXiv:1707.06347; 2017. doi:10.48550/arXiv.1707.06347.

4. Alrashdan KR. FlexFlowSim-CPPO: simulator, protocol documents and archived results [software]. GitHub; 2026. Available from: https://github.com/alrashdank/flexflowsim-cppo

5. Li C, Zhao X, Lin L, Zhang W, Gen M, Zhang Q. An evolutionary knowledge training-based proximal policy optimization algorithm for job shop scheduling in flexible intelligent manufacturing. Comput Ind Eng. 2025;210:111533. doi:10.1016/j.cie.2025.111533.

6. Liu Y, Fan J, Shen W. A deep reinforcement learning approach with graph attention network and multi-signal differential reward for dynamic hybrid flow shop scheduling problem. J Manuf Syst. 2025;80:643–661. doi:10.1016/j.jmsy.2025.03.028.

7. Shen Y, Zhang X, Jin T. Transformer-based multi-agent reinforcement learning for flexible job shop scheduling with AGVs. Appl Soft Comput. 2026;193:114899. doi:10.1016/j.asoc.2026.114899.

8. Wang R, Jing Y, Gu C, He S, Chen J. End-to-end multitarget flexible job shop scheduling with deep reinforcement learning. IEEE Internet Things J. 2025;12(4):4420–4434. doi:10.1109/JIOT.2024.3485748.

9. Ali AM, Tirel L. Action masked deep reinforcement learning for controlling industrial assembly lines. In: 2023 IEEE World AI IoT Congress (AIIoT); 2023. doi:10.1109/AIIoT58121.2023.10174426.

10. Tang C-Y, Liu C-H, Chen W-K, You SD. Implementing action mask in proximal policy optimization (PPO) algorithm. ICT Express. 2020;6(3):200–203. doi:10.1016/j.icte.2020.05.003.

11. Zhang N, Liu B, Zhang J. Dual resource scheduling method of production equipment and rail-guided vehicles based on proximal policy optimization algorithm. Technologies. 2025;13(12):573. doi:10.3390/technologies13120573.

12. Ferreira C, Figueira G, Amorim P. Effective and interpretable dispatching rules for dynamic job shops via guided empirical learning. Omega. 2022;111:102643. doi:10.1016/j.omega.2022.102643.

13. Huang Z, Mei Y, Zhang F, Zhang M. Toward evolving dispatching rules with flow control operations by grammar-guided linear genetic programming. IEEE Trans Evol Comput. 2025;29(1):217–231. doi:10.1109/TEVC.2024.3353207.

14. Marques N, Figueira G, Guimarães L. Dynamic dispatching rule selection for the job shop scheduling problem. Comput Ind Eng. 2025;210:111471. doi:10.1016/j.cie.2025.111471.

15. Doherty M, Matzner R, Sadeghi R, Bayvel P, Beghelli A. Reinforcement learning for dynamic resource allocation in optical networks: hype or hope? J Opt Commun Netw. 2025;17(9):D1–D17. doi:10.1364/JOCN.559990.

16. Alrashdan KR. Routing under machine breakdowns: a benchmark of dispatching rules, bandits, and reinforcement learning for multi-server flow shops. J King Saud Univ Eng Sci. 2026;38:55. doi:10.1007/s44444-026-00128-9.

17. Rinciog A, Meyer A. Fabricatio-RL: a reinforcement learning simulation framework for production scheduling. In: Proceedings of the 2021 Winter Simulation Conference; 2021. doi:10.1109/WSC52266.2021.9715366.

18. Altman E. Constrained Markov decision processes. Boca Raton (FL): Chapman & Hall/CRC; 1999.

19. Achiam J, Held D, Tamar A, Abbeel P. Constrained policy optimization. In: Proceedings of the 34th International Conference on Machine Learning. PMLR 70; 2017. p. 22–31.

20. Tessler C, Mankowitz DJ, Mannor S. Reward constrained policy optimization. In: 7th International Conference on Learning Representations; 2019.

21. Paternain S, Chamon LFO, Calvo-Fullana M, Ribeiro A. Constrained reinforcement learning has zero duality gap. In: Advances in Neural Information Processing Systems 32; 2019.

22. Stooke A, Achiam J, Abbeel P. Responsive safety in reinforcement learning by PID Lagrangian methods. In: Proceedings of the 37th International Conference on Machine Learning. PMLR 119; 2020. p. 9133–9143.

23. Henderson P, Islam R, Bachman P, Pineau J, Precup D, Meger D. Deep reinforcement learning that matters. In: Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence; 2018. p. 3207–3214. doi:10.1609/aaai.v32i1.11694.

24. Agarwal R, Schwarzer M, Castro PS, Courville A, Bellemare MG. Deep reinforcement learning at the edge of the statistical precipice. In: Advances in Neural Information Processing Systems 34; 2021. p. 29304–29320.

25. Babor M, Hitzmann B. Small and medium-sized bakery production data for scheduling. Version 2 [dataset]. Mendeley Data; 2022. doi:10.17632/dhgbssb8ns.2.

26. Raffin A, Hill A, Gleave A, Kanervisto A, Ernestus M, Dormann N. Stable-Baselines3: reliable reinforcement learning implementations. J Mach Learn Res. 2021;22(268):1–8.

## Appendix A. Reproducibility

**A.1 Software.** All experiments use the FlexFlowSim-CPPO simulator
[4]. The testbeds are defined in `configs/bakery_bk50.json` and
`configs/electronics_3stage.json`. The original Lagrangian wrapper is
`pilot_constrained_v4_auto.py`; the corrected wrapper, which also reproduces the
original signal as a control mode and provides the symmetric variant, is
`lagrangian_slack.py`. Training, validation and test are driven by
`run_ablation_2x2.py`, the archival re-evaluation by `archival_reeval.py`, and the
policy statistics, seeded per-episode re-evaluation, bootstrap and threshold
re-scoring by `analysis_stateless_and_entropy.py`, `analysis_r2.py`,
`analysis_r2_stats.py` and `analysis_r2_tables.py`. The fill-phase statistics of
Section 4.2, the saturation episodes read from the archived multiplier
histories, the marginal satisfaction rates and checkpoint counts of Section 6.7,
and the total-cost and base-reward bootstraps of Sections 6.7 and 6.1 are
produced by `analysis_audit_r3.py`, which writes `r3_numbers.json`. The protocol is `protocol.md`;
its amendments are `protocol_amendment_r1.md` and `protocol_amendment_r2.md`;
deviations are recorded in `protocol_deviations.md`.

**A.2 Hyperparameters.** PPO (Stable-Baselines3, MlpPolicy, two hidden layers of
64 tanh units): learning rate 3 × 10⁻⁴; n_steps 2048; batch size 64; 10 epochs;
γ = 0.99; GAE λ = 0.95; clip range 0.2; entropy coefficient 0.01; value-function
coefficient 0.5; maximum gradient norm 0.5. Lagrangian outer loop, identical in
both slack modes: U_min = 0.50; T_min = 18 (bakery) or 50 (electronics); λ_U
initial 5, step 20, cap 500; λ_T initial 200, step 4000, cap 20,000. The
cumulative-rate mode applies a 100-step warm-up and the episode mode none.
Multipliers are updated once per episode in all modes. The symmetric cell of
Section 6.7 uses the signed update with λ_T initial 0.1, step 2.0, cap 10 and λ_U
initial 0.002, step 0.01, cap 0.2, with all PPO hyperparameters unchanged.

**A.3 Budgets and seeds.** Training seeds [42, 7, 2024, 123, 999] throughout. The
mechanism ablation uses 400K timesteps per run over four cells on electronics;
the full-budget comparison uses 1.6M timesteps on electronics and 1.5M on bakery
over two cells; the symmetric cell uses 1.6M timesteps on electronics. Full-budget
runs are executed in 400K-step segments carrying policy weights, optimiser state,
dual variables and the dual episode counter across boundaries; only the in-flight
rollout buffer is flushed, three times in 1.6M steps. Checkpoints are written
every 100K timesteps and additionally at each 400K segment boundary, giving 19
per full-budget electronics run and 18 per bakery run; because PPO completes
whole 2,048-step rollouts, checkpoint labels are nominal and actual step counts
exceed them by up to 4,224 steps (0.3%), identically in every cell. A full-budget
run completes 3,352 episodes on electronics and 3,142 on bakery. Validation seeds are [10000, 10020) and test
seeds [11000, 11050), both disjoint from training.

**A.4 Evaluation.** Every checkpoint is evaluated on the 20 validation episodes in
both modes: greedy, and stochastic with one action sample per step and one
trajectory per episode. Selection follows Section 5.2 on the stochastic results,
with greedy selection also computed and stored. The selected checkpoint is
evaluated on the 50 test episodes in both modes; reported stochastic figures are
the seeded re-evaluation described in Section 5.2, with the generator seeded by
the episode seed before each episode. Cost per unit is total episode cost divided
by episode departures, averaged over episodes. Joint satisfaction is the fraction
of episodes in which every constrained server's realised utilisation is at least
U_min and departures are at least T_min, computed per episode.

**A.5 Statistical treatment.** All comparisons in Table 3 use a paired
hierarchical bootstrap on the per-episode test records. Let X be the seeds ×
episodes matrix of a learned cell's per-episode metric (5 × 50) and Y the
comparator's (5 × 50 for another learned cell, 1 × 50 for a dispatching rule).
Each of 10,000 replicates draws seed indices with replacement, independently for
X and Y, and a single vector of 50 episode indices with replacement applied to
both, so the two policies are compared on the same resampled episodes; the
statistic is the difference of grand means. Intervals are 2.5th–97.5th
percentiles; p is twice the smaller tail proportion at zero, floored at 1/10,000.
The pre-specified family on each testbed is the corrected cell against the four
rules of Table 1 and against the original signal, on cost per unit and on joint
satisfaction, giving ten tests adjusted by Holm's step-down procedure at
α = 0.05; the symmetric cell forms its own family of twelve. Confidence intervals
use the t-distribution with n − 1 degrees of freedom across seeds and the normal
approximation across episodes.

**A.6 Hardware and runtime.** Training and evaluation ran on a two-vCPU Linux
container at approximately 800–1,000 environment steps per second. A 400K-step
segment takes 7–8 minutes and a full 1.6M-step run about 30 minutes plus 10–15
minutes of dual-mode validation. The complete matrix reported here, comprising 20
reduced-budget runs, 25 full-budget runs, 18 archival re-evaluations of 279
checkpoints, the baselines and all analyses, consumed approximately 21
compute-hours.

## Appendix B. Supplementary tables

[Table B1 near here]


[Table B2 near here]


[Table B3 near here]


## Appendix C. Nomenclature

| Symbol | Meaning |
|---|---|
| PPO | Proximal Policy Optimization |
| CMDP | Constrained Markov decision process |
| CPU | Cost per unit ($ per completed job; primary metric) |
| TP(τ) | Throughput: jobs completed in episode τ |
| C(τ), c_t | Episode cost total; cost accrued in step t |
| H, Δt | Episode horizon, 480 steps; step length, 1 min |
| F_fast | Set of fast servers subject to the utilisation constraint |
| T_min, U_min | Throughput floor; per-server utilisation floor |
| λ_T, λ_U | Lagrange multipliers for the throughput and utilisation constraints |
| η_T, η_U | Multiplier step sizes |
| g_T,t, g_U,t | Per-step slack signals of the original implementation, Eq. (4) |
| p_t | Corrected per-step penalty, Eq. (6) |
| d_t, δ_t | Cumulative departures to step t; departures in step t |
| b_{i,t} | In-service indicator of server i at step t |
| ū_i(τ), u_i(τ) | Realised utilisation of server i over episode τ |
| W_t | Work in process at step t |
| κ, w_c, w_t, w_w | Reward scale and shaping weights of Eq. (1) |
| C_norm, N_norm, W_norm | Reward normalisation constants of Eq. (1) |
| π | Policy |

## Tables

\newpage

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

\newpage

**Table 2.** Full-budget comparison. CI is the 95% two-sided t-interval across
seeds; "sat." is joint constraint satisfaction over test episodes. References
from Table 1: ShortestQueue $73.25 / 100% and RoundRobin $78.90 / 86% on
electronics; ShortestQueue $138.66 / 96% and RoundRobin $137.55 / 88% on bakery.

| Testbed | Cell | stochastic CPU ± CI | stoch. sat. | val.-satisfied | λ_T at cap | argmax CPU ± CI | argmax sat. |
|---|---|---|---|---|---|---|---|
| electronics | corrected | $78.10 ± 1.64 | 95.2% | 5/5 | 0/5 | $162.94 ± 75.56 | 0.0% |
| electronics | original signal | $82.12 ± 1.64 | 70.0% | 5/5 | 5/5 | $196.04 ± 42.87 | 0.0% |
| bakery | corrected | $141.07 ± 0.68 | 94.4% | 5/5 | 0/5 | $165.66 ± 46.86 | 54.4% |
| bakery | original signal | $139.61 ± 2.04 | 86.4% | 5/5 | 5/5 | $180.37 ± 40.63 | 13.2% |

\newpage

**Table 3.** Paired hierarchical bootstrap (10,000 resamples of seeds and
episodes) of the learned cells, evaluated stochastically, against each
comparator. Entries are the difference in means (learned cell minus comparator)
with 95% percentile interval; Δsat. is in percentage points. Asterisks mark
comparisons surviving the Holm adjustment within their family.

| Testbed and cell | Comparator | ΔCPU ($) | p | Δsat. (pp) | p |
|--------|-----------|------------------|-------|------------------|-------|
| electronics, corrected | ShortestQueue | +4.85 [+2.94, +6.88] | < 0.001* | −4.8 [−9.2, −1.2] | 0.005* |
| electronics, corrected | LeastUtilised | −2.37 [−4.11, −0.42] | 0.017 | −2.8 [−7.2, +2.0] | 0.24 |
| electronics, corrected | RoundRobin | −0.79 [−2.74, +1.37] | 0.43 | +9.2 [−0.8, +19.6] | 0.073 |
| electronics, corrected | UniformRandom | −1.81 [−3.96, +0.41] | 0.10 | +17.2 [+6.0, +29.2] | 0.002* |
| electronics, corrected | original signal | −4.01 [−5.87, −1.99] | < 0.001* | +25.2 [+12.0, +38.8] | < 0.001* |
| bakery, corrected | ShortestQueue | +2.41 [−1.69, +6.55] | 0.25 | −1.6 [−7.6, +5.2] | 0.64 |
| bakery, corrected | LeastUtilised | −4.56 [−9.45, +0.50] | 0.079 | +4.4 [−5.2, +14.8] | 0.40 |
| bakery, corrected | RoundRobin | +3.52 [−1.02, +8.06] | 0.13 | +6.4 [−2.4, +16.4] | 0.18 |
| bakery, corrected | UniformRandom | +0.02 [−3.93, +3.88] | 0.99 | +4.4 [−4.0, +14.0] | 0.35 |
| bakery, corrected | original signal | +1.46 [−1.79, +4.78] | 0.38 | +8.0 [0.0, +17.2] | 0.063 |
| electronics, symmetric | ShortestQueue | +8.65 [+5.86, +11.39] | < 0.001* | −25.2 [−34.0, −16.8] | < 0.001* |
| electronics, symmetric | LeastUtilised | +1.43 [−1.39, +4.14] | 0.32 | −23.2 [−32.4, −14.8] | < 0.001* |
| electronics, symmetric | RoundRobin | +3.01 [+0.17, +5.86] | 0.036 | −11.2 [−22.4, +0.8] | 0.073 |
| electronics, symmetric | UniformRandom | +1.99 [−1.05, +4.90] | 0.20 | −3.2 [−16.4, +10.4] | 0.65 |
| electronics, symmetric | corrected (hinged) | +3.81 [+0.92, +6.61] | 0.012 | −20.4 [−30.0, −11.2] | < 0.001* |
| electronics, symmetric | original signal | −0.21 [−3.22, +2.80] | 0.89 | +4.8 [−9.6, +19.2] | 0.54 |

\newpage

**Table 4.** Re-evaluation of the archived checkpoints. Q1 evaluates the
originally selected checkpoint in both modes; Q2 applies stochastic selection.
n = 4 for the PID variant because one seed's checkpoints were not archived.

| Run | n | Q1 argmax CPU ± CI | Q1 argmax sat. | Q1 stochastic CPU ± CI | Q1 stoch. sat. | Q2 stochastic CPU ± CI | Q2 sat. | Q2 val.-satisfied |
|---|---|---|---|---|---|---|---|---|
| One-sided, electronics | 5 | $112.71 ± 13.46 | 6% | $81.35 ± 2.25 | 71% | $81.59 ± 2.68 | 74% | 5/5 (originally 0/5) |
| PID, electronics | 4 | $106.75 ± 23.87 | 22% | $83.94 ± 9.92 | 58% | $80.83 ± 3.40 | 72% | 3/4 |
| One-sided, bakery | 5 | $149.30 ± 27.73 | 69% | $140.56 ± 3.15 | 91% | $139.60 ± 2.49 | 80% | 5/5 |
| PID, bakery | 4 | $141.57 ± 22.71 | 74% | $140.46 ± 1.60 | 84% | $139.94 ± 3.72 | 83% | 4/4 |

\newpage

**Table 5.** Symmetric, reward-scaled cell, electronics, per seed. λ_T statistics
are over the 3,352 training episodes; the penalty ratio is the magnitude of the
episode penalty relative to the base return on the test episodes at the
training-mean multipliers. CIs are 95% t-intervals across seeds.

| Seed | Selected checkpoint | Stochastic CPU | Stoch. TP | Stoch. joint sat. | Argmax CPU | Argmax TP | λ_T mean / max | Penalty ratio |
|------|-----------|----------|--------|--------|----------|--------|-------------|--------|
| 42 | 601K | $81.89 | 54.6 | 80% | $107.78 | 43.4 | 0.09 / 0.73 | 0.12 |
| 7 | 300K | $84.87 | 52.2 | 78% | $106.10 | 43.5 | 0.06 / 0.50 | 0.08 |
| 2024 | 200K | $80.92 | 53.9 | 76% | $135.70 | 34.4 | 0.06 / 0.46 | 0.09 |
| 123 | 300K | $77.80 | 57.0 | 72% | $92.76 | 48.1 | 0.13 / 1.48 | 0.14 |
| 999 | 1,103K | $84.05 | 52.6 | 68% | $109.88 | 39.6 | 0.07 / 0.55 | 0.07 |
| mean ± CI | | $81.91 ± 3.47 | 54.1 ± 2.4 | 74.8% | $110.45 ± 19.40 | 41.8 ± 6.3 | | |

\newpage

**Table B1.** Mechanism ablation, electronics, 400K steps, 5 seeds per cell,
greedy selection. Reference: ShortestQueue $73.25 per unit at throughput 61.4.

| Cell | argmax CPU ± CI | argmax joint sat. | stochastic CPU ± CI | stochastic joint sat. | λ_T at cap |
|---|---|---|---|---|---|
| shaped + cumulative-rate (original) | $152.22 ± 84.67 | 0.8% | $82.37 ± 2.80 | 63.6% | 5/5 |
| shaped + episode | $122.26 ± 13.46 | 0.0% | $78.50 ± 1.27 | 94.4% | 0/5 |
| cost + cumulative-rate | $155.18 ± 88.17 | 0.0% | $80.44 ± 1.01 | 75.2% | 5/5 |
| cost + episode (the Lagrangian of Section 4.3) | $155.35 ± 80.54 | 0.0% | $78.73 ± 1.57 | 93.6% | 0/5 |

\newpage

**Table B2.** Action-distribution statistics of the selected checkpoints, ranges
or means across five seeds. Uniform reference: normalised entropy 1.00; effective
routes 12 (electronics) or 4 (bakery); top-1 probability 0.083 or 0.25; cross-seed
greedy agreement 8.3% or 25%. The last three columns are computed on 2,395 common
states; the gap is a mean with the per-seed range in parentheses.

| Testbed | Cell | norm. entropy | eff. routes | top-1 prob. | top-1 − top-2 gap | greedy agreement | flip rate (σ = 0.1) |
|--------|--------|--------|----------|--------|-------------|--------|--------|
| electronics | corrected | 0.94–0.97 | 10.4–11.3 of 12 | 0.13–0.18 | 0.021 (0.008–0.029) | 18.1% | 25.5% |
| electronics | original signal | 0.96–0.98 | 10.9–11.5 of 12 | 0.13–0.18 | 0.019 (0.014–0.026) | 9.2% | 24.3% |
| bakery | corrected | 0.91–0.98 | 3.5–3.9 of 4 | 0.31–0.44 | 0.154 (0.021–0.281) | 39.4% | 11.4% |
| bakery | original signal | 0.95–0.99 | 3.7–3.9 of 4 | 0.29–0.40 | 0.113 (0.027–0.214) | 33.4% | 15.3% |
| electronics | symmetric, reward-scaled | 0.63–0.83 | 4.8–7.8 of 12 | 0.28–0.48 | 0.179 (0.070–0.287) | 9.7% | 10.2% |

\newpage

**Table B3.** Joint constraint satisfaction on the 50 test episodes at alternative
thresholds, without retraining. SQ ShortestQueue, LU LeastUtilised, RR
RoundRobin, Random UniformRandom; "corrected", "original" and "symmetric" are the
learned cells evaluated stochastically and pooled over five seeds (the symmetric
cell was run on electronics only). The first row of each testbed is the protocol
threshold.

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

\newpage

## Figure captions

**Figure 1.** Throughput multiplier λ_T over training, five seeds per cell, both testbeds, logarithmic scale. Upper traces: the cumulative-rate signal of Eq. (4). Lower traces: the episode-level slack of Eqs. (8) and (9). (file: fig_lambda_T.tif)

**Figure 2.** The symmetric, reward-scaled cell during training, five seeds. (a) λ_T per episode on a linear scale; the dashed line is the shadow-price estimate of 0.1 from which the initial value was set. (b) Throughput slack TP − T_min per training episode, 50-episode moving average; the dashed line is the mean slack of uniform-random routing on the test episodes. (file: fig_symmetric.tif)
