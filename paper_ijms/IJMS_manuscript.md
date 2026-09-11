---
title: "Objective, constraint and evaluation alignment in constrained reinforcement learning for multi-server flow-shop routing"
---

Khaled R. Alrashdan

Department of Manufacturing Engineering Technology, College of Technological
Studies, Public Authority for Applied Education and Training (PAAET), Kuwait

kr.alrashdan@paaet.edu.kw · ORCID 0000-0001-6304-9061

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
in production systems. Two recent systematic reviews [@mayerhoff2026slr;
@schneider2026role] report the same weaknesses across roughly two hundred
studies: validation in simulation only, dispatching-rule baselines tuned far less
attentively than the learned agent, and no standardised benchmarks. Work
answering that critique concentrates on the baselines and the protocol. This
paper argues that a second class of problem sits upstream of both, in the
instrumentation of the constrained formulation itself, and can produce a reported
result that is an artefact of the pipeline rather than a property of the method.

A constrained learning pipeline contains five objects that are easy to conflate:
the constraint written in the formulation, normally an expectation over an
episode; the surrogate the training loop penalises, which must be expressed per
step for a policy-gradient method; the scale of the multipliers weighting that
surrogate against the objective; the mode in which the trained stochastic policy
is evaluated; and the metric finally reported, in this literature almost always a
ratio such as cost per unit. Each link is defensible in isolation. When two
disagree, aggregate performance figures do not reveal which is at fault, and the
natural reading of a poor result is that the method does not work.

This study measures three such breaks using Lagrangian Proximal Policy
Optimization (PPO) [@schulman2017ppo] on two multi-server flow-shop testbeds,
where the answer can be checked against tuned dispatching rules. The vehicle is
the author's own earlier study, run under a protocol committed in advance, which
reported that no seed satisfied throughput and utilisation constraints on a
12-route electronics testbed and that the throughput multiplier saturated in
every run. That study is referred to below as the initial study; its protocol,
archived checkpoints and result files are in the companion repository
[@flexflowsimcppo] and are the objects re-examined here.

The first break is between the stated constraint and the training surrogate: the
constraint is on an episode total, but the implemented signal penalised the
shortfall of a cumulative average rate at every step, which in a flow shop whose
flow time is a sizeable fraction of the shift sits below its floor for a long
prefix of every episode whatever the policy does. Under a monotone one-sided dual
update this grows without bound, and an oracle dispatching rule exceeds the
saturation threshold by a factor of two while satisfying the true constraint with
a 14–23% margin (Section 4.2). The second is between the policy trained and the
policy evaluated: the learned distributions are close to uniform, so their greedy
action turns on logit differences that neither replicate across seeds nor survive
small perturbations, and evaluating the archived checkpoints by sampling reverses
the original verdict without retraining (Sections 6.3 and 6.5).

The third break, between the objective optimised and the metric reported,
survives correction of the other two. With both artefacts removed the agent
satisfies the constraint criterion on every seed yet is indistinguishable from
round-robin routing on cost per unit, because in every one-sided configuration
studied the constraint terms exceed the cost term by two to four orders of
magnitude and cannot be released (Section 6.4). A further cell, specified in a
protocol amendment before its runs, gives the multipliers the scale of the reward
and a symmetric update; the dual then behaves as a dual and the policy becomes
state-dependent, but it moves to the throughput floor exactly as a total-cost
objective directs (Section 6.7). The mechanism is measurable across the seven
load-spreading policies of this study: total episode cost spans 4.6% while
throughput spans 15.5%, so cost per unit falls with throughput and an agent
minimising total cost has no reason to buy throughput above its floor.

The scope of these results should be stated plainly. The first two breaks are
properties of the instrumentation, not of constrained RL. Correcting them does
not make the method competitive here, nor deterministic deployment feasible. What
the study offers is a measured account of where a constrained pipeline comes
apart, the diagnostics that expose each break, and evidence that the residual
result is driven by the formulation rather than by any failure of the optimiser
to pursue it.

## 2. Related work

### 2.1 Reinforcement learning and dispatching rules in production control

The two reviews cited above map a field that has shifted from value-based methods
to policy gradients, with PPO dominant, and that repeatedly reports strong
performance against weak baselines and ambiguous performance against well-tuned
heuristics. Recent flow-shop work has converged on graph-based architectures
trained with PPO [@li2025evolutionary; @liu2025gat; @shen2026transformer;
@wang2025end]. An alternative to soft penalties is action masking
[@ali2023masked; @tang2020mask; @zhang2025dual], which presupposes infeasible
actions and so does not apply to the fully feasible routing problem studied here;
a parallel line bypasses RL and evolves interpretable dispatching rules directly
[@ferreira2022dispatching; @huang2025evolving; @marques2025dynamic]. On
benchmarking itself, Doherty and colleagues [@doherty2025hype] reproduce five
landmark RL studies in optical resource allocation with properly tuned heuristic
baselines and find that simple heuristics consistently match or outperform the
published results. On the two testbeds used here, Alrashdan
[@alrashdan2026breakdowns] benchmarked dispatching rules, Thompson-sampling
bandits and unconstrained PPO under machine breakdowns, finding ShortestQueue
ahead of PPO on cost per unit by 38–153% depending on disruption level; that
study addressed neither constraints nor evaluation mode. Rinciog and Meyer
[@rinciog2021fabricatio] introduced FabricatioRL, the closest comparator to the
simulator used here.

### 2.2 Constrained MDPs, Lagrangian methods and evaluation practice

The constrained Markov decision process (CMDP) formulation is due to Altman
[@altman1999cmdp], who shows that a finite CMDP with K constraints under
discounted cost admits an optimal stationary policy requiring at most K
randomisations, and that optimal constrained policies generally require
randomisation or time-sharing between deterministic policies. Those results are
stated for a setting different from ours, which is finite-horizon and uses
function approximation; they are used here to motivate reporting stochastic as
well as greedy evaluation, not to explain the learned policies. The modern
policy-gradient pipeline begins with Constrained Policy Optimization
[@achiam2017cpo]; Reward Constrained Policy Optimization [@tessler2019rcpo]
introduces the multi-timescale Lagrangian approach used here, and Paternain and
colleagues [@paternain2019duality] prove a zero duality gap underwriting
primal-dual methods despite the non-convexity of policy optimisation. Stooke and
colleagues [@stooke2020pid] are the closest methodological precedent: the
standard Lagrangian update behaves as integral control on the violation signal,
producing oscillation and overshoot, which a PID controller on the multiplier
damps.

That literature analyses dual dynamics given a constraint signal, and has little
to say about whether the signal fed to the dual update is the constraint the
paper claims to impose, which is the gap this study occupies. On the evaluation
side, Henderson and colleagues [@henderson2018matters] showed that reported
deep-RL results are sensitive to protocol choices that are rarely stated, and
Agarwal and colleagues [@agarwal2021precipice] that point estimates over a
handful of runs routinely misstate both the level and the uncertainty of
performance. Whether a stochastic policy is evaluated by sampling or by its mode
is one such choice, usually left to a library default; for a near-uniform policy
the two modes measure different objects, and this paper is an extended example of
the consequence.

## 3. Problem formulation and testbeds

### 3.1 Flow-shop model

The simulator represents a flow shop of N stages, stage s holding m_s parallel
servers differing in service-time distribution and cost rate, with Poisson
arrivals and exponential, server-specific service times. Cost accrues at
per-server processing and idle rates and a per-job waiting rate; an episode is
one shift, H = 480 one-minute steps. Actions are complete downstream routes
rather than per-stage assignments, tuples a = (a₁, …, a_N) from a space of size
∏_s m_s. The observation holds queue lengths, in-service indicators and
accumulated cost signals, but not the dual multipliers.

The bakery testbed has two stages of two servers, giving 4 routes, each pairing a
fast expensive machine with a slow cheap one, service times calibrated to the
BK50 subset of a bakery dataset [@babor2022bakery]. The electronics testbed has
three stages of 2, 3 and 2 servers, giving 12 routes, with asymmetric capacities
and one over-provisioned station; machine specifications are in
[@flexflowsimcppo] and [@alrashdan2026breakdowns], whose breakdown extensions are
unused here. The constrained servers F_fast are the fast machines of the first
two stages, with throughput floor T_min = 18 (bakery) or T_min = 50 (electronics)
and utilisation floor U_min = 0.50. The over-provisioned station is excluded
because LeastUtilised itself reaches only 42% utilisation there, so a 0.50 floor
would be infeasible for any policy.

### 3.2 Reward and the conservative routing attractor

With c_t the step-t cost summed over servers and C(τ) = Σ_t c_t the episode total,
the implemented environment reward is Eq. (1),

$$r_t = \kappa\left(-\,w_c\,\frac{c_t}{C_{\mathrm{norm}}} + w_t\,\frac{\dot n_t\,\Delta t}{N_{\mathrm{norm}}} - w_w\,\frac{W_t\,\Delta t}{W_{\mathrm{norm}}}\right)\qquad(1)$$

with weights (w_c, w_t, w_w) summing to one, κ = 10, Δt = 1 min, W_t the work in
process and normalisation constants (C_norm, N_norm, W_norm) = (2740, 20, 3220)
on bakery and (4230, 55, 5100) on electronics, typical episode totals making each
term of order one per episode. Here ṅ_t is the cumulative average rate d_t / t,
not an instantaneous one, which matters in Section 4.2. Under the cost-only
weights (1, 0, 0) the episode return is −κ C(τ)/C_norm, so the objective of
Eq. (2) is optimised without approximation. For w_c near unity PPO idles
expensive servers, as that objective directs: cost falls, throughput falls
further, and cost per unit rises above every load-spreading dispatching rule. A
sweep of w_c to 0.5 in the initial study did not escape this attractor;
constraining the problem is the natural response.

### 3.3 Constrained formulation and three constraint objects

With TP(τ) the episode departures and u_i(τ) server i's realised utilisation, the
agent is posed Eq. (2),

$$\max_{\pi}\; \mathbb{E}_\pi\!\left[-C(\tau)\right] \quad \text{s.t.}\quad \mathbb{E}_\pi[\mathrm{TP}(\tau)] \ge T_{\min},\quad \mathbb{E}_\pi[u_i(\tau)] \ge U_{\min}\;\; \forall\, i \in F_{\mathrm{fast}}\qquad(2)$$

Three distinct constraint objects appear. The first is Eq. (2) itself, the problem
as stated. The second is the training surrogate: the primal update penalises the
exact per-episode Lagrangian of Eq. (2), but the dual update reads a hinged
shortfall, max(0, T_min − TP(τ)), not the signed slack, so the multipliers track
violation frequency and depth, not expected constraint value. The third is the
selection and reporting criterion: a checkpoint qualifies if each constraint
holds on at least 16 of 20 validation episodes, that is, under chance constraints
P(TP(τ) ≥ T_min) ≥ 0.8 and P(min_i u_i(τ) ≥ U_min) ≥ 0.8, while results report
the fraction of test episodes satisfying both, stricter again; for per-episode
distributions as nearly symmetric as those observed here, both chance constraints
are stricter than Eq. (2). Section 6.7 exhibits a configuration satisfying
Eq. (2) and both chance constraints yet failing the joint criterion. The initial
study treated the three as interchangeable; Section 4.2 shows that what it
trained on was a fourth object coinciding with none.

### 3.4 The objective is not the reported metric

The headline metric is cost per unit, the per-episode ratio C(τ)/TP(τ), whereas
Eq. (2) minimises expected total cost subject to a throughput floor; here the two
diverge in a direction that matters. Interpolating between the mean cost and
throughput of CostMinimising and ShortestQueue on bakery gives a marginal cost of
about $40 per additional unit against an average of about $139, so cost per unit
falls with throughput across the operating range and a total-cost minimiser has
every incentive to sit at the floor, where the interpolated $151 is worse than
ShortestQueue's $138.66. This two-point argument is motivation only; Section 6.7
measures it directly across the study's seven load-spreading policies: the four
rules of Table 1 that spread work across routes and the three learned cells, but
not CostMinimising or FastServerFirst, which concentrate it.

Any apparent success on cost per unit must therefore come from elsewhere; in the
initial study the candidate is the selection rule: requiring 16 of 20 validation
episodes to satisfy TP ≥ 18, at a per-episode standard deviation of about 2.0
units, implies a mean throughput near 19.7, an effective floor above the nominal
18 though less than one standard deviation above it.

## 4. Constrained method and its implementation

### 4.1 Lagrangian PPO

PPO is the inner loop; a Gymnasium wrapper augments the per-step reward with
penalties, as in Eq. (3),

$$\tilde r_t = r_t - \lambda_U\, g_{U,t} - \lambda_T\, g_{T,t}\qquad(3)$$

where r_t is the environment reward of Eq. (1), not the bare cost term. The
initial study posed single-objective cost minimisation, yet every constrained
variant kept the shaping weights (0.8, 0.1, 0.1) active. An ablation isolating
them (Appendix B, Table B1) finds no resolvable effect under the Section 6.1
bootstrap, but the discrepancy is the first break in the chain.

### 4.2 The cumulative-rate slack and why it saturates any multiplier

Dual variables are updated from a slack signal g, initially Eq. (4), at every
step after a 100-step warm-up,

$$g_{T,t} = \max\!\left(0,\; \frac{T_{\min}}{H} - \frac{d_t}{t}\right)\qquad(4)$$

with d_t the cumulative departures to step t. This is not the constraint of
Eq. (2): where flow time is a sizeable fraction of the horizon, the filling line
holds the cumulative rate below the floor for a long prefix of every episode
whatever the policy does. Under ShortestQueue over the 50 test episodes it first
reaches the floor at a median step of 184 (bakery, range 100 to 434) and 194
(electronics, 101 to 399), staying below on 45% and 31% of post-warm-up steps. A
policy achieving exactly TP = 50 on electronics meets the constraint but does not
cross until t ≈ 480, never satisfying the training signal within the horizon. The
utilisation slack carried the same bias in milder form.

The one-sided dual update of Eq. (5), applied once per episode with the
post-warm-up mean,

$$\lambda \leftarrow \operatorname{clip}\!\left(\lambda + \eta\,\operatorname{mean}_t\, g_t,\; 0,\; \lambda_{\max}\right)\qquad(5)$$

is a violation-driven ratchet rather than a dual ascent, turning that bias into a
structural outcome. Over the same episodes, the mean per-step g_T under
ShortestQueue is 0.0031 (bakery) and 0.0054 (electronics); at η_T = 4000 that is
12.5 and 21.6 per episode against the 6.30 and 5.91 needed to carry λ_T from 200
to its cap of 20,000 within the 3,142 and 3,352 training episodes, so any mean
slack above 0.0016 per step saturates the multiplier within budget. The oracle's
exceeds it by a factor of two, capping at episode 1,581 of 3,142 on bakery and
915 of 3,352 on electronics; the trained policies saturate sooner, at increments
of 15–16 and 39–49, and at episodes 1,223–1,359 (39–43% of training) and 402–504
(12–15%). Saturation reflects the signal, not the policy; reporting it as
evidence about the method is a category error.

### 4.3 Episode-level slack and the corrected dual updates

The per-step signal becomes the signed pro-rata decomposition of the episode
Lagrangian, Eq. (6),

$$p_t = \lambda_T\left(\frac{T_{\min}}{H} - \delta_t\right) + \lambda_U \sum_{i\in F_{\mathrm{fast}}}\left(U_{\min} - b_{i,t}\right)\qquad(6)$$

with δ_t the departures in step t and b_{i,t} ∈ {0, 1} indicating whether server i
is busy; the augmented reward is r̃_t = r_t − p_t, with no warm-up. Over the
episode, Eq. (6) telescopes to Eq. (7),

$$\sum_{t=0}^{H-1} p_t = \lambda_T\left(T_{\min} - \mathrm{TP}(\tau)\right) + \lambda_U\, H \sum_{i\in F_{\mathrm{fast}}}\left(U_{\min} - \bar u_i(\tau)\right)\qquad(7)$$

the Lagrangian penalty for the expectation constraint of Eq. (2), in the same
episode-level quantities that validation and test evaluate, and free of
fill-phase bias. Multipliers update once per episode from the hinged
episode-level slack of Eqs. (8) and (9),

$$\lambda_T \leftarrow \operatorname{clip}\!\left(\lambda_T + \eta_T\,\frac{\max\!\left(0,\; T_{\min} - \mathrm{TP}(\tau)\right)}{H},\; 0,\; \lambda_{T,\max}\right)\qquad(8)$$

$$\lambda_U \leftarrow \operatorname{clip}\!\left(\lambda_U + \eta_U \sum_{i\in F_{\mathrm{fast}}} \max\!\left(0,\; U_{\min} - \bar u_i(\tau)\right),\; 0,\; \lambda_{U,\max}\right)\qquad(9)$$

Dividing by H keeps the throughput slack in the per-step rate units of Eq. (4),
so η_T, λ_T's initial value and cap carry over unchanged. The ratchet advances
only on violating episodes, so saturation is no longer guaranteed for an oracle
(Section 6.1).

Retaining the hinge keeps the update monotone: Eqs. (8) and (9) remove the
guaranteed divergence of Eq. (4), not the monotonicity, and under a stochastic
policy violating on a persistent fraction of episodes the multiplier grows at a
rate set by violation frequency, not expected slack. A symmetric variant uses the
signed slack (T_min − TP(τ))/H in Eq. (8) and, in Eq. (9), the summed shortfall
when any constrained server violates and the binding server's negative margin
otherwise, letting a multiplier fall back toward zero. Section 6.4 shows why
multiplier scale matters as much as the hinge; Section 6.7 reports that variant
at full budget.

### 4.4 Evaluation of a stochastic policy

The initial protocol evaluated policies greedily, but PPO optimises a stochastic
policy, and a constrained optimum may itself be randomised (Section 2.2); more
immediately, a near-uniform distribution's greedy action turns on small, unstable
logit differences, so evaluating it measures those rather than the policy. Both
modes are computed and reported throughout, checkpoints selected under the
reported mode; Section 6.5 measures the distance from uniform and the stability
of the greedy action to seed and perturbation. This is described as an
evaluation-mode mismatch, not an error: greedy evaluation is legitimate for a
deterministic deployment target (Section 7.3); the problem is reporting it for a
near-uniform policy without saying so.

## 5. Experimental protocol

### 5.1 Baselines

Six dispatching rules are evaluated on the 50 test episodes (Table 1). Four
are state-aware: ShortestQueue minimises summed instantaneous load over the
route; LeastUtilised, that load weighted by mean service time; CostMinimising,
summed processing cost per unit time, ignoring queues; and FastServerFirst
takes the fastest server at each stage. Two are stateless and resemble the
learned policies (Section 6.5): UniformRandom draws a route uniformly at each
arrival, RoundRobin cycles through routes in order. ShortestQueue, the primary
comparator, is cheapest per unit on electronics with both constraints
satisfied on every test episode; on bakery it is not separable from
RoundRobin, whose mean is $1.11 lower with heavily overlapping intervals.

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

F_fast and U_min come from a one-shot LeastUtilised run on each testbed,
anchoring the feasible region on the heuristic family the agent is compared
against. Even stateless routing then satisfies both constraints on 78–90% of
episodes (Table 1), so the constraints are not demanding; Section 6.6 reports
how tightening them changes the ordering.

### 5.2 Training, selection and statistical treatment

Five seeds committed in advance [42, 7, 2024, 123, 999]; 1.6M timesteps on
electronics and 1.5M on bakery; checkpoints every 100K. Validation uses seeds
[10000, 10020), test seeds [11000, 11050), both disjoint from training.
Selection takes the lowest mean cost per unit among checkpoints satisfying
each constraint on at least 16 of 20 validation episodes, or the lowest
overall if none qualifies, with the run marked a fallback. Cost per unit is
computed per episode and averaged. Confidence intervals across seeds are 95%
two-sided t-intervals; those in Table 1, across episodes, are not directly
comparable. PPO is the Stable-Baselines3 implementation [@raffin2021sb3] with
the initial study's hyperparameters, listed with the dual parameters in
Appendix A.

Three protocol documents in the repository govern the runs: the initial
study's protocol; an amendment, committed before any full-budget run here,
requiring both evaluation modes for every checkpoint and selection on the
stochastic validation episodes; and a second, committed before the runs of
Section 6.7, fixing that cell's multiplier scales a priori and listing the
comparisons and the wording each of three outcomes would receive. Two pipeline
defects are recorded as one deviation: unseeded action sampling, and joint
satisfaction aggregated across seeds as the smaller mean marginal rate rather
than the per-episode joint rate. Every stochastic figure here is therefore a
seeded re-evaluation of the selected checkpoints, every joint-satisfaction
figure a per-episode recomputation from archived records; Appendix A.5 gives
each correction's size, none of which changes the direction of any comparison.

Comparisons in Section 6.2 use a paired hierarchical bootstrap on the
per-episode test records, resampling training seeds within each learned cell
and test episodes jointly across the policies compared. This replaces
one-sample tests that treat baseline means as known constants and overstate
the evidence. Ten tests form the pre-specified family on each testbed,
adjusted by Holm at α = 0.05; the cell of Section 6.7 forms its own family of
twelve.

## 6. Results

### 6.1 Multiplier saturation is a property of the slack signal

A 2×2 design crossing base reward {shaped, cost-only} with slack signal
{cumulative-rate, episode-level}, five seeds per cell at a reduced 400K-step
budget on electronics, attributes saturation unambiguously (Appendix B,
Table B1). All ten cumulative-rate seeds end at the λ_T cap of 20,000, all ten
episode-level seeds between 908 and 2,833: no overlap, no seed-level exception.
Base reward has no resolvable effect in either arm: under the paired bootstrap of
Section 5.2, slack held fixed, the cost-only reward moves stochastic cost per
unit by +$1.93 [−0.42, +4.06] (cumulative-rate) and −$0.22 [−2.07, +1.47]
(episode-level), and joint satisfaction by −11.6 [−32.4, +8.0] and
+0.8 [−6.4, +8.0] percentage points. The Section 4.1 discrepancy is therefore
documentation, not a driver, though the first interval is wide enough that a
moderate saturated-signal effect cannot be excluded at five seeds.

The pattern holds at full budget on both testbeds (Figure 1). Under the
cumulative-rate signal the multiplier ramps monotonically to its cap on every
seed, at 0.19–0.24M steps on electronics and 0.59–0.65M on bakery, then is flat,
as Section 4.2 predicts. Under the episode-level signal it advances only on
violating episodes, ending between 2,525 and 7,858, never within a factor of 2.5
of the cap: it does not saturate, but neither has it converged — the monotonicity
of Section 4.3.

![](fig_lambda_T.png){width=16cm}\

**Figure 1.** Throughput multiplier λ_T over training, five seeds per cell, both
testbeds, logarithmic scale. Upper traces: the cumulative-rate signal of Eq. (4).
Lower traces: the episode-level slack of Eqs. (8) and (9).

### 6.2 Full-budget comparison

Two cells ran at full budget, five seeds each, selected on stochastic validation
and evaluated in both modes on the 50 test episodes: the original implementation
(shaped reward, cumulative-rate slack) as control, and the Section 4.3 correction
(cost-only reward, episode-level slack).

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

First, every seed now satisfies the validation criterion, including the
unmodified original signal, which passed on 0 of 5 electronics seeds under the
initial protocol: evaluation mode alone reverses the pass/fail verdict on the
harder testbed. Second, on electronics the correction buys 25 percentage points
of joint satisfaction and about $4 per unit, both surviving adjustment. Third, on
bakery it makes no measurable difference in either metric: the multiplier still
saturates on all five control seeds, so the Section 4.2 mechanism is present, but
the penalty does not damage the policy enough to show — a null result, not a
partial success.

Fourth and most consequential, the corrected agent does not separate from
stateless routing on cost per unit: −$0.79 [−2.74, +1.37] against RoundRobin,
−$1.81 [−3.96, +0.41] against UniformRandom, intervals admitting a few dollars
either way — a failure to detect rather than a demonstration of equality. It
leads UniformRandom by 17 satisfaction points, which survives adjustment, and
RoundRobin by 9, which does not; on bakery no comparison against any rule is
significant. ShortestQueue stays ahead of every learned policy, 6.2% cheaper per
unit on electronics at 100% satisfaction.

Table 2 also shows the seed dispersion: greedy evaluation gives electronics
intervals of ±43 to ±76, stochastic evaluation of the same checkpoints ±1.6. The
severe seed-to-seed instability the initial study reported was, to a first
approximation, greedy-action variance over near-uniform distributions.

### 6.3 Re-evaluation of the archived checkpoints

The initial study's checkpoints were re-evaluated without retraining — the
one-sided Lagrangian of Section 4.2 and a PID-Lagrangian comparator with
exponentially smoothed slack [@stooke2020pid] — on both testbeds and every
archived seed. Applied to the archived validation sweep, the original greedy rule
reproduces the selected checkpoint in 18 of 18 runs and the test figures to the
cent: this pipeline produced them. Table 4 gives the result.

**Table 4.** Re-evaluation of the archived checkpoints. Q1 evaluates the
originally selected checkpoint in both modes; Q2 applies stochastic selection.
n = 4 for the PID variant because one seed's checkpoints were not archived.

| Run | n | Q1 argmax CPU ± CI | Q1 argmax sat. | Q1 stochastic CPU ± CI | Q1 stoch. sat. | Q2 stochastic CPU ± CI | Q2 sat. | Q2 val.-satisfied |
|---|---|---|---|---|---|---|---|---|
| One-sided, electronics | 5 | $112.71 ± 13.46 | 6% | $81.35 ± 2.25 | 71% | $81.59 ± 2.68 | 74% | 5/5 (originally 0/5) |
| PID, electronics | 4 | $106.75 ± 23.87 | 22% | $83.94 ± 9.92 | 58% | $80.83 ± 3.40 | 72% | 3/4 |
| One-sided, bakery | 5 | $149.30 ± 27.73 | 69% | $140.56 ± 3.15 | 91% | $139.60 ± 2.49 | 80% | 5/5 |
| PID, bakery | 4 | $141.57 ± 22.71 | 74% | $140.46 ± 1.60 | 84% | $139.94 ± 3.72 | 83% | 4/4 |

The five electronics checkpoints recorded as failing systematically, at $113.31,
$112.08, $95.38, $118.35 and $124.44, score $83.78, $81.23, $78.68, $81.73 and
$81.33 as stochastic policies, with joint satisfaction of 64–88%; the files are
unchanged. Had the original protocol selected on stochastic validation, all five
would have been reported as satisfying the criterion at $81.59 ± 2.68, level with
RoundRobin and UniformRandom in Table 1. The negative result is thus reproduced
and reversed on its own artefacts.

The bakery rows show the complementary pattern: greedy and stochastic scores lie
within a few dollars on most seeds, and greedy selection already satisfied
validation on 4 of 5 seeds. Greedy extraction thus succeeds often on bakery and
almost never on electronics — not because no feasible deterministic policy exists
there, since ShortestQueue is one and a better one than anything learned, but
because the learned distributions there are near enough uniform that their mode
turns on unstable logit differences.

### 6.4 The scale of the constraint terms against the objective

The third link in the chain, the penalty's scale against the objective, needs
stating in the units the agent optimises: by Eq. (1) the per-episode base return
under cost-only weights is −κ C(τ)/C_norm, or −10.1 to −10.6 on both testbeds,
whereas the initial study compared the integrated penalty against a per-episode
cost in dollars, mixing units. Evaluated correctly, the penalty of Eq. (7) on the
corrected runs' test episodes is already 190–300 times larger in magnitude at the
initial multipliers (λ_T, λ_U) = (200, 5), and 1,200–10,000 times larger at those
reached by training's end; in the saturated control cell the ratio is of order
10⁴.

The penalty's sign is negative on almost every episode, since any load-spreading
policy over-satisfies both constraints: mean throughput 57.0 against a floor of
50 on electronics and 20.2 against 18 on bakery, fast-server utilisation
0.79–0.91 against a floor of 0.50. It therefore rewards surplus throughput and
fast-server busy time rather than acting as a penalty, and cannot shrink:
Eqs. (8) and (9) never decrease a multiplier, whereas the equilibrium multiplier
on a slack constraint is zero. Returns understate the imbalance seen by the
policy gradient: the cost term of Eq. (1) differs across the routing actions at a
decision by about 10⁻³ per step, whereas one departure moves the per-step term of
Eq. (6) by λ_T, between 200 and 7,858.

In every one-sided configuration, original and corrected alike, the objective
optimised was, to within a fraction of a percent, the throughput and fast-server
utilisation terms, with Eq. (2)'s cost term numerically irrelevant. This is
independent of the two artefacts, following from two inherited choices:
multiplier initial values and step sizes set in the units of the per-step signal
of Eq. (4), large against an episode return of order ten, and the hinge in the
dual update. Section 6.7 reports the experiment that removes both.

### 6.5 Structure of the learned policies

For the selected checkpoint of every full-budget run, the stochastic policy was
rolled out on ten test episodes and its action distribution recorded at every
decision, together with three measures of how much the greedy action means,
computed on a common set of 2,395 states visited under uniform-random routing
(Appendix B.1 and Table B2).

The hinged cells learn near-uniform routers with a tilt toward the constrained
servers. On electronics the most probable route at a typical decision carries
13–18% of the mass against 8.3% for uniform, and the corrected policies load the
two constrained fast servers to 0.82 and 0.79 mean utilisation against 0.65 and
0.69 under uniform-random routing. That is the response invited by the objective
of Section 6.4, in which surplus fast-server busy time is rewarded, and it buys
constraint reliability without buying cost efficiency.

The greedy diagnostics account for the wide intervals in Table 2. On electronics
the top action leads the runner-up by two percentage points of probability, two
seeds' greedy policies agree on 18% of states in the corrected cell against 8.3%
by chance, and logit noise of standard deviation 0.1 changes the greedy action at
a quarter of visited states. Greedy evaluation therefore selects a route
determined by differences that neither replicate across seeds nor survive small
perturbations, and it has no reason to be feasible. Appendix B.1 reports the
bakery figures, where four routes make the gaps larger and greedy extraction
feasible far more often, and two robustness checks on the stochastic figures.

### 6.6 Sensitivity to the constraint thresholds

The thresholds were set from a one-shot LeastUtilised run, and Table 1 shows that
at those values stateless routing satisfies both constraints on most episodes.
Joint satisfaction was therefore recomputed from the archived per-episode records
at stricter thresholds, without retraining, which re-scores policies trained for
the original thresholds rather than testing policies trained for the stricter
ones (Appendix B.2 and Table B3).

The two floors separate the policies in opposite directions. Raising the
utilisation floor rewards the tilt of Section 6.5: at U_min = 0.70 on
electronics, ShortestQueue and LeastUtilised still satisfy on 96% of episodes,
uniform-random routing on 16%, and the corrected agent on 71%, recovering roughly
two-thirds of the gap between stateless routing and the state-aware rules.
Raising the throughput floor exposes what the agent never learned: at T_min = 58,
where ShortestQueue still satisfies on 86% of episodes, the corrected agent falls
to 46%, closer to RoundRobin at 30% than to ShortestQueue. Meeting a demanding
throughput floor requires the queue-sensitive routing ShortestQueue performs and
the learned policies do not, and the constraint they were trained against did not
require it.

### 6.7 A functioning dual, and what it reveals about the objective

The final cell removes the two features of Section 6.4 that held the multipliers
from equilibrium: the hinge, replaced by the signed update of Section 4.3, and
the inherited scale, replaced by multipliers initialised, stepped and capped on
the reward's scale (λ_T: 0.1, step 2.0, cap 10; λ_U: 0.002, step 0.01, cap 0.2).
Two independent, agreeing arguments fixed these before any run: the Section 3.4
interpolation's shadow prices, about 0.1 reward units per throughput unit and
about 0.002 per fast-server busy-step, and the factor of roughly 2,400 dividing
the inherited multipliers to bring a typical episode penalty to a tenth of the
return. Five seeds, full budget, electronics.

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

![](fig_symmetric.png){width=16cm}\

**Figure 2.** The symmetric, reward-scaled cell during training, five seeds.
(a) λ_T per episode on a linear scale; the dashed line is the shadow-price
estimate of 0.1 from which the initial value was set. (b) Throughput slack
TP − T_min per training episode, 50-episode moving average; the dashed line is
the mean slack of uniform-random routing on the test episodes.

The dual now behaves as a dual. λ_T falls to zero within a few episodes when the
constraint is slack, rises when violated, oscillates between 0 and about 0.5,
never nears its cap, and is positive on 60–72% of episodes; its per-seed training
mean, 0.06–0.13, brackets the a priori shadow-price estimate. The constraint
terms stay on the scale of the cost term, 7–14% of the return at the
training-mean multipliers, against 190 to 10,000 times it in the hinged cells.

Within the first 400K steps mean throughput per training episode falls from
random routing's 4.4 units above T_min to 1–3 units above it and stays there
(Figure 2b), with 24–39% of training episodes below T_min in every quarter of
every seed; the expectation constraint of Eq. (2) is thus met in training by one
to three units and at the selected checkpoints on test by 4.1. Of 95 checkpoints
validated across five seeds only 14 qualify under the 16-of-20 rule, though every
seed had at least one and none fell back. The selected checkpoints satisfy the
throughput constraint on 84.8% of test episodes and the utilisation constraint on
82.0%, both just above the 80% demanded, but the two jointly on only 74.8%, 20
points below the corrected cell and 25 below ShortestQueue. This is the
configuration Section 3.3 promised: Eq. (2) and both chance constraints
satisfied, the reported joint criterion failed.

On the reported metric it is no better: $81.91 ± 3.47, significantly above
ShortestQueue and indistinguishable from RoundRobin, UniformRandom, LeastUtilised
and the original signal (Table 3). Nor is it cheapest on the total cost it
minimises: $4,414 per episode against $4,342 (RoundRobin), $4,346
(UniformRandom), $4,445 (corrected cell), $4,493 (ShortestQueue) and $4,542
(LeastUtilised). Under the paired bootstrap of Section 5.2 on total cost it sits
$72 above RoundRobin [−$42, +$183] and $67 above UniformRandom [−$53, +$187],
neither resolvable, and $129 below LeastUtilised [−$229, −$30], which is.
Section 3.4's direct measurement shows why: across the seven load-spreading
policies mean episode cost spans 4.6%, $4,342 to $4,542, because the arrival
process fixes the work and routing changes only where it runs and how long jobs
wait; throughput spans 15.5%, 53.2 to 61.4 units. An objective nearly flat in
routing and indifferent to throughput above the floor teaches a policy gradient
little, and what it teaches, that such throughput is not worth paying for, is the
opposite of what cost per unit rewards.

Its action distributions are unlike any other cell's: normalised entropy
0.63–0.83 against 0.94–0.97 for the hinged cells, 4.8–7.8 effective routes
against 10.4–11.3, mean top-1 probability 0.28–0.48 against 0.13–0.18, yet
marginal route frequencies over an episode near-uniform at 0.89–0.96. The
concentration is therefore state-dependent as the near-uniform cells were not,
and randomised in the sense of Section 2.2 rather than merely flat: its greedy
action is a coherent conservative router at throughput 41.8, $110 per unit,
feasible on 10% of episodes, while the stochastic policy behind it reaches 54
units at $82; the randomisation carries a quarter of the throughput. Greedy
extraction fails for the reason the CMDP literature predicts, not because the
logits are flat: logit noise of 0.1 now moves the greedy action at 10% of states
rather than 25%.

A functioning dual on the stated CMDP therefore produces no policy competitive
with ShortestQueue on either metric. Whether it approaches the total-cost optimum
cannot be established here: its total cost is not below that of stateless
routing, so the objective was pursued but not demonstrably solved.

## 7. Discussion

### 7.1 Where the chain broke, and what each break cost

The three breaks differ in kind. The first, a per-step surrogate for an
episode-level constraint, is a modelling error of generalisable form, defensible
in isolation: wherever transit time is a sizeable fraction of the horizon, a
cumulative-average translation is biased throughout the fill phase by a small
amount, of order 10⁻³ per step, that a symmetric dual update would release once
the line filled and a monotone one-sided update integrates without bound. Its
consequence is problem-dependent: the multiplier saturates on every control seed,
but the policy is measurably damaged only on the 12-route instance. The second,
greedy evaluation of a near-uniform policy, reverses the reported verdict. The
third, the scale of the multipliers, survives correction of the other two and
explains the residual result. It is also the least visible: nothing in the
original pipeline would have surfaced it, whereas a single ratio of penalty to
base return exposes it at once.

### 7.2 What the method learned, and what it was asked to learn

The penalty did not overwhelm the agent's objective: for practical purposes it
was that objective. With the multipliers two to four orders of magnitude above
the cost term, never released, and both constraints slack under any
load-spreading policy, the augmented return rewarded surplus throughput and
fast-server busy time, registering cost only in the third or fourth significant
figure; every hinged cell responded with a near-uniform router tilted toward the
fast servers.

The symmetric cell moves the fault upstream: a reward-scaled dual free to fall
does make PPO leave the uniform policy, and the state-dependent router it learns
moves to the throughput floor as a total-cost objective directs. "Failed to
learn" therefore needs qualification twice over: the hinged cells were never
seriously asked to minimise cost, and the symmetric cell was asked to minimise a
quantity that is not the reported metric, and did so.

Three consequences follow. If cost per unit is the criterion, the objective
should be aligned with it, by a ratio objective or a throughput term weighted by
that metric, not by a constraint; a per-departure bonus of that kind narrowed but
did not close the gap to ShortestQueue for unconstrained PPO on these testbeds
[@alrashdan2026breakdowns]. A total-cost CMDP is legitimate but should then be
judged on total cost. And the constraint enforced should be the constraint
checked: a joint per-episode criterion needs a per-episode formulation, a
violation-indicator penalty or a conditional-value-at-risk constraint on the
joint event, not an expectation constraint whose satisfaction still leaves a
quarter of episodes jointly infeasible. A functioning dual is necessary and not
sufficient: here it changed what was learned without making it competitive, and
tuned dispatching rules remain the baseline to beat at this problem size and
structure.

### 7.3 Deterministic deployment

Feasible deterministic policies exist on both testbeds, ShortestQueue among them,
but none was extracted from a learned agent on the harder testbed: greedy
extraction from the near-uniform hinged distributions yields a route fixed by
unstable logit differences, and from the concentrated distributions of the
symmetric cell a coherent but conservative router, feasible on 10% of episodes,
because the randomisation carries a quarter of the throughput. Where auditability
or operator trust rules out a stochastic controller, these policies are not
deployable, and the corrections do not change that.

### 7.4 Reporting practice

Four items follow for simulation studies of learned production control. A report
that a constrained method failed its constraints should let the reader check that
the constraint optimised is the one stated: the augmented per-step reward, the
dual update, and the slack signal's units and measured value under a known-good
reference policy. The ratio of the constraint terms to the objective should be
reported with them, at initialisation and at the end of training. Where a
stochastic policy is evaluated, the mode should be stated, both modes reported,
and the learned distribution's distance from uniform measured. And uniform-random
and round-robin routing should be among the dispatching baselines, evaluated
under the same protocol with their uncertainty propagated; without that control
this study would have read a near-random policy meeting undemanding constraints
as a recovery.

## 8. Limitations

The two instances differ in action-space size, stage count, capacity ratios
and reward scale, so while the failure is attributable to the directly
manipulated implementation, the difference in difficulty between them is
attributable to no single structural feature, absent a controlled ablation.
Five seeds per cell is a small sample: stochastic-mode intervals are tight
(±0.7 to ±2.0 for the four cells of Table 2, ±3.5 for the symmetric cell), but
the bootstrap has only 126 distinct resamples of those five seed-level means
and is anti-conservative at this cluster count, leaving between-cell intervals
approximate and p-values between 0.01 and 0.05 no better than indicative. The
null comparisons against stateless routing are failures to detect rather than
evidence of equality: the electronics interval against RoundRobin spans −$2.74
to +$1.37 per unit.

Thresholds come from a one-shot LeastUtilised run that Table 1 shows stateless
routing meeting on 78–90% of episodes, so reliability is compared at
undemanding thresholds, and Section 6.6 tightens them only by re-scoring
policies trained for them. Each validation and test episode samples one action
trajectory, bounded in Section 6.5 at roughly a third of the between-seed
interval, and the validation draws were single and unseeded. Q2 re-selection
(Section 6.3) reuses the reporting episodes, so Q1 carries the principal
claim; and the amendment introducing stochastic selection, though
pre-committed, followed the reduced-budget ablation and is not independent of
the same study's data.

The symmetric cell tests a functioning dual in one configuration only:
multiplier scales fixed a priori, though after inspecting the earlier results,
electronics alone, and plain dual ascent, whose λ_T oscillates rather than
converging (Figure 2a); a damped or averaged dual might select different
checkpoints. The mechanism it exposes is measured across the study's seven
load-spreading policies and survives those choices; its numbers do not. All
results are from simulation, with no physical validation.

## 9. Conclusions

A reported failure of constrained reinforcement learning traces to three breaks
between the constraint stated and the metric reported. A per-step cumulative-rate
surrogate for an episode-level constraint capped the multiplier within budget for
an oracle dispatching rule and every trained policy. Greedy evaluation of
near-uniform policies measured a router selected by unstable logit differences;
correcting that alone reverses the verdict on the archived checkpoints without
retraining. With both corrected, every seed satisfies the constraint criterion,
yet the agent is not distinguishable from round-robin routing on cost per unit at
this sample size and is 6.6% more expensive than ShortestQueue, with multipliers
two to four orders of magnitude above the cost term and never released. A
reward-scaled, symmetric dual behaves as a dual: the policy becomes
state-dependent and moves to the throughput floor as a total-cost objective
directs, where it is no better on the reported metric than the stateless rules
and remains significantly worse than ShortestQueue.

The objective posed, the constraint enforced, the surrogate trained against, the
evaluation mode and the reported metric form a chain that must be coherent; here
it was not. Across the seven load-spreading policies, total cost spans 4.6% and
throughput 15.5%, so cost per unit falls with throughput: a total-cost CMDP is
the wrong instrument for a cost-per-unit objective however well its dual behaves.
Separating these claims required reporting the constraint actually implemented,
the penalty's scale against the objective, the evaluation mode used, and a
random-routing control; all four belong in routine practice.

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
Hitzmann [@babor2022bakery].

## References

## Appendix A. Reproducibility

**A.1 Software.** All experiments use the FlexFlowSim-CPPO simulator
[@flexflowsimcppo]. The testbeds are defined in `configs/bakery_bk50.json` and
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
percentiles; p is twice the smaller tail proportion at zero, so with 10,000
replicates its resolution limit is 2 × 10⁻⁴ and entries reported as < 0.001 are
at or below that limit. Resampling five seed-level means admits 126 distinct
multisets in the seed dimension, which makes the procedure anti-conservative at
this cluster count; p-values between 0.01 and 0.05 should be read accordingly.
The pre-specified family on each testbed is the corrected cell against the four
load-spreading rules of Table 1 and against the original signal, on cost per unit
and on joint satisfaction, giving ten tests adjusted by Holm's step-down
procedure at α = 0.05; the symmetric cell forms its own family of twelve.
Confidence intervals use the t-distribution with n − 1 degrees of freedom across
seeds and the normal approximation across episodes.

**A.6 Size of the recorded deviation.** Two pipeline defects are recorded in
`protocol_deviations.md` and corrected in every figure reported here (Section
5.2). Greedy cost per unit is unaffected by both and reproduces the pipeline's
values to the cent, which is the fidelity check on the re-evaluation. Greedy
joint satisfaction moves by at most 1.2 percentage points under the aggregation
correction. Between the unseeded pipeline draw and the seeded re-evaluation,
cell-level figures differ by at most $1.2 per unit and 5.6 percentage points of
joint satisfaction; no comparison changes direction, and the largest consequence
is that the slack correction of Section 6.2 is worth 19 rather than 25 percentage
points on the other draw.

**A.7 Hardware and runtime.** Training and evaluation ran on a two-vCPU Linux
container at approximately 800–1,000 environment steps per second. A 400K-step
segment takes 7–8 minutes and a full 1.6M-step run about 30 minutes plus 10–15
minutes of dual-mode validation. The complete matrix reported here, comprising 20
reduced-budget runs, 25 full-budget runs, 18 archival re-evaluations of 279
checkpoints, the baselines and all analyses, consumed approximately 21
compute-hours.

## Appendix B. Supplementary tables

**B.1 Structure of the learned policies (Section 6.5).** Normalised entropy of
the hinged electronics cells is 0.94–0.98 and the effective number of routes
10.4–11.5 of 12; the corrected policies load the two constrained fast servers to
0.82 and 0.79 mean utilisation against 0.81 and 0.90 under ShortestQueue. Two
seeds' greedy policies agree on 9% of states in the control cell against 18% in
the corrected cell and 8.3% by chance. On bakery, with four routes, the top-1
gaps are larger and cross-seed agreement higher, and greedy extraction is
feasible on 54% of episodes rather than none. Two robustness checks bound the
stochastic figures. Single-draw evaluation carries sampling variance of its own:
re-evaluating one electronics checkpoint five times with different sampling seeds
gives cost per unit 79.11–81.63 (SD 0.96) and joint satisfaction 88–96%
(SD 0.036), against a between-seed SD of 1.32 for the same cell, so roughly half
the between-seed variance the bootstrap resamples is single-draw noise; the
bootstrap of Table 3 resamples episodes as well as seeds. And the control cell is
if anything closer to uniform than the corrected cell, consistent with Section
6.1: a saturated penalty four orders of magnitude larger than the base reward
leaves the policy gradient with almost nothing to say about routing.

**B.2 Threshold sensitivity (Section 6.6).** Table B3 re-scores every policy at
alternative thresholds without retraining. Beyond the two patterns given in
Section 6.6, raising the throughput floor to T_min = 60 on electronics puts every
learned and stateless policy at or below 30% while ShortestQueue holds 74%, and
on bakery the orderings are compressed throughout because four routes leave less
room for a routing policy to differ from another.

**Table B1.** Mechanism ablation, electronics, 400K steps, 5 seeds per cell,
greedy selection. Reference: ShortestQueue $73.25 per unit at throughput 61.4.

| Cell | argmax CPU ± CI | argmax joint sat. | stochastic CPU ± CI | stochastic joint sat. | λ_T at cap |
|---|---|---|---|---|---|
| shaped + cumulative-rate (original) | $152.22 ± 84.67 | 0.8% | $82.37 ± 2.80 | 63.6% | 5/5 |
| shaped + episode | $122.26 ± 13.46 | 0.0% | $78.50 ± 1.27 | 94.4% | 0/5 |
| cost + cumulative-rate | $155.18 ± 88.17 | 0.0% | $80.44 ± 1.01 | 75.2% | 5/5 |
| cost + episode (the Lagrangian of Section 4.3) | $155.35 ± 80.54 | 0.0% | $78.73 ± 1.57 | 93.6% | 0/5 |

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
