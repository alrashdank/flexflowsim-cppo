---
title: "Objective, constraint and evaluation alignment in constrained reinforcement learning for multi-server flow-shop routing"
---

Khaled R. Alrashdan

Department of Manufacturing Engineering Technology, College of Technological
Studies, Public Authority for Applied Education and Training (PAAET), Kuwait

kr.alrashdan@paaet.edu.kw · ORCID: https://orcid.org/0000-0001-6304-9061

## Abstract

Constrained reinforcement learning is increasingly applied to production
routing, but a reported result rests on a chain of choices seldom reported
together: the constraint stated, the surrogate the training loop penalises, the
scale of the multipliers, the evaluation mode, and the metric reported. This
simulation study measures what happens when links in that chain disagree, using
Lagrangian proximal policy optimisation (PPO) on two capacity-limited flow-shop
testbeds with 4 and 12 routes. Implementing an episode-level throughput
constraint as a per-step penalty on the cumulative rate drives the multiplier to
its cap for every trained policy, and by the same arithmetic would do so for a
dispatching rule that satisfies the constraint with a 14–23% margin, because that
rate sits below its floor while the line fills: 10 of 10 seeds saturate with this
signal, 0 of 10 with an episode-level slack. Evaluating near-uniform stochastic
policies greedily measures a router selected by unstable logit differences; on
the 12-route testbed, sampling instead raises joint constraint satisfaction on
the archived checkpoints from 6% to 71% without retraining, and correcting the
slack signal raises it from 70% to 95% in new runs. The agent is nonetheless not
distinguishable from round-robin routing on cost per unit at five seeds, and is
6.6% more expensive than ShortestQueue. Giving the dual the scale of the reward
makes it behave as a dual and the policy becomes state-dependent, but its
training throughput falls towards the floor as a total-cost objective directs and
the checkpoints retained are no better on the reported metric than stateless
routing, because on a capacity-limited line total cost is nearly flat in routing
while cost per unit falls with throughput.

**Keywords:** discrete-event simulation, constrained reinforcement learning,
flow-shop routing, dispatching rules, benchmarking, production control

## 1. Introduction

Reinforcement learning (RL) is now routinely proposed for routing and sequencing
in production systems. Two recent systematic reviews [1,2] map a field validated almost entirely in simulation and
without standardised benchmarks, the larger of them across 196 studies, and a
reproduction study in a neighbouring resource-allocation problem
[3] found properly tuned heuristics matching or beating every one
of five landmark RL results. Work answering that critique concentrates on the
baselines and the protocol. This
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

This study measures three such breaks using Lagrangian proximal policy
optimisation (PPO) [4] on two multi-server flow-shop testbeds,
where the answer can be checked against tuned dispatching rules. The vehicle is
the author's own earlier study, run under a protocol committed in advance, which
reported that no seed satisfied throughput and utilisation constraints on a
12-route electronics testbed and that the throughput multiplier saturated in
every run. That study is referred to below as the initial study; its protocol,
archived checkpoints and result files are in the companion repository
[5] and are the objects re-examined here.

The first break is between the stated constraint and the training surrogate: the
constraint is on an episode total, but the implemented signal penalised the
shortfall of a cumulative average rate at every step, which in a flow shop whose
flow time is a sizeable fraction of the shift sits below its floor for a long
prefix of every episode whatever the policy does. Under a monotone one-sided dual
update this grows without bound, and a dispatching rule that satisfies the true
constraint with a 14–23% margin exceeds the saturation threshold by a factor of
two on the 4-route testbed and 3.7 on the 12-route one (Section 4.2). The second is between the policy trained and the
policy evaluated: the learned distributions are close to uniform, so their greedy
action turns on logit differences that neither replicate across seeds nor survive
small perturbations, and evaluating the archived checkpoints by sampling reverses
the original verdict without retraining (Sections 6.3 and 6.5).

The third break, between the objective optimised and the metric reported,
survives correction of the other two. With both artefacts removed the agent
satisfies the constraint criterion on every seed yet is not distinguishable, at
five seeds, from round-robin routing on cost per unit, because in every one-sided configuration
studied the constraint terms exceed the cost term by two to four orders of
magnitude and cannot be released (Section 6.4). A further cell, specified in a
protocol amendment before its runs, gives the multipliers the scale of the reward
and a symmetric update; the dual then behaves as a dual and the policy becomes
state-dependent, but its training throughput falls towards the floor as a
total-cost objective directs, and the checkpoints the selection rule retains sit
at the throughput of random routing (Section 6.7). The mechanism is measurable
across the seven load-spreading policies of the 12-route testbed: total episode
cost spans 4.6% while throughput spans 15.5%, so an agent minimising total cost
has no reason to buy throughput above its floor.

The scope of these results should be stated plainly. The first two breaks are
properties of the instrumentation, not of constrained RL. Correcting them does
not make the method competitive here, nor deterministic deployment feasible. What
the study offers is a measured account of where a constrained pipeline comes
apart, the diagnostics that expose each break, and evidence that the residual
result follows the incentives of the formulation, while leaving open whether the
optimiser solved the problem it was given.

## 2. Related work

### 2.1 Reinforcement learning and dispatching rules in production control

The two reviews cited above map a field that has shifted from value-based methods
to policy gradients, with PPO dominant, and that repeatedly reports strong
performance against weak baselines and ambiguous performance against well-tuned
heuristics. Recent job-shop and flow-shop work has converged on graph- and
attention-based encoders trained with PPO [6,7,8,9]. An alternative to soft penalties is action
masking [10,11,12], which presupposes
infeasible actions and so does not apply to the fully feasible routing problem
studied here; a parallel line evolves interpretable dispatching rules directly
[13,14] or learns to select among existing
rules [15]. On the two testbeds used here, Alrashdan
[16] benchmarked dispatching rules, Thompson-sampling
bandits and unconstrained PPO under machine breakdowns, finding ShortestQueue
ahead of PPO on cost per unit by 38–153% depending on disruption level; that
study addressed neither constraints nor evaluation mode. Rinciog and Meyer
[17] introduced FabricatioRL, the closest comparator to the
simulator used here.

### 2.2 Constrained MDPs, Lagrangian methods and evaluation practice

The constrained Markov decision process (CMDP) formulation is due to Altman
[18], who shows that a finite CMDP with K constraints under
discounted cost admits an optimal stationary policy requiring at most K
randomisations, so that optimal constrained policies generally randomise or
time-share between deterministic policies. Those results are stated for a
setting different from ours, which is finite-horizon and uses function
approximation; here they motivate reporting stochastic as well as greedy
evaluation, not an explanation of the learned policies. The modern
policy-gradient pipeline begins with Constrained Policy Optimization
[19]; Reward Constrained Policy Optimization [20]
introduces the multi-timescale Lagrangian approach used here, and Paternain and
colleagues [21] prove a zero duality gap underwriting
primal-dual methods despite the non-convexity of policy optimisation. Stooke and
colleagues [22] are the closest methodological precedent: the
standard Lagrangian update behaves as integral control on the violation signal,
producing oscillation and overshoot, which a PID controller on the multiplier
damps.

That literature analyses dual dynamics given a constraint signal, and has little
to say about whether the signal fed to the dual update is the constraint the
paper claims to impose, which is the gap this study occupies. On the evaluation
side, Henderson and colleagues [23] showed that reported
deep-RL results are sensitive to protocol choices that are rarely stated, and
Agarwal and colleagues [24] that point estimates over a
handful of runs routinely misstate both the level and the uncertainty of
performance. Whether a stochastic policy is evaluated by sampling or by its mode
is one such choice, usually left to a library default; for a near-uniform policy
the two modes measure different objects.

## 3. Problem formulation and testbeds

### 3.1 Flow-shop model

The simulator represents a flow shop of N stages, stage s holding \(m_s\) parallel
servers differing in service-time distribution and cost rate, with Poisson
arrivals and truncated-normal, server-specific service times. Cost accrues at
per-server processing and idle rates and a per-job waiting rate; an episode is
one shift of \(H = 480\) minutes. A routing decision
is taken once per minute and assigns any job arriving in that minute its complete
downstream route, a tuple \(a = (a_1, \ldots, a_N)\) from a space of size
\(\prod_s m_s\); because the simulator advances to the first minute on reset, an
episode presents 479 decisions. The observation holds each server's queue
length, normalised by 50 and clipped to [0, 1], and its in-service indicator; it
carries neither the dual multipliers nor the time index, so the policy is
stationary over an episode. Appendix C, Table C1, lists the symbols.

The bakery testbed has two stages of two servers, giving 4 routes, each pairing a
fast expensive machine with a slow cheap one, service times calibrated to the
BK50 subset of a bakery dataset [25]. The electronics testbed has
three stages of 2, 3 and 2 servers, giving 12 routes, with asymmetric capacities
and one over-provisioned station. Table A1 gives the arrival, service and cost
parameters; both instances are defined in the configuration files of
[5] and were introduced in [16], whose
breakdown extensions are unused here. Both lines are capacity-limited: the
bakery baking stage can complete about 23 jobs per shift against about 50
arrivals, and the electronics soldering stage about 73 against about 80. The
constrained servers \(F_{\mathrm{fast}}\) are the fast machines of the first two
stages, with throughput floor \(T_{\min} = 18\) (bakery) or \(T_{\min} = 50\)
(electronics) and utilisation floor \(U_{\min} = 0.50\), all three fixed in the
protocol committed before the initial study; \(F_{\mathrm{fast}}\) and
\(U_{\min}\) were read from a one-shot LeastUtilised run (Section 5.1). The
over-provisioned station is excluded because LeastUtilised itself reaches only
42% utilisation there, so a 0.50 floor would be infeasible for any policy.

### 3.2 Reward and the conservative routing attractor

With \(c_t\) the step-t cost summed over servers and \(C(\tau) = \sum_t c_t\) the episode total,
the implemented environment reward is Eq. (1),

$$r_t = \kappa\left(-\,w_c\,\frac{c_t}{C_{\mathrm{norm}}} + w_t\,\frac{\dot n_t\,\Delta t}{N_{\mathrm{norm}}} - w_w\,\frac{W_t\,\Delta t}{W_{\mathrm{norm}}}\right)\qquad(1)$$

with weights \((w_c, w_t, w_w)\) summing to one, \(\kappa = 10\), \(\Delta t = 1\) min, \(W_t\) the work in
process and normalisation constants \((C_{\mathrm{norm}}, N_{\mathrm{norm}}, W_{\mathrm{norm}}) = (2740, 20, 3220)\)
on bakery and (4230, 55, 5100) on electronics, typical episode totals making each
term of order one per episode. Here \(\dot n_t\) is the cumulative average rate \(d_t/t\),
not an instantaneous one, which matters in Section 4.2. Under the cost-only
weights (1, 0, 0) the undiscounted episode return is
\(-\kappa\, C(\tau)/C_{\mathrm{norm}}\), proportional to total cost. PPO optimises
its discounted counterpart with \(\gamma = 0.99\) (Appendix A.2), which weights a
departure at step \(t\) by \(0.99^t\), 0.09 at step 240, so the objective actually
optimised is a horizon-weighted approximation to Eq. (2); the discount is held
fixed across every cell and is a link in the chain of Section 1 that this study
does not vary. For \(w_c\) near unity PPO idles
expensive servers, as that objective directs: cost falls, throughput falls
further, and cost per unit rises above every load-spreading dispatching rule. A
sweep of \(w_c\) to 0.5 in the initial study did not escape this attractor;
constraining the problem is the natural response.

### 3.3 Constrained formulation and three constraint objects

With \(\mathrm{TP}(\tau)\) the episode departures and \(u_i(\tau)\) server i's realised utilisation, the
agent is posed Eq. (2),

$$\max_{\pi}\; \mathbb{E}_\pi\!\left[-C(\tau)\right] \quad \text{s.t.}\quad \mathbb{E}_\pi[\mathrm{TP}(\tau)] \ge T_{\min},\quad \mathbb{E}_\pi[u_i(\tau)] \ge U_{\min}\;\; \forall\, i \in F_{\mathrm{fast}}\qquad(2)$$

Three distinct constraint objects appear. The first is Eq. (2) itself, the problem
as stated. The second is the training surrogate: the primal update penalises a
per-step decomposition of the episode Lagrangian (Section 4.3), but the dual update reads a hinged
shortfall, \(\max(0,\, T_{\min} - \mathrm{TP}(\tau))\), not the signed slack, so the multipliers track
violation frequency and depth, not expected constraint value. The third is the
selection and reporting criterion: a checkpoint qualifies if each constraint
holds on at least 16 of 20 validation episodes, that is, under chance constraints
\(P(\mathrm{TP}(\tau) \ge T_{\min}) \ge 0.8\) and \(P(\min_i u_i(\tau) \ge U_{\min}) \ge 0.8\), while results report
the fraction of test episodes satisfying both, stricter again; for per-episode
distributions as nearly symmetric as those observed here, both chance constraints
are stricter than Eq. (2). Section 6.7 exhibits a configuration satisfying
Eq. (2) and both chance constraints yet failing the joint criterion. The initial
study treated the three as interchangeable; Section 4.2 shows that what it
trained on was a fourth object coinciding with none.

### 3.4 The objective is not the reported metric

The headline metric is cost per unit, the per-episode ratio \(C(\tau)/\mathrm{TP}(\tau)\), whereas
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

The selection rule adds an effective floor of its own: requiring 16 of 20
validation episodes to satisfy \(\mathrm{TP} \ge 18\), at a per-episode standard
deviation of about 2.0 units, implies a mean throughput near 19.7, less than one
standard deviation above the nominal 18.

## 4. Constrained method and its implementation

### 4.1 Lagrangian PPO

PPO is the inner loop; a Gymnasium wrapper augments the per-step reward with
penalties, as in Eq. (3),

$$\tilde r_t = r_t - \lambda_U\, g_{U,t} - \lambda_T\, g_{T,t}\qquad(3)$$

where \(r_t\) is the environment reward of Eq. (1), not the bare cost term. The
initial study posed single-objective cost minimisation, yet every constrained
variant kept the shaping weights (0.8, 0.1, 0.1) active. An ablation isolating
them (Appendix B, Table B1) finds no resolvable effect under the Section 6.1
bootstrap, but the discrepancy is recorded as the first place at which the
implementation departed from the problem posed.

### 4.2 The cumulative-rate slack and why it saturates any multiplier

The dual variables are driven by a slack signal \(g\). The initial implementation
computed, at every step after a 100-step warm-up, the throughput signal of
Eq. (4),

$$g_{T,t} = \max\!\left(0,\; \frac{T_{\min}}{H} - \frac{d_t}{t}\right)\qquad(4)$$

with \(d_t\) the cumulative departures to step \(t\), and the utilisation signal
\(g_{U,t} = \sum_{i \in F_{\mathrm{fast}}} \max(0,\, U_{\min} - \bar u_{i,t})\), with
\(\bar u_{i,t}\) the cumulative utilisation of server \(i\) to step \(t\); both are
averaged over the post-warm-up steps and the multipliers are updated once per
episode from those means. Eq. (4) is not the constraint of
Eq. (2): where flow time is a sizeable fraction of the horizon, the filling line
holds the cumulative rate below the floor for a long prefix of every episode
whatever the policy does. Under ShortestQueue over the 50 test episodes it first
reaches the floor at a median step of 185 (bakery, range 101 to 434, with one
episode never reaching it) and 194 (electronics, 101 to 399), staying below on
45% and 31% of post-warm-up steps. A
policy achieving exactly \(\mathrm{TP} = 50\) on electronics meets the constraint but does not
cross until \(t \approx 480\), never satisfying the training signal within the horizon. The
utilisation slack carried the same bias in milder form.

The one-sided dual update of Eq. (5), applied once per episode with the
post-warm-up mean,

$$\lambda \leftarrow \operatorname{clip}\!\left(\lambda + \eta\,\operatorname{mean}_t\, g_t,\; 0,\; \lambda_{\max}\right)\qquad(5)$$

is a violation-driven ratchet rather than a dual ascent, turning that bias into a
structural outcome. Over the same episodes, the mean per-step \(g_T\) under
ShortestQueue is 0.0031 (bakery) and 0.0054 (electronics); at \(\eta_T = 4000\) that is
12.5 and 21.6 per episode against the 6.30 and 5.91 needed to carry \(\lambda_T\) from 200
to its cap of 20,000 within the 3,142 and 3,352 training episodes, so any mean
slack above 0.0016 (bakery) or 0.0015 (electronics) per step saturates the
multiplier within budget. ShortestQueue's exceeds the threshold by a factor of
two on bakery and 3.7 on electronics, capping at episode 1,581 of 3,142 and 915
of 3,352 respectively; the trained policies saturate sooner, at increments
of 15–16 and 39–49, and at episodes 1,223–1,359 (39–43% of training) and 402–504
(12–15%). Saturation reflects the signal, not the policy; reporting it as
evidence about the method is a category error.

### 4.3 Episode-level slack and the corrected dual updates

The per-step signal becomes the signed pro-rata decomposition of the episode
Lagrangian, Eq. (6),

$$p_t = \lambda_T\left(\frac{T_{\min}}{H} - \delta_t\right) + \lambda_U \sum_{i\in F_{\mathrm{fast}}}\left(U_{\min} - b_{i,t}\right)\qquad(6)$$

with \(\delta_t\) the departures in step t and \(b_{i,t} \in \{0, 1\}\) indicating whether server i
is busy; the augmented reward is \(\tilde r_t = r_t - p_t\), with no warm-up. Summed
without discount over the episode, Eq. (6) gives Eq. (7),

$$\sum_{t=0}^{H-1} p_t = \lambda_T\left(T_{\min} - \mathrm{TP}(\tau)\right) + \lambda_U\, H \sum_{i\in F_{\mathrm{fast}}}\left(U_{\min} - \bar u_i(\tau)\right)\qquad(7)$$

the Lagrangian penalty for the throughput constraint of Eq. (2) and for the
utilisation constraints in aggregate, in the same episode-level quantities that
validation and test evaluate, and free of fill-phase bias. Two qualifications
belong here. A single \(\lambda_U\) multiplies the shortfall summed over
\(F_{\mathrm{fast}}\), so the penalty enforces
\(\sum_i \mathbb{E}[\bar u_i] \ge |F_{\mathrm{fast}}|\, U_{\min}\) rather than each
server's floor separately, and surplus on one fast server can offset deficit on
the other; the selection rule of Section 3.3 checks each server. And the identity
holds for the undiscounted sum over the 479 decisions, up to a constant boundary
term from the first minute; what PPO optimises is the discounted sum of
Section 3.2. Multipliers update once per episode from the hinged episode-level
slack of Eqs. (8) and (9),

$$\lambda_T \leftarrow \operatorname{clip}\!\left(\lambda_T + \eta_T\,\frac{\max\!\left(0,\; T_{\min} - \mathrm{TP}(\tau)\right)}{H},\; 0,\; \lambda_{T,\max}\right)\qquad(8)$$

$$\lambda_U \leftarrow \operatorname{clip}\!\left(\lambda_U + \eta_U \sum_{i\in F_{\mathrm{fast}}} \max\!\left(0,\; U_{\min} - \bar u_i(\tau)\right),\; 0,\; \lambda_{U,\max}\right)\qquad(9)$$

Dividing by H keeps the throughput slack in the per-step rate units of Eq. (4),
so \(\eta_T\) and the initial value and cap of \(\lambda_T\) carry over unchanged.
The ratchet advances only on violating episodes, so a policy that satisfies the
constraint on most episodes no longer drives the multiplier to its cap; Section
6.1 confirms this for the trained policies.

Retaining the hinge keeps the update monotone: Eqs. (8) and (9) remove the
guaranteed divergence of Eq. (4), not the monotonicity, and under a stochastic
policy violating on a persistent fraction of episodes the multiplier grows at a
rate set by violation frequency, not expected slack. A symmetric variant uses the
signed slack \((T_{\min} - \mathrm{TP}(\tau))/H\) in Eq. (8) and, in Eq. (9), the summed shortfall
when any constrained server violates and the binding server's negative margin
otherwise, letting a multiplier fall back towards zero; the utilisation update
remains a heuristic on the aggregate. Section 6.4 shows why
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

Six dispatching rules are evaluated on the 50 test episodes (Table 1), all
deciding once per minute from the same observation as the agent (Section 3.1);
Appendix A.9 defines them as implemented. Four are state-aware: ShortestQueue,
which routes first to idle servers and breaks ties by queue length,
LeastUtilised, CostMinimising and FastServerFirst. Two are stateless and
resemble the learned policies (Section 6.5): UniformRandom, and RoundRobin,
which advances one route per decision rather than per job. ShortestQueue, the
primary comparator, is cheapest per unit on electronics with both constraints
satisfied on every test episode; on bakery it is not separable from RoundRobin,
whose mean is $1.11 lower with heavily overlapping intervals.

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

\(F_{\mathrm{fast}}\) and \(U_{\min}\) come from a one-shot LeastUtilised run on each testbed,
anchoring the feasible region on the heuristic family the agent is compared
against; \(T_{\min}\) was fixed in the same protocol below that run's throughput
of 19.6 and 56.5. Even stateless routing then satisfies both constraints on 78–90% of
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
comparable. PPO is the Stable-Baselines3 implementation [26] with
the initial study's hyperparameters, listed with the dual parameters in
Appendix A.

Three protocol documents govern the runs (Appendix A.1): the initial study's
protocol; an amendment, committed before any full-budget run here, requiring
both evaluation modes for every checkpoint and selection on the stochastic
validation episodes; and a second, committed before the runs of Section 6.7,
fixing that cell's multiplier scales a priori and listing its comparisons and
the wording each of three outcomes would receive. Two pipeline defects, unseeded
action sampling and joint satisfaction aggregated across seeds as the smaller
mean marginal rate, are recorded as one deviation and corrected in every figure
reported here; Appendix A.6 describes the correction and gives its size, which
changes the direction of no comparison.

Comparisons in Section 6.2 use a paired hierarchical bootstrap on the
per-episode test records, resampling training seeds within each learned cell
and test episodes jointly across the policies compared. This replaces
one-sample tests that treat baseline means as known constants and overstate
the evidence. On each testbed the ten reported tests, five comparators on two
metrics, form one family adjusted by Holm at \(\alpha = 0.05\); that family is the
complete set of comparisons reported rather than a pre-specified one, since the
bootstrap and the adjustment were introduced after the R1 runs in response to
review. The cell of Section 6.7 forms its own family of twelve, which its
protocol amendment did specify in advance.

## 6. Results

### 6.1 Multiplier saturation is a property of the slack signal

A 2×2 design crossing base reward {shaped, cost-only} with slack signal
{cumulative-rate, episode-level}, five seeds per cell at a reduced 400K-step
budget on electronics, attributes saturation unambiguously (Appendix B,
Table B1). All ten cumulative-rate seeds end at the \(\lambda_T\) cap of 20,000, all ten
episode-level seeds between 908 and 2,833: no overlap, no seed-level exception.
The base reward has no resolvable effect in either arm under the paired
bootstrap of Section 5.2 with the slack signal held fixed (Table B1, note), so
the Section 4.1 discrepancy is not a resolvable driver of the result at this
sample size, though the forty-point joint-satisfaction interval under the
saturated signal leaves a moderate effect of the shaping terms open, and the
attribution holds at the 400K-step budget on which it was tested.

The pattern holds at full budget on both testbeds (Figure 1). Under the
cumulative-rate signal the multiplier ramps monotonically to its cap on every
seed, at 0.19–0.24M steps on electronics and 0.59–0.65M on bakery, then is flat,
as Section 4.2 predicts. Under the episode-level signal it advances only on
violating episodes, ending between 2,525 and 7,858, never within a factor of 2.5
of the cap: it does not saturate, but neither has it converged — the monotonicity
of Section 4.3.

![](fig_lambda_T.png){width=16cm}\

**Figure 1.** Throughput multiplier \(\lambda_T\) over training, five seeds per cell,
logarithmic scale: (a) bakery, (b) electronics. Upper traces: the original signal,
the cumulative-rate slack of Eq. (4). Lower traces: the corrected cell, the
episode-level slack of Eqs. (8) and (9). The dashed line is the cap of 20,000.

### 6.2 Full-budget comparison

Two cells ran at full budget, five seeds each, selected on stochastic validation
and evaluated in both modes on the 50 test episodes: the original implementation
(shaped reward, cumulative-rate slack) as control, and the Section 4.3 correction
(cost-only reward, episode-level slack).

**Table 2.** Full-budget comparison. CI is the 95% two-sided t-interval across
seeds; "sat." is joint constraint satisfaction over test episodes. References
from Table 1: ShortestQueue $73.25 / 100% and RoundRobin $78.90 / 86% on
electronics; ShortestQueue $138.66 / 96% and RoundRobin $137.55 / 88% on bakery.

| Testbed | Cell | Stochastic CPU ± CI | Stochastic joint sat. | Val.-satisfied | \(\lambda_T\) at cap | Greedy CPU ± CI | Greedy joint sat. |
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

| Testbed and cell | Comparator | ΔCPU ($) | p (ΔCPU) | Δsat. (pp) | p (Δsat.) |
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
harder testbed. Second, on electronics the corrected cell is 25 percentage points
of joint satisfaction and about $4 per unit better than the control, both
surviving adjustment; the two cells differ in both reward and slack signal, and
the attribution of the gain to the slack signal rests on the reduced-budget
ablation of Section 6.1, within the interval noted there. Third, on bakery
neither difference is resolvable (+8.0 [0.0, +17.2] points of joint satisfaction,
p = 0.063; +$1.46 on cost per unit): the multiplier still saturates on all five
control seeds, so the Section 4.2 mechanism is present, but the penalty does not
damage the policy enough to show — a null result, not a partial success.

Fourth and most consequential, the corrected agent does not separate from
stateless routing on cost per unit: −$0.79 [−2.74, +1.37] against RoundRobin,
−$1.81 [−3.96, +0.41] against UniformRandom, intervals admitting a few dollars
either way — a failure to detect rather than a demonstration of equality. It
leads UniformRandom by 17 satisfaction points, which survives adjustment, and
RoundRobin by 9, which does not; on bakery no comparison against any rule is
significant. ShortestQueue stays ahead of every learned policy: on electronics
the corrected agent is 6.6% more expensive per unit than a rule at 100%
satisfaction.

Table 2 also shows the seed dispersion: greedy evaluation gives electronics
intervals of ±43 to ±76, stochastic evaluation of the same checkpoints ±1.6. The
severe seed-to-seed instability the initial study reported was, to a first
approximation, greedy-action variance over near-uniform distributions.

### 6.3 Re-evaluation of the archived checkpoints

The initial study's checkpoints were re-evaluated without retraining — the
one-sided Lagrangian of Section 4.2 and a PID-Lagrangian comparator in the manner
of Stooke and colleagues [22], driven by the same cumulative-rate
signal after exponential smoothing (Appendix A.2) — on both testbeds and every
archived seed. Applied to the archived validation sweep, the original greedy rule
reproduces the selected checkpoint in 18 of 18 runs and the test figures to the
cent: this pipeline produced them. Table 4 gives the result. The PID comparator
is the published remedy for the integral wind-up of Section 4.2, and on the
archived electronics runs its \(\lambda_T\) nonetheless reached the cap on all four
seeds, because the signal it damped was the biased one; damping does not remove a
bias.

**Table 4.** Re-evaluation of the archived checkpoints. Q1 evaluates the
originally selected checkpoint in both modes; Q2 applies stochastic selection.
n = 4 for the PID variant because one seed's checkpoints were not archived.

| Run | n | Q1 greedy CPU ± CI | Q1 greedy joint sat. | Q1 stochastic CPU ± CI | Q1 stochastic joint sat. | Q2 stochastic CPU ± CI | Q2 joint sat. | Q2 val.-satisfied |
|---|---|---|---|---|---|---|---|---|
| One-sided, electronics | 5 | $112.71 ± 13.46 | 6% | $81.35 ± 2.25 | 71% | $81.59 ± 2.68 | 74% | 5/5 (originally 0/5) |
| PID, electronics | 4 | $106.75 ± 23.87 | 22% | $83.94 ± 9.92 | 58% | $80.83 ± 3.40 | 72% | 3/4 (originally 1/4) |
| One-sided, bakery | 5 | $149.30 ± 27.73 | 69% | $140.56 ± 3.15 | 91% | $139.60 ± 2.49 | 80% | 5/5 (originally 4/5) |
| PID, bakery | 4 | $141.57 ± 22.71 | 74% | $140.46 ± 1.60 | 84% | $139.94 ± 3.72 | 83% | 4/4 (originally 3/4) |

The five electronics checkpoints recorded as failing, at $95–124 per unit under
greedy evaluation, score $79–84 as stochastic policies with joint satisfaction
of 64–88%; the files are unchanged. Had the original protocol selected on stochastic validation, all five
would have been reported as satisfying the criterion at $81.59 ± 2.68, at the
level of RoundRobin's and UniformRandom's point estimates in Table 1. The
negative result is thus reproduced and reversed on its own artefacts. The PID
rows show the same reversal: greedy joint satisfaction of 22% and 74% becomes
58% and 84% under sampling.

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
under cost-only weights is \(-\kappa\, C(\tau)/C_{\mathrm{norm}}\), between −10.0 and
−10.7 across seeds and testbeds, whereas the initial study compared the
integrated penalty against a per-episode cost in dollars, mixing units. In the
same units, the penalty of Eq. (7) on the corrected runs' test episodes is
already 190–300 times the return in magnitude at the initial multipliers
\((\lambda_T, \lambda_U) = (200, 5)\), and 1,200–10,000 times at those reached by
training's end; in the saturated control cell the ratio is of order \(10^{4}\).

The penalty's sign is negative on most episodes, since the load-spreading
policies over-satisfy both constraints in expectation: mean throughput 57.0
against a floor of 50 on electronics and 20.2 against 18 on bakery, fast-server
utilisation 0.79–0.91 against a floor of 0.50 for the corrected cell and
ShortestQueue. The fraction of episodes on which it is negative is the joint
satisfaction rate itself, 78–100% for the rules of Table 1 and 95% for the
corrected cell. It therefore rewards surplus throughput and fast-server busy
time rather than acting as a penalty, and cannot shrink: Eqs. (8) and (9) never
decrease a multiplier, whereas the equilibrium multiplier on a slack constraint
is zero. A ratio of totals is a diagnostic rather than a measure of gradient
influence, since a per-step constant is absorbed by the advantage baseline; the
comparison that bears on the gradient is between action-dependent quantities. A
unit of throughput moves the penalty by \(\lambda_T\), between 200 and 7,858
reward units, against a marginal value in the return of 0.03–0.15 reward units
by the chords of Sections 3.4 and 6.7, a ratio of \(10^3\) to \(10^5\); and the
penalty's per-episode standard deviation, about \(\lambda_T\) times that of
throughput, is 350–940 reward units at the initial multipliers against a return
standard deviation of about 1, so the advantage estimates are dominated by
penalty noise.

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

The hinged cells learn near-uniform routers with a tilt towards the constrained
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
feasible far more often, a bound on single-draw sampling variance, and a
comparison of the two hinged cells.

### 6.6 Sensitivity to the constraint thresholds

Because stateless routing meets the thresholds on most episodes (Table 1),
joint satisfaction was recomputed from the archived per-episode records at
stricter thresholds without retraining (Appendix B.2, Table B3); this re-scores
policies trained for the original thresholds rather than testing policies
trained for stricter ones. The two floors separate the policies in opposite
directions. Raising the utilisation floor to 0.70 rewards the tilt of
Section 6.5: the corrected agent recovers roughly two-thirds of the gap between
stateless routing and the state-aware rules. Raising the throughput floor
exposes what the agent never learned: at \(T_{\min} = 58\) the corrected agent
satisfies on 46% of episodes against ShortestQueue's 86%, closer to RoundRobin
than to the rule, because a demanding throughput floor requires routing that,
of the policies studied, only ShortestQueue performs, and the constraint the
agent was trained against did not require it.

### 6.7 A functioning dual, and what it reveals about the objective

The final cell removes the two features of Section 6.4 that held the multipliers
from equilibrium: the hinge, replaced by the signed update of Section 4.3, and
the inherited scale, replaced by multipliers initialised, stepped and capped on
the reward's scale (\(\lambda_T\): 0.1, step 2.0, cap 10; \(\lambda_U\): 0.002, step 0.01, cap 0.2).
The protocol amendment fixed these before any run from a shadow-price estimate,
about 0.1 reward units per unit of throughput and 0.002 per fast-server busy
step, and from the factor of roughly 2,400 by which the inherited multipliers
had to be divided to bring a typical episode penalty from about 250 times the
return to a tenth of it. Appendix A.7 gives the derivation and its two
weaknesses: the dollar figure behind the shadow price came from the bakery
testbed, and this testbed's own chord gives about 0.03, so the initial
\(\lambda_T\) sat two to three times above it; and the second argument rescales
the measured ratio of Section 6.4 rather than corroborating it. The amendment
also fixed three possible outcomes with the wording each would receive; the
result matches the second, a policy whose cost per unit sits above stateless
routing, with the qualification that the difference is not resolvable and that
no seed fell back. Five seeds, full budget, electronics (Table 5).

**Table 5.** Symmetric, reward-scaled cell, electronics, per seed. \(\lambda_T\) statistics
are over the 3,352 training episodes; the penalty ratio is the magnitude of the
episode penalty relative to the base return on the test episodes at the
training-mean multipliers. CIs are 95% t-intervals across seeds.

| Seed | Selected checkpoint | Stochastic CPU | Stochastic TP | Stochastic joint sat. | Greedy CPU | Greedy TP | Greedy joint sat. | \(\lambda_T\) mean / max | Penalty ratio |
|------|-----------|----------|--------|--------|----------|--------|--------|-------------|--------|
| 42 | 601K | $81.89 | 54.6 | 80% | $107.78 | 43.4 | 8% | 0.09 / 0.73 | 0.12 |
| 7 | 300K | $84.87 | 52.2 | 78% | $106.10 | 43.5 | 12% | 0.06 / 0.50 | 0.08 |
| 2024 | 200K | $80.92 | 53.9 | 76% | $135.70 | 34.4 | 0% | 0.06 / 0.46 | 0.09 |
| 123 | 300K | $77.80 | 57.0 | 72% | $92.76 | 48.1 | 28% | 0.13 / 1.48 | 0.14 |
| 999 | 1,103K | $84.05 | 52.6 | 68% | $109.88 | 39.6 | 2% | 0.07 / 0.55 | 0.07 |
| mean ± CI | | $81.91 ± 3.47 | 54.1 ± 2.4 | 74.8% | $110.45 ± 19.40 | 41.8 ± 6.3 | 10.0% | | |

![](fig_symmetric.png){width=16cm}\

**Figure 2.** The symmetric, reward-scaled cell during training, five seeds.
(a) \(\lambda_T\) per episode on a linear scale; the dashed line is the shadow-price
estimate of 0.1 from which the initial value was set. (b) Throughput slack
\(\mathrm{TP} - T_{\min}\) per training episode, 50-episode moving average; the dashed line is
the mean slack of uniform-random routing on the test episodes.

The throughput dual now behaves as a dual. \(\lambda_T\) falls to zero within a few
episodes when the constraint is slack, rises when violated, spends most of
training between 0 and 0.5 with excursions to 0.73 and 1.48 on two seeds
(Figure 2a, Table 5), never nears its cap, and is positive on 60–72% of episodes;
its per-seed training mean, 0.06–0.13, lies between this testbed's chord estimate
of 0.03 and the 0.1 at which it was initialised. The constraint
terms stay on the scale of the cost term, 7–14% of the return at the
training-mean multipliers, against 190 to 10,000 times it in the hinged cells.

Within the first 400K steps mean throughput per training episode falls from
random routing's 4.4 units above \(T_{\min}\) to one to two units above it, where
its per-seed mean stays (Figure 2b), with 24–39% of training episodes below
\(T_{\min}\) in every quarter of every seed. The selected checkpoints tell a
different story: four of the five were selected at or before 601K steps, inside
the window in which the fall occurs, and their test throughput of 54.1 ± 2.4 is
that of UniformRandom (54.40), because the 16-of-20 rule retains the checkpoints
from before the move to the floor, the ones that pass it. Of 95 checkpoints
validated across five seeds only 14 qualify, though every seed had at least one
and none fell back. Pooled over the 250 test episodes the selected checkpoints
satisfy the throughput constraint on 84.8% and the utilisation constraint on
82.0%, neither distinguishable at five seeds from the 80% the rule demands, and
per seed two of five fall below it on utilisation (74%, 74%) and one on
throughput (76%); jointly they satisfy both on only 74.8%, 20 points below the
corrected cell and 25 below ShortestQueue, a resolvable gap. This is the
configuration Section 3.3 promised, with the qualification that the marginal
chance constraints are met on average rather than on every seed: the expectation
constraint of Eq. (2) holds with a margin of 4.1 units, the two marginals sit at
the line, and the reported joint criterion fails by a wide margin.

On the reported metric it is no better: $81.91 ± 3.47, significantly above
ShortestQueue, nominally above RoundRobin (+$3.01 [+0.17, +5.86]) though not
after adjustment, and not separable from UniformRandom, LeastUtilised or the
original signal (Table 3). Nor is it cheapest on the total cost it minimises:
$4,414 per episode against $4,342 (RoundRobin), $4,346 (UniformRandom), $4,445
(corrected cell), $4,493 (ShortestQueue) and $4,542 (LeastUtilised). Under the
paired bootstrap of Section 5.2 on total cost it sits $72 above RoundRobin
[−$42, +$183] and $67 above UniformRandom [−$53, +$187], neither resolvable, and
$129 below LeastUtilised [−$229, −$30], nominally resolvable but outside any
pre-specified family and at a p-value Section 8 treats as indicative only. The
direct measurement promised in Section 3.4 shows why: across the seven
load-spreading policies mean episode cost spans 4.6%, $4,342 to $4,542, because
on a capacity-limited line (Section 3.1) the work completed is fixed by server
busy time and routing changes only which server does it and how long jobs wait;
throughput spans 15.5%, 53.2 to 61.4 units. An objective nearly flat in
routing and indifferent to throughput above the floor teaches a policy gradient
little, and what it teaches, that such throughput is not worth paying for, is the
opposite of what cost per unit rewards.

Its action distributions are unlike any other cell's (Table B2): normalised
entropy 0.63–0.83 against 0.94–0.98 for the hinged cells, yet the entropy of the
route frequencies marginalised over an episode remains 0.89–0.96, so the
concentration is state-dependent: this policy routes differently in different
states. Its greedy action is a coherent conservative router at throughput 41.8
and $110 per unit, feasible on 10% of episodes (Table 5), while the stochastic
policy behind it reaches 54 units at $82. Greedy extraction fails here not
because the logits are flat, since noise of 0.1 now moves the greedy action at
10% of states rather than 25%, but because a quarter of the throughput is
carried by the randomisation, whether the time-sharing of a constrained optimum
(Section 2.2) or the residue of entropy regularisation on a nearly flat
objective; the two cannot be told apart here.

A functioning dual on the stated CMDP therefore produces no policy competitive
with ShortestQueue on either metric. Whether it approaches the total-cost optimum
cannot be established here: its total cost is not below that of stateless
routing, so the objective was pursued but not demonstrably solved.

## 7. Discussion

### 7.1 Where the chain broke, and what each break cost

The three breaks differ in kind. The first, a per-step surrogate for an
episode-level constraint, is a modelling error of generalisable form: wherever
flow time is a sizeable fraction of the horizon, a cumulative-average translation
is biased throughout the fill phase by a small amount, of order \(10^{-3}\) per
step, that a symmetric dual update would release once the line filled and a
monotone one-sided update integrates without bound; the multiplier saturates on
every control seed, but the policy is measurably damaged only on the 12-route
instance. The second, greedy evaluation of a near-uniform policy, reverses the
reported verdict. The third, the scale of the multipliers, survives correction
of the other two, explains the residual result and is the least visible: nothing
in the original pipeline would have surfaced it, whereas a single ratio of
penalty to base return exposes it at once.

### 7.2 What the method learned, and what it was asked to learn

The penalty did not overwhelm the agent's objective: for practical purposes it
was that objective. With the multipliers two to four orders of magnitude above
the cost term, never released, and both constraints slack under any
load-spreading policy, the augmented return rewarded surplus throughput and
fast-server busy time, registering cost only in the third or fourth significant
figure; every hinged cell responded with a near-uniform router tilted towards the
fast servers.

The symmetric cell moves the fault upstream, though not as far as a clean result
would. A reward-scaled dual free to fall does make PPO leave the uniform policy
and learn a state-dependent router whose training throughput falls towards the
floor as a total-cost objective directs; but the checkpoints the selection rule
retains sit at random routing's throughput, and the cell's total cost is not
below that of the stateless rules (Section 6.7). "Failed to learn" therefore
needs qualification twice over: the hinged cells were never seriously asked to
minimise cost, and the symmetric cell was asked to minimise a quantity that is
not the reported metric and moved in that direction without demonstrably
reaching it.

Three consequences follow. If cost per unit is the criterion, the objective
should be aligned with it, by a ratio objective or a throughput term weighted by
that metric, not by a constraint; a per-departure bonus of that kind narrowed but
did not close the gap to ShortestQueue for unconstrained PPO on these testbeds
[16]. A total-cost CMDP is legitimate but should then be
judged on total cost. And the constraint enforced should be the constraint
checked: a joint per-episode criterion needs a per-episode formulation, a
violation-indicator penalty or a conditional-value-at-risk constraint on the
joint event, not an expectation constraint whose satisfaction still leaves a
quarter of episodes jointly infeasible. A functioning dual is not sufficient:
here it changed what was learned without making it competitive, and on these
testbeds the hinged cell, whose dual never released, remains the better policy on
both metrics. Tuned dispatching rules remain the baseline to beat at this problem
size and structure.

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
attributable to no single structural feature, absent a controlled ablation. Three
features of the method were held fixed and not examined: the discount
\(\gamma = 0.99\), which makes the optimised objective a horizon-weighted
approximation to Eq. (2) (Section 3.2); the single utilisation multiplier, which
enforces the fast-server floors in aggregate rather than separately
(Section 4.3); and an observation without a time index, so that a stationary
policy is asked to solve a finite-horizon problem. Five
seeds per cell is a small sample: stochastic-mode intervals are tight
(±0.7 to ±2.0 for the four cells of Table 2, ±3.5 for the symmetric cell), but
the bootstrap has only 126 distinct resamples of those five seed-level means
and is anti-conservative at this cluster count, leaving between-cell intervals
approximate and p-values between 0.01 and 0.05 no better than indicative. The
null comparisons against stateless routing, and the bakery null of Section 6.2,
are failures to detect rather than evidence of equality: the electronics
interval against RoundRobin spans −$2.74 to +$1.37 per unit.

The thresholds are ones that Table 1 shows stateless routing meeting on 78–90%
of episodes, so reliability is compared at
undemanding thresholds, and Section 6.6 tightens them only by re-scoring
policies trained for them. Each validation and test episode samples one action
trajectory; Appendix B.1 puts that single-draw noise at roughly half the
between-seed variance, and the validation draws were single and unseeded. Q2 re-selection
(Section 6.3) reuses the reporting episodes, so Q1 carries the principal
claim; and the amendment introducing stochastic selection, though
pre-committed, followed the reduced-budget ablation and is not independent of
the same study's data.

The symmetric cell tests a functioning dual in one configuration only:
multiplier scales fixed a priori, though after inspecting the earlier results,
electronics alone, and plain dual ascent, whose \(\lambda_T\) oscillates rather than
converging (Figure 2a); a damped or averaged dual might select different
checkpoints. Its multiplier scale was set from a shadow price derived on the other
testbed (Appendix A.7). The mechanism it exposes is measured across the seven
load-spreading policies of the 12-route testbed and survives those choices; its
numbers do not, and no total-cost figures exist for the 4-route testbed. The
thresholds at which Section 6.6 shows the problem becoming informative,
\(T_{\min}\) of 56–58, were not trained against. All results are from
simulation, with no physical validation.

## 9. Conclusions

The failure reported by the author's initial study of constrained reinforcement
learning on this problem traces to three breaks between the constraint stated
and the metric reported. A per-step cumulative-rate surrogate for an
episode-level constraint capped the multiplier within budget for every trained
policy, and by the same arithmetic would do so for a dispatching rule that
satisfies the constraint with a 14–23% margin. Greedy evaluation of near-uniform
policies measured a router selected by unstable logit differences; correcting
that alone reverses the verdict on the archived checkpoints without retraining.
With both corrected, every seed satisfies the constraint criterion, yet on the
12-route testbed the agent is not distinguishable from round-robin routing on
cost per unit at this sample size and is 6.6% more expensive than ShortestQueue,
with multipliers two to four orders of magnitude above the cost term and never
released. A reward-scaled, symmetric dual behaves as a dual and the policy
becomes state-dependent, but its training throughput falls towards the floor as
a total-cost objective directs, the checkpoints retained are no better on the
reported metric than the stateless rules, and their total cost is not below
that of stateless routing either: the objective was pursued but not demonstrably
solved.

The objective posed, the constraint enforced, the surrogate trained against, the
evaluation mode and the reported metric form a chain that must be coherent; here
it was not. Across the seven load-spreading policies of the 12-route testbed,
total cost spans 4.6% and throughput 15.5%, so on a capacity-limited line a
total-cost CMDP is the wrong instrument for a cost-per-unit objective however
well its dual behaves. Separating these claims required reporting the constraint
actually implemented, the penalty's scale against the objective, the evaluation
mode used, and a random-routing control; all four belong in routine practice.

## Author contributions

The author conceived and designed the work, implemented the simulator, wrappers
and analysis, conducted the experiments, analysed and interpreted the data,
drafted the paper and revised it critically for intellectual content, approved
the version to be published, and agrees to be accountable for all aspects of the
work.

## Funding

This research received no specific grant from funding agencies in the public,
commercial, or not-for-profit sectors.

## Disclosure statement

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
The bakery service-time data are from the openly available dataset of Babor
[25].

## References

1. Mayerhoff J, Schmidt M. Reinforcement learning for autonomous production planning and control: a systematic literature review. J Manuf Syst. 2026;86:546–568. doi:10.1016/j.jmsy.2026.03.023.

2. Schneider J, Pfannschmidt C, Nyhuis P, Schmidt M. The role of reinforcement learning in production control: a systematic literature review. IEEE Access. 2026;14:34375–34389. doi:10.1109/ACCESS.2026.3668903.

3. Doherty M, Matzner R, Sadeghi R, Bayvel P, Beghelli A. Reinforcement learning for dynamic resource allocation in optical networks: hype or hope? J Opt Commun Netw. 2025;17(9):D1–D17. doi:10.1364/JOCN.559990.

4. Schulman J, Wolski F, Dhariwal P, Radford A, Klimov O. Proximal policy optimization algorithms. arXiv:1707.06347; 2017. doi:10.48550/arXiv.1707.06347.

5. Alrashdan KR. FlexFlowSim-CPPO: simulator, protocol documents and archived results [software]. GitHub; 2026 [cited 2026 Sep 12]. Available from: https://github.com/alrashdank/flexflowsim-cppo

6. Li C, Zhao X, Lin L, Zhang W, Gen M, Zhang Q. An evolutionary knowledge training-based proximal policy optimization algorithm for job shop scheduling in flexible intelligent manufacturing. Comput Ind Eng. 2025;210:111533. doi:10.1016/j.cie.2025.111533.

7. Liu Y, Fan J, Shen W. A deep reinforcement learning approach with graph attention network and multi-signal differential reward for dynamic hybrid flow shop scheduling problem. J Manuf Syst. 2025;80:643–661. doi:10.1016/j.jmsy.2025.03.028.

8. Shen Y, Zhang X, Jin T. Transformer-based multi-agent reinforcement learning for flexible job shop scheduling with AGVs. Appl Soft Comput. 2026;193:114899. doi:10.1016/j.asoc.2026.114899.

9. Wang R, Jing Y, Gu C, He S, Chen J. End-to-end multitarget flexible job shop scheduling with deep reinforcement learning. IEEE Internet Things J. 2025;12(4):4420–4434. doi:10.1109/JIOT.2024.3485748.

10. Ali AM, Tirel L. Action masked deep reinforcement learning for controlling industrial assembly lines. In: 2023 IEEE World AI IoT Congress (AIIoT); 2023. p. 797–803. doi:10.1109/AIIoT58121.2023.10174426.

11. Tang CY, Liu CH, Chen WK, You SD. Implementing action mask in proximal policy optimization (PPO) algorithm. ICT Express. 2020;6(3):200–203. doi:10.1016/j.icte.2020.05.003.

12. Zhang N, Liu B, Zhang J. Dual resource scheduling method of production equipment and rail-guided vehicles based on proximal policy optimization algorithm. Technologies. 2025;13(12):573. doi:10.3390/technologies13120573.

13. Ferreira C, Figueira G, Amorim P. Effective and interpretable dispatching rules for dynamic job shops via guided empirical learning. Omega. 2022;111:102643. doi:10.1016/j.omega.2022.102643.

14. Huang Z, Mei Y, Zhang F, Zhang M. Toward evolving dispatching rules with flow control operations by grammar-guided linear genetic programming. IEEE Trans Evol Comput. 2025;29(1):217–231. doi:10.1109/TEVC.2024.3353207.

15. Marques N, Figueira G, Guimarães L. Dynamic dispatching rule selection for the job shop scheduling problem. Comput Ind Eng. 2025;210:111471. doi:10.1016/j.cie.2025.111471.

16. Alrashdan KR. Routing under machine breakdowns: a benchmark of dispatching rules, bandits, and reinforcement learning for multi-server flow shops. J King Saud Univ Eng Sci. 2026;38(7):55. doi:10.1007/s44444-026-00128-9.

17. Rinciog A, Meyer A. Fabricatio-RL: a reinforcement learning simulation framework for production scheduling. In: Proceedings of the 2021 Winter Simulation Conference; 2021. p. 1–12. doi:10.1109/WSC52266.2021.9715366.

18. Altman E. Constrained Markov decision processes. Boca Raton (FL): Chapman & Hall/CRC; 1999.

19. Achiam J, Held D, Tamar A, Abbeel P. Constrained policy optimization. In: Proceedings of the 34th International Conference on Machine Learning. PMLR 70; 2017. p. 22–31.

20. Tessler C, Mankowitz DJ, Mannor S. Reward constrained policy optimization. In: 7th International Conference on Learning Representations; 2019.

21. Paternain S, Chamon LFO, Calvo-Fullana M, Ribeiro A. Constrained reinforcement learning has zero duality gap. In: Advances in Neural Information Processing Systems 32; 2019. p. 7553–7563.

22. Stooke A, Achiam J, Abbeel P. Responsive safety in reinforcement learning by PID Lagrangian methods. In: Proceedings of the 37th International Conference on Machine Learning. PMLR 119; 2020. p. 9133–9143.

23. Henderson P, Islam R, Bachman P, Pineau J, Precup D, Meger D. Deep reinforcement learning that matters. In: Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence; 2018. p. 3207–3214. doi:10.1609/aaai.v32i1.11694.

24. Agarwal R, Schwarzer M, Castro PS, Courville A, Bellemare MG. Deep reinforcement learning at the edge of the statistical precipice. In: Advances in Neural Information Processing Systems 34; 2021. p. 29304–29320.

25. Babor M. Small and medium-sized bakery production data for scheduling. Version 2 [dataset]. Mendeley Data; 2022. doi:10.17632/dhgbssb8ns.2.

26. Raffin A, Hill A, Gleave A, Kanervisto A, Ernestus M, Dormann N. Stable-Baselines3: reliable reinforcement learning implementations. J Mach Learn Res. 2021;22(268):1–8.

## Appendix A. Reproducibility

**A.1 Software.** All experiments use the FlexFlowSim-CPPO simulator
[5]. The testbeds are defined in `configs/bakery_bk50.json` and
`configs/electronics_3stage.json`. The original Lagrangian wrapper is
`pilot_constrained_v4_auto.py`; the corrected wrapper, which also reproduces the
original signal as a control mode and provides the symmetric variant, is
`lagrangian_slack.py`; the PID-Lagrangian comparator of Section 6.3 is
`pilot_constrained_v6_auto.py` with the controller of `pilot_constrained_v6_pid.py`.
Training, validation and test are driven by `run_ablation_2x2.py`, the archival
re-evaluation by `archival_reeval.py`, and the
policy statistics, seeded per-episode re-evaluation, bootstrap and threshold
re-scoring by `analysis_stateless_and_entropy.py`, `analysis_r2.py`,
`analysis_r2_stats.py` and `analysis_r2_tables.py`. The fill-phase statistics of
Section 4.2, the saturation episodes read from the archived multiplier
histories, the marginal satisfaction rates and checkpoint counts of Section 6.7,
and the total-cost and base-reward bootstraps of Sections 6.7 and 6.1 are
produced by `analysis_audit_r3.py`, which writes `r3_numbers.json`; Figures 1 and
2 are drawn by `make_fig_lambda.py` and `make_fig_r2.py`. The protocol is `protocol.md`;
its amendments are `protocol_amendment_r1.md` and `protocol_amendment_r2.md`;
deviations are recorded in `protocol_deviations.md`.

**Table A1.** Testbed parameters from the configuration files. Inter-arrival
times are exponential with mean 9.6 min (bakery) and 6.0 min (electronics);
waiting cost is $0.10 (bakery) and $0.15 (electronics) per queued job per
minute. Service times are normal with the stated mean and standard deviation in
minutes, truncated below at the floor. Processing and idle costs are $ per
minute while busy and while idle. Queue lengths are normalised by 50 in the
observation. Capacity is the stage's expected completions per 480-minute shift
with every server busy. Constrained servers \(F_{\mathrm{fast}}\) are marked
\(\ast\).

| Testbed | Stage | Server | Mean | SD | Floor | Processing | Idle | Stage capacity |
|---|---|---|---|---|---|---|---|---|
| Bakery | 1 | fast \(\ast\) | 14.2 | 5.8 | 1.0 | 1.5 | 0.5 | 62.5 |
| Bakery | 1 | slow | 16.7 | 6.5 | 1.0 | 1.0 | 0.5 | |
| Bakery | 2 | fast \(\ast\) | 36.6 | 15.2 | 5.0 | 1.5 | 0.5 | 23.1 |
| Bakery | 2 | slow | 47.9 | 10.0 | 5.0 | 1.0 | 0.5 | |
| Electronics | 1 | fast \(\ast\) | 8.0 | 2.5 | 1.0 | 2.0 | 0.3 | 100.0 |
| Electronics | 1 | standard | 12.0 | 3.0 | 1.0 | 1.0 | 0.3 | |
| Electronics | 2 | fast \(\ast\) | 15.0 | 4.0 | 1.0 | 2.5 | 0.5 | 73.1 |
| Electronics | 2 | medium | 20.0 | 5.0 | 1.0 | 1.5 | 0.5 | |
| Electronics | 2 | slow | 28.0 | 6.0 | 1.0 | 0.8 | 0.3 | |
| Electronics | 3 | fast | 5.0 | 1.5 | 0.5 | 1.8 | 0.2 | 144.0 |
| Electronics | 3 | slow | 10.0 | 3.0 | 1.0 | 0.6 | 0.1 | |

**A.2 Hyperparameters.** PPO (Stable-Baselines3, MlpPolicy, two hidden layers of
64 tanh units): learning rate \(3 \times 10^{-4}\); n_steps 2048; batch size 64; 10 epochs;
\(\gamma = 0.99\), so that the return optimised is the discounted sum with the
truncated final step bootstrapped by the value function, as Stable-Baselines3
does for time-limit truncation; GAE \(\lambda = 0.95\); clip range 0.2; entropy
coefficient 0.01; value-function coefficient 0.5; maximum gradient norm 0.5.
Lagrangian outer loop, identical in
both slack modes: \(U_{\min} = 0.50\); \(T_{\min} = 18\) (bakery) or 50 (electronics); \(\lambda_U\)
initial 5, step 20, cap 500; \(\lambda_T\) initial 200, step 4000, cap 20,000. The
cumulative-rate mode applies a 100-step warm-up and the episode mode none.
Multipliers are updated once per episode in all modes. The symmetric cell of
Section 6.7 uses the signed update with \(\lambda_T\) initial 0.1, step 2.0, cap 10 and \(\lambda_U\)
initial 0.002, step 0.01, cap 0.2, with all PPO hyperparameters unchanged.

The PID-Lagrangian comparator re-evaluated in Section 6.3 applies the same
per-step cumulative-rate penalty and warm-up as the original wrapper, feeds the
dual controller the per-episode mean signed slack smoothed by an exponential
moving average with \(\alpha = 0.3\), and sets each multiplier to the clipped sum
of proportional, integral and derivative terms with gains
\((K_P, K_I, K_D) = (100, 1000, 500)\) for \(\lambda_T\) and \((5, 5, 25)\) for
\(\lambda_U\), caps 20,000 and 500, and integrator freeze as anti-windup. Its
integrator was initialised to \(\lambda_0 / K_I\) rather than to \(\lambda_0\),
so the effective initial \(\lambda_T\) was near zero rather than the nominal 200;
the archived histories start at 0–3. On the four archived electronics seeds
\(\lambda_T\) reached the cap of 20,000 in every run; on the four bakery seeds it
ended between 5,531 and 7,267.

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
the seeded re-evaluation described in A.6. Cost per unit is total episode cost divided
by episode departures, averaged over episodes. Joint satisfaction is the fraction
of episodes in which every constrained server's realised utilisation is at least
\(U_{\min}\) and departures are at least \(T_{\min}\), computed per episode.

**A.5 Statistical treatment.** All comparisons in Table 3 use a paired
hierarchical bootstrap on the per-episode test records. Let X be the seeds ×
episodes matrix of a learned cell's per-episode metric (5 × 50) and Y the
comparator's (5 × 50 for another learned cell, 1 × 50 for a dispatching rule).
Each of 10,000 replicates draws seed indices with replacement, independently for
X and Y, and a single vector of 50 episode indices with replacement applied to
both, so the two policies are compared on the same resampled episodes; the
statistic is the difference of grand means. Intervals are 2.5th–97.5th
percentiles; p is twice the smaller tail proportion at zero, so with 10,000
replicates its resolution limit is \(2 \times 10^{-4}\); of the seven entries
reported as < 0.001, six are at that limit and one, the corrected cell against
the original signal on cost per unit, is 0.0004. Resampling five seed-level means admits 126 distinct
multisets in the seed dimension, which makes the procedure anti-conservative at
this cluster count; p-values between 0.01 and 0.05 should be read accordingly.
The family on each testbed is the corrected cell against the four load-spreading
rules of Table 1 and against the original signal, on cost per unit and on joint
satisfaction, ten tests adjusted by Holm's step-down procedure at
\(\alpha = 0.05\); it is the complete set of comparisons reported, defined when
the bootstrap was introduced after the R1 runs. The symmetric cell's family of
twelve was specified in its protocol amendment before its runs. The nine further
bootstrap comparisons reported in Sections 6.1 and 6.7 belong to no family and
are interpreted as exploratory.
Confidence intervals use the t-distribution with n − 1 degrees of freedom across
seeds and the normal approximation across episodes.

**A.6 The recorded deviation and its size.** Two pipeline defects are recorded
in `protocol_deviations.md` (Section 5.2): action sampling in validation and
test was unseeded, and joint satisfaction was aggregated across seeds as the
smaller of the two mean marginal rates rather than as the per-episode joint
rate. Every stochastic figure reported here is therefore a seeded re-evaluation
of the selected checkpoints, with the generator seeded by the episode seed
before each episode, and every joint-satisfaction figure a per-episode
recomputation from the archived records. Greedy cost per unit is unaffected by both and reproduces the pipeline's
values to the cent, which is the fidelity check on the re-evaluation. Greedy
joint satisfaction moves by at most 1.2 percentage points under the aggregation
correction. Between the unseeded pipeline draw and the seeded re-evaluation,
cell-level figures differ by at most $1.2 per unit and 5.6 percentage points of
joint satisfaction; no comparison changes direction, and the largest consequence
is that the slack correction of Section 6.2 is worth 19 rather than 25 percentage
points on the other draw.

**A.7 Multiplier scales of the symmetric cell.** The protocol amendment for the
cell of Section 6.7 fixed the multiplier scales listed in A.2 before any run, by
two arguments. The first was a shadow price: the marginal cost of a unit of
throughput from the bakery chord of Section 3.4, about $40, converted at the
electronics normalisation \(\kappa/C_{\mathrm{norm}} = 10/4230\) to about 0.1
reward units; and a busy-step of a fast server priced at the fast–slow
processing-cost difference of $0.5–1.0 per minute, about 0.002 reward units. The
second was the factor of roughly 2,400 by which the inherited multipliers had to
be divided to bring a typical episode penalty from about 250 times the return to
a tenth of it. Two things about that derivation should be said plainly. The
dollar figure came from the other testbed and the normaliser from this one; the
same chord on electronics, from CostMinimising to ShortestQueue, gives about $12
per unit, or 0.03 reward units, so the initial \(\lambda_T\) was set two to
three times above this testbed's own estimate, and the per-seed training means of
0.06–0.13 in Table 5 lie between the two. And the second argument is not
independent of the first, since the 2,400 is the measured penalty ratio of
Section 6.4 divided by the chosen target; it is a scaling heuristic, not
corroboration.

**A.8 Hardware and runtime.** Training and evaluation ran on a two-vCPU Linux
container at approximately 800–1,000 environment steps per second. A 400K-step
segment takes 7–8 minutes and a full 1.6M-step run about 30 minutes plus 10–15
minutes of dual-mode validation. The complete matrix reported here, comprising 20
reduced-budget runs, 25 full-budget runs, 18 archival re-evaluations of 279
checkpoints, the baselines and all analyses, consumed approximately 21
compute-hours.

**A.9 Dispatching rules as implemented.** All rules decide once per minute from
the observation of Section 3.1 and assign the arriving job a complete route.
ShortestQueue minimises, summed over the servers on a route, the in-service
indicator plus the queue length normalised by 50; because the indicator
outweighs the normalised queue, it routes first to idle servers and breaks ties
by queue length, and is named for the tie-break. LeastUtilised weights that load
by mean service time. CostMinimising minimises summed processing cost per unit
time, ignoring queues. FastServerFirst takes the fastest server at each stage.
UniformRandom draws a route uniformly at each decision. RoundRobin advances one
route per decision; with a job every 6–10 minutes on average it is therefore
round-robin over decisions rather than over jobs. The rules are implemented in
`baselines.py`.

## Appendix B. Supplementary tables

**B.1 Structure of the learned policies (Section 6.5).** Normalised entropy of
the hinged electronics cells is 0.94–0.98 and the effective number of routes
10.4–11.5 of 12; the corrected policies load the two constrained fast servers to
0.82 and 0.79 mean utilisation against 0.81 and 0.90 under ShortestQueue. Two
seeds' greedy policies agree on 9% of states in the control cell against 18% in
the corrected cell and 8.3% by chance. On bakery, with four routes, the top-1
gaps are larger and cross-seed agreement higher, and greedy extraction is
feasible on 54% of episodes rather than none. One check bounds the stochastic
figures. Single-draw evaluation carries sampling variance of its own:
re-evaluating one electronics checkpoint five times with different sampling seeds
gives cost per unit 79.11–81.63 (SD 0.96) and joint satisfaction 88–96%
(SD 0.036), against a between-seed SD of 1.32 for the same cell, so roughly half
the between-seed variance the bootstrap resamples is single-draw noise; the
bootstrap of Table 3 resamples episodes as well as seeds. Comparing the two
hinged cells, the control is if anything closer to uniform than the corrected
cell, consistent with Section 6.4: a saturated penalty four orders of magnitude
larger than the base reward leaves the policy gradient with almost nothing to say
about routing.

**B.2 Threshold sensitivity (Section 6.6).** Table B3 re-scores every policy at
alternative thresholds without retraining. At \(U_{\min} = 0.70\) on electronics,
ShortestQueue and LeastUtilised still satisfy on 96% of episodes, uniform-random
routing on 16% and the corrected agent on 71%. At \(T_{\min} = 58\) the corrected
agent falls to 46% against ShortestQueue's 86% and RoundRobin's 30%, and
LeastUtilised, the other state-aware rule, falls further still, to 34%. Raising
the throughput floor to \(T_{\min} = 60\) puts every learned and stateless policy
at or below 30% while ShortestQueue holds 74%. On bakery the orderings are
compressed throughout because four routes leave less room for a routing policy
to differ from another.

**Table B1.** Mechanism ablation, electronics, 400K steps, 5 seeds per cell,
greedy selection. Reference: ShortestQueue $73.25 per unit at throughput 61.4.
Note: with the slack signal held fixed, replacing the shaped reward with the
cost-only reward moves stochastic cost per unit by −$1.93 [−3.99, +0.36]
(cumulative-rate) and +$0.22 [−1.42, +2.05] (episode-level), and joint
satisfaction by +11.6 [−7.6, +32.0] and −0.8 [−8.0, +6.0] percentage points
(paired bootstrap of Section 5.2, 95% intervals).

| Cell | Greedy CPU ± CI | Greedy joint sat. | Stochastic CPU ± CI | Stochastic joint sat. | \(\lambda_T\) at cap |
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

| Testbed | Cell | norm. entropy | eff. routes | top-1 prob. | top-1 − top-2 gap | greedy agreement | flip rate (\(\sigma = 0.1\)) |
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

| Testbed | \(T_{\min}\) | \(U_{\min}\) | SQ | LU | RR | Random | corrected | original | symmetric |
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

**Table C1.** Symbols and abbreviations.

| Symbol | Meaning |
|---|---|
| PPO | Proximal policy optimisation |
| CMDP | Constrained Markov decision process |
| CPU | Cost per unit ($ per completed job; primary metric) |
| \(\mathrm{TP}(\tau)\) | Throughput: jobs completed in episode \(\tau\) |
| \(C(\tau)\), \(c_t\) | Episode cost total; cost accrued in step \(t\) |
| \(H\), \(\Delta t\) | Episode horizon, 480 steps; step length, 1 min |
| \(F_{\mathrm{fast}}\) | Set of fast servers subject to the utilisation constraint |
| \(T_{\min}\), \(U_{\min}\) | Throughput floor; per-server utilisation floor |
| \(\lambda_T\), \(\lambda_U\) | Lagrange multipliers for the throughput and utilisation constraints |
| \(\eta_T\), \(\eta_U\) | Multiplier step sizes |
| \(g_{T,t}\), \(g_{U,t}\) | Per-step throughput and utilisation slack signals of the original implementation, Eq. (4) and Section 4.2 |
| \(r_t\), \(\tilde r_t\) | Environment reward, Eq. (1); augmented reward, Eqs. (3) and (6) |
| \(\dot n_t\) | Cumulative average throughput rate \(d_t/t\) in Eq. (1) |
| \(\gamma\) | PPO discount factor, 0.99 |
| \(\tau\) | Episode (one simulated shift) |
| CI, SD, pp | Confidence interval; standard deviation; percentage points |
| PID | Proportional–integral–derivative multiplier controller (Section 6.3) |
| Q1, Q2 | Archival re-evaluation questions of Section 6.3: originally selected checkpoint in both modes; stochastic re-selection |
| \(p_t\) | Corrected per-step penalty, Eq. (6) |
| \(d_t\), \(\delta_t\) | Cumulative departures to step t; departures in step t |
| \(b_{i,t}\) | In-service indicator of server \(i\) at step \(t\) |
| \(\bar u_i(\tau)\), \(u_i(\tau)\) | Realised utilisation of server \(i\) over episode \(\tau\) |
| \(W_t\) | Work in process at step \(t\) |
| \(\kappa, w_c, w_t, w_w\) | Reward scale and shaping weights of Eq. (1) |
| \(C_{\mathrm{norm}}\), \(N_{\mathrm{norm}}\), \(W_{\mathrm{norm}}\) | Reward normalisation constants of Eq. (1) |
| \(\pi\) | Policy |
