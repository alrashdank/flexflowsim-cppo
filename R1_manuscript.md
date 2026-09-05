---
title: "Why Constrained Reinforcement Learning Appeared to Fail on Flow-Shop Routing: Two Artefacts in Training and Evaluation, and What Their Correction Reveals"
---

## Abstract

An earlier submission by the author reported that Lagrangian Proximal Policy
Optimization (PPO) fails to
satisfy throughput and utilisation constraints on a 12-action electronics
flow-shop testbed on every seed, with the throughput multiplier saturating in
every run. This paper shows that the failure was produced by two implementation
choices, not by the method. First, the training constraint penalised the
shortfall of the cumulative throughput rate at every step, a signal that is
positive during the pipeline fill of every episode regardless of the policy;
under a one-sided dual update it saturates the multiplier for any policy,
including an oracle (10 of 10 seeds saturate with this signal, 0 of 10 with an
episode-level slack). Second, policies were evaluated by greedy action selection;
evaluated stochastically, the same checkpoints move from approximately 0% to
68–94% joint constraint satisfaction. Re-evaluating the archived checkpoints
without retraining reproduces the reported figures under greedy evaluation and
reverses them under stochastic evaluation. A corrected formulation re-run at the
full pre-registered budget satisfies the validation criterion on every seed, at
$78.67 ± 2.32 per unit on electronics against $73.25 for ShortestQueue. The
correction also reveals what was learned: the policies are near-uniform routers
(normalised entropy 0.94–0.97) whose cost-per-unit is indistinguishable from
round-robin and uniform-random routing, better than stateless load-balancing only
in constraint reliability. The two artefacts are real and generalisable, but
constrained PPO here did not fail to satisfy its constraints so much as fail to
learn anything a state-aware dispatching rule does not do better. All code and
results are open source.

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
standardised benchmarks. An earlier manuscript by the author (Author, 2026a;
referred to below as the original submission) was a response to that critique,
with a pre-registered protocol, disjoint train, validation and test seed ranges,
tuned dispatching baselines, and non-convergence reported openly.

That submission reported a negative result. Under cost-emphasising reward
weights, PPO converges to what we called the conservative routing attractor: it
idles expensive servers, total cost falls, throughput falls further, and
cost-per-unit (CPU) ends up worse than that of any dispatching rule.
Reformulating the problem as a Constrained Markov Decision Process (CMDP; Altman,
1999) and applying Lagrangian penalties
repaired this partially on a 4-action bakery testbed and not at all on a
12-action electronics testbed, where no seed satisfied the validation criterion
and the multiplier on the throughput constraint saturated against its cap in
every run.

This paper revisits that conclusion. It finds that the reported failure was an
artefact of how the constraints were implemented and how the policies were
evaluated, and that once both are corrected a different and less flattering
picture of the method emerges. The paper makes four contributions.

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
the episode Lagrangian, which sums over the episode to the constraint the paper
states and the protocol evaluates. Every seed of the corrected formulation
satisfies the validation criterion on both testbeds. On the harder testbed the
correction also improves constraint satisfaction over the original implementation
(p = 0.002); the cost-per-unit difference (p = 0.021) does not survive adjustment
for the four tests performed.

The fourth follows from the second and is the finding we would least have
predicted. Compared against uniform-random and round-robin routing under the same
stochastic evaluation, the corrected constrained agent is indistinguishable on
cost-per-unit and better only on constraint reliability. Its cost-per-unit is 7.4%
above ShortestQueue, a state-aware rule that requires no training. The method
did not fail to satisfy its constraints; stateless load-balancing satisfies them
too. It failed to learn a routing policy that a two-line dispatching rule does not
beat.

The scope of the result should be stated plainly. The two artefacts are
properties of the pipeline, not of constrained reinforcement learning, and the
first in particular is easy to introduce and difficult to detect from aggregate
metrics. Correcting them does not make constrained PPO competitive on this
problem, and it does not make deterministic deployment feasible. What it does is
replace a wrong negative result with a correct one.

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
et al., 2025). Masking presupposes that some actions are genuinely infeasible,
which does not apply in the fully feasible routing problem we study.

Doherty et al. (2025) is the most directly relevant single paper on the
benchmarking problem. They reproduce five landmark RL studies in optical resource
allocation, apply properly tuned heuristic baselines, and find that simple
heuristics consistently match or outperform the published RL results. Rinciog and
Meyer (2021) introduced FabricatioRL, the closest comparator framework to our
simulator. A parallel line bypasses RL entirely and learns interpretable
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
in-service indicators and accumulated cost signals. The original submission
stated that it also contained the current dual multipliers. It does not; the
Lagrangian wrapper modifies the reward and leaves the observation unchanged, and
we correct the description here.

### 3.2 The conservative routing attractor

The unconstrained baseline uses the per-step scalarised reward of Eq. (1),

$$r_t = -\kappa\left(w_c\,\mathrm{cost\_rate}_t + w_t\left(1 - \dot n_t/\dot n_{\mathrm{target}}\right) + w_w\,\mathrm{wip}_t/\mathrm{wip}_{\mathrm{ref}}\right)\qquad(1)$$

with (w_c, w_t, w_w) summing to one and κ = 10 a fixed scale factor. Here ṅ_t is
the cumulative average throughput rate departures_t / t, not an instantaneous
rate. The original submission described it as instantaneous; this was incorrect,
and the distinction matters in §4.2. For w_c near unity PPO solves the problem it
was given and shuts down expensive servers. Cost rate falls, throughput falls
more, and cost-per-unit rises. Sweeping w_c does not escape the attractor: down
to w_c = 0.5, cost-per-unit never falls below that of the LeastUtilised rule.

### 3.3 Constrained formulation

Let H = 480. Write TP(τ) for the number of departures in episode τ and u_i(τ) for
the realised utilisation of server i. The agent solves the constrained problem of
Eq. (2),

$$\begin{aligned}\max_{\pi}\quad & \mathbb{E}_\pi\!\left[\sum_{t=0}^{H-1}(-c_t)\right]\\ \text{s.t.}\quad & \mathbb{E}_\pi[\mathrm{TP}(\tau)] \ge T_{\min},\\ & \mathbb{E}_\pi[u_i(\tau)] \ge U_{\min}\quad \text{for } i \in F_{\mathrm{fast}}\end{aligned}\qquad(2)$$

with T_min = 18 (bakery) or 50 (electronics) and U_min = 0.50. Both constraints
are on episode-level quantities. We state this more carefully than the original
submission did, where the constraint was written as an episode count while the
implementation used a per-step rate; §4.2 describes the consequences.

### 3.4 The objective differs from the reported metric

The headline metric throughout this literature, and in our own work, is
cost-per-unit, a ratio E[Σc]/E[TP], whereas the CMDP objective above minimises
total cost subject to a throughput floor. The two differ, and on our testbeds
they appear to diverge in a direction that matters. Interpolating linearly
between the CostMinimising and ShortestQueue baselines on bakery gives a marginal
cost of about $40 per additional unit against an average of about $139, which
would make cost-per-unit decreasing in throughput across the operating range. If
so, a total-cost minimiser has every incentive to sit at the throughput floor,
where the interpolated cost-per-unit (approximately $151) is worse than
ShortestQueue's $138.66, and the constrained optimum as posed would be dominated
on the metric we report. This is a plausibility argument from two points, not a
measurement, and we present it as motivation.

Its consequence, if it holds, is that any apparent success of the constrained
agent on cost-per-unit must be delivered by something other than the objective.
In the original submission a candidate is the checkpoint-selection rule.
Requiring at least 16 of 20 validation episodes to satisfy TP ≥ 18, with a
per-episode standard deviation of about 2.0 units, implies a mean throughput of
about 19.7, an effective floor well above the nominal 18. We note this here and
return to it in §7. The ratio-objective formulation it motivates is future work
rather than a claim of this paper.

## 4. Method

### 4.1 Lagrangian PPO and the augmented reward

The constrained problem is solved with PPO as the inner loop and a Gymnasium
wrapper that augments the per-step reward with penalties. The augmented reward
used is given by Eq. (3),

$$\tilde r_t = r_t - \lambda_U\, g_{U,t} - \lambda_T\, g_{T,t}\qquad(3)$$

where r_t is the environment reward of Eq. (1) rather than the bare cost term. In
the original submission both the abstract and the formulation section stated
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

$$g_{T,t} = \max\!\left(0,\; \frac{T_{\min}}{H} - \frac{\mathrm{departures}_t}{t}\right)\qquad\text{(cumulative-rate slack)}\qquad(4)$$

This differs from the constraint of Eq. (2). It compares a cumulative average
rate against the floor at every instant. In a flow shop whose flow time is a sizeable
fraction of the horizon, the cumulative rate is below the floor for a long prefix
of every episode, whatever the policy does, because the line is filling.
Measured crossing times under ShortestQueue are t = 102–207 on bakery and
t = 129–227 on electronics. A policy achieving TP = 50 on electronics, which
meets the stated constraint, does not cross until t ≈ 480, so it never satisfies
the training signal within the horizon. The utilisation slack was defined
analogously, g_{U,t} = Σ_{i∈F_fast} max(0, U_min − ū_{i,t}), with ū_{i,t} the
cumulative utilisation of server i up to step t; it carries the same fill-phase
bias, although servers become busy quickly and the effect is milder.

The one-sided dual update of Eq. (5),

$$\lambda \leftarrow \operatorname{clip}\!\left(\lambda + \eta\,\operatorname{mean}_t\, g_t,\; 0,\; \lambda_{\max}\right)\qquad(5)$$

which the original submission described as "a violation-driven penalty
accumulator with monotone growth, not true primal-dual ascent", turns this bias
into a structural outcome. The measured mean per-step g_T for ShortestQueue is
0.0003–0.0029 on bakery and 0.0010–0.0064 on electronics; at η_T = 4000 this is a
per-episode increment of 1–26 even for an oracle. Over 3,333 training episodes,
λ_T reaches its cap of 20,000 for any policy. For realistic electronics policies
(TP ≈ 35–46), the increment is 40–80 per episode and the cap is reached after
250–500 episodes, or 8–15% of training, consistent with where the original
submission observed it. Saturation therefore reflects the signal, not the policy.

### 4.3 Corrected slack

We replace the per-step signal with the signed pro-rata decomposition of the
episode Lagrangian, Eq. (6),

$$\mathrm{penalty}_t = \lambda_T\left(\frac{T_{\min}}{H} - \mathrm{departures}_t\right) + \lambda_U \sum_{i\in F_{\mathrm{fast}}}\left(U_{\min} - \mathrm{busy}_{i,t}\right)\qquad(6)$$

which sums over the episode to Eq. (7),

$$\lambda_T\left(T_{\min} - \mathrm{TP}(\tau)\right) + \lambda_U\, H \sum_{i\in F_{\mathrm{fast}}}\left(U_{\min} - \bar u_i(\tau)\right)\qquad(7)$$,

the Lagrangian of the constraint of Eq. (2) that is checked at validation and
test, where busy_{i,t} ∈ {0, 1} indicates whether server i is in service at step
t and ū_i(τ) is its realised utilisation over the episode. There is no warm-up
and no fill-phase bias; an action's contribution
depends only on the departures and busy time it causes. Dual variables are
updated once per episode from episode-level slack, hinged as before, so the
ratchet advances only on episodes that violate rather than growing without
bound. A signed (symmetric) variant is available and reported where relevant.
Multiplier units, initial values, learning rates and caps are unchanged from the
original submission, so results are directly comparable.

### 4.4 Evaluation of a stochastic policy

The original protocol evaluated with `deterministic=True`, that is, greedy action
selection. PPO optimises a stochastic policy, and for a constrained problem the
optimum may itself be randomised (§2.2). More immediately, if the learned
distribution is close to uniform, its argmax is a deterministic policy selected
by small and unstable differences in logits, and evaluating it measures those
differences rather than the policy. We therefore report both evaluation modes
throughout, select checkpoints under the same mode we report, and in §6.5 measure
directly how far the learned policies are from uniform. Protocol Amendment R1,
committed before any full-budget run of this paper, records the change and its
rationale.

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
T_min = 50. Service-time distributions and costs are in the FlexFlowSim-CPPO
configuration files.

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
compared against, a limitation discussed in §8. The original submission stated
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
ratios). Confidence intervals in Tables 2–4 are 95% two-sided t-intervals
across seeds, and in Table 1 across episodes, so the two are not directly
comparable. PPO is the Stable-Baselines3 implementation (Raffin et al., 2021) with
the hyperparameters of the original submission (learning rate 3×10⁻⁴,
n_steps 2048, batch 64, 10 epochs, γ = 0.99, GAE λ = 0.95, clip 0.2, entropy
coefficient 0.01). Dual variables: λ_U initialised at 5 with step 20 and cap 500;
λ_T initialised at 200 with step 4000 and cap 20,000.

Two protocol changes are recorded in `protocol_amendment_r1.md`, committed
before the full-budget runs it governs. First, both evaluation modes are computed
and reported for every checkpoint and every test, and checkpoint selection reads
the stochastic validation episodes so that selection and reporting share a
criterion. The original submission's argmax selection is still computed and
reported alongside. Second, full-budget training is executed in 400K-step
segments to fit the execution environment. Policy weights, optimiser state, dual
variables and the dual episode counter are carried across segment boundaries;
only PPO's in-flight rollout buffer is flushed, three times in 1.6M steps, or
about 0.4% of rollouts.

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

Four hypothesis tests are reported in §6.2. With a Bonferroni adjustment the
threshold for α = 0.05 is 0.0125; results are stated with and without it.

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
| shaped + cumulative-rate (original) | $152.22 ± 84.67 | 0.8% | $83.05 ± 2.93 | 68.0% | 5/5 |
| shaped + episode | $122.26 ± 13.46 | 0.0% | $78.64 ± 1.70 | 94.0% | 0/5 |
| cost + cumulative-rate | $155.18 ± 88.17 | 0.0% | $80.67 ± 1.89 | 74.4% | 5/5 |
| cost + episode (the CMDP of §3.3) | $155.35 ± 80.54 | 0.0% | $78.69 ± 2.46 | 94.4% | 0/5 |

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
including the original implementation, reproduces the original submission's
failure pattern under argmax evaluation, with $122–155 per unit and essentially
zero joint satisfaction, and every cell passes on the same checkpoints under
stochastic evaluation. The between-seed confidence intervals also collapse, from
±13–88 to ±1.7–2.9. §6.5 shows why: the learned distributions are close to
uniform, and the argmax of a near-uniform distribution varies arbitrarily
between seeds.

The slack bias damages the learned policy as well as the dual variables, on the
harder testbed. Even under stochastic evaluation, the cumulative-rate cells
remain $2–4 per unit worse and 20–26 percentage points lower in joint
satisfaction than their episode-slack counterparts at 400K, and the satisfaction
gap survives at full budget on electronics (§6.2). Once λ_T is pinned at its cap
the policy gradient is dominated by penalty avoidance, and the resulting policy
does not recover. On bakery the same saturation produces no measurable damage
(§6.2), so the claim is testbed-dependent.

The base reward has little effect. Shaped and cost-only cells are within
confidence intervals of each other in both evaluation modes. The misdescription
identified in §4.1 is a documentation error rather than a driver of the result.

### 6.2 Full-budget comparison

Two cells were run at the full pre-registered budget: the original
implementation (shaped reward, cumulative-rate slack) as control, and the
corrected formulation of §3.3 and §4.3 (cost reward, episode slack). Five seeds
each, 1.6M steps on electronics and 1.5M on bakery, checkpoint selection on
stochastic validation per Amendment R1, and both evaluation modes on the 50 test
episodes. Table 3 reports the results.

**Table 3.** Full-budget comparison. CI is the 95% two-sided t-interval across
seeds. "Sat." is joint constraint satisfaction over test episodes. References
(Table 1): ShortestQueue $73.25 / 100% and RoundRobin $78.90 / 86% on
electronics; ShortestQueue $138.66 / 96% and RoundRobin $137.55 / 88% on bakery.

| Testbed | Cell | stochastic CPU ± CI | stoch. sat. | val.-satisfied | λ_T at cap | argmax CPU ± CI | argmax sat. |
|---|---|---|---|---|---|---|---|
| electronics | corrected | $78.67 ± 2.32 | 94.4% | 5/5 | 0/5 | $162.94 ± 75.56 | 0.0% |
| electronics | original signal | $81.88 ± 2.07 | 75.6% | 5/5 | 5/5 | $196.04 ± 42.87 | 0.0% |
| bakery | corrected | $139.95 ± 3.35 | 92.4% | 5/5 | 0/5 | $165.66 ± 46.86 | 54.4% |
| bakery | original signal | $139.00 ± 2.28 | 90.8% | 5/5 | 5/5 | $180.37 ± 40.63 | 14.4% |

Welch's t-test, corrected versus original signal: electronics cost-per-unit
p = 0.021, joint satisfaction p = 0.002; bakery cost-per-unit p = 0.54, joint
satisfaction p = 0.73. Under the Bonferroni threshold of 0.0125 only the
electronics joint-satisfaction result is significant.

Four results follow. First, every seed in every cell satisfies the validation
criterion, including the unmodified original signal, which under the original
protocol satisfied it on 0 of 5 electronics seeds and 4 of 5 bakery seeds.
Evaluation mode alone reverses the original pass/fail verdict on electronics and
completes the set on bakery. Second, on electronics the slack correction is worth
18.8 points of joint satisfaction (p = 0.002, significant after adjustment) and
$3.21 per unit (p = 0.021, not significant after adjustment). Third, on bakery
the slack correction makes no measurable difference. The multiplier still
saturates on all five control seeds, so the mechanism of §4.2 is present, but on
the easier problem the saturated penalty does not damage the policy enough to
show in either metric. We report this as a null result, not as a partial
success.

Fourth, and most important for interpretation, the corrected agent does not
separate from stateless routing on cost-per-unit. On electronics its $78.67 is
within confidence intervals of RoundRobin's $78.90 and UniformRandom's $79.91
(one-sample t-tests against the baseline means, p = 0.80 and p = 0.21). Its joint
satisfaction of 94.4% is higher than RoundRobin's 86% (p = 0.020) and
UniformRandom's 78% (p = 0.002). On bakery no comparison against either stateless
rule reaches significance in either metric. ShortestQueue, a state-aware rule,
remains ahead of all of them: 7.4% cheaper per unit on electronics with 100%
constraint satisfaction.

A second feature of Table 3 is the between-seed dispersion. Under argmax
evaluation the electronics confidence intervals are ±43 to ±76; under stochastic
evaluation of the same checkpoints they are ±2.1 to ±2.3. The high seed-to-seed
instability that the original submission reported on electronics was, to a first
approximation, the variance of an argmax taken over near-uniform action
distributions (§6.5).

### 6.3 Archival re-evaluation of the original submission's checkpoints

The original submission's checkpoints were re-evaluated under Amendment R1
§A2.1 without retraining: V4, the one-sided Lagrangian of §4.2, and V6, a
PID-Lagrangian comparator with exponentially smoothed slack (Stooke et al.,
2020), on both testbeds and every archived seed (five for V4, four of five for
V6). We ask two questions. The first (Q1)
concerns the exact checkpoint the original submission selected and reported,
evaluated stochastically. The second (Q2) concerns the checkpoint that stochastic
selection would have chosen, and what it scores.

Applying the original argmax selection rule to the archived validation sweep
reproduces the originally selected checkpoint in 18 of 18 runs and the reported
test figures to the cent: V4 bakery $149.30 ± 27.73 and V4 electronics
$112.71 ± 13.46 are Tables 3 and 5 of the original submission. The pipeline
evaluated here is the one that produced those figures. Table 4 reports both
questions for all four archived runs.

**Table 4.** Archival re-evaluation. Q1 columns evaluate the originally selected
checkpoint in both modes; Q2 applies stochastic selection. n = 4 for V6 because
one seed's checkpoints were not archived.

| Run | n | Q1 argmax CPU ± CI (original) | Q1 sat. | Q1 stochastic CPU ± CI | Q1 sat. | Q2 stochastic CPU ± CI | Q2 sat. | Q2 val.-satisfied |
|---|---|---|---|---|---|---|---|---|
| V4 electronics | 5 | $112.71 ± 13.46 | 6% | $81.57 ± 1.93 | 77% | $80.36 ± 1.80 | 79% | 5/5 (original 0/5) |
| V6 electronics | 4 | $106.75 ± 23.87 | 22% | $83.93 ± 10.76 | 62% | $81.35 ± 5.68 | 76% | 3/4 |
| V4 bakery | 5 | $149.30 ± 27.73 | 69% | $140.32 ± 1.08 | 88% | $138.37 ± 1.52 | 80% | 5/5 |
| V6 bakery | 4 | $141.57 ± 22.71 | 74% | $140.60 ± 4.69 | 92% | $140.42 ± 5.55 | 83% | 4/4 |

The five electronics V4 checkpoints that the original submission reported as
failing systematically, at $113.31, $112.08, $95.38, $118.35 and $124.44, score
$83.13, $82.67, $79.88, $79.91 and $82.24 as stochastic policies, with joint
constraint satisfaction of 66–88%. The checkpoint files are unchanged. Had the
original protocol selected on stochastic validation, all five would have been
reported as `satisfied` at $80.36 ± 1.80. The original negative result on
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
budget, matching the original submission's Figure 3) and at 0.59–0.65M on bakery
(39–43%), and is flat thereafter. The difference in saturation time between
testbeds is what §4.2 predicts: the fill-phase gap is larger where flow time is a
larger fraction of the horizon and the floor sits closer to the achievable rate.
Under the corrected episode signal the multiplier rises slowly, advancing only on
episodes that violate, and ends between 2,525 and 7,858, never within a factor of
2.5 of the cap.

Selected checkpoints do not cluster: under stochastic selection they range from
100K to 1.6M in both cells on both testbeds, so we make no claim that the
original budget was too long or too short.

The magnitude argument of the original submission also needs correcting. It
compared the integrated penalty at saturation with "the unconstrained per-episode
cost reward (−$2,000 to −$4,000)" and concluded a ratio of 35–70×. That
comparison mixed units. Dollars never enter the reward; the per-episode base
reward in the units the agent optimises is −9 to −13, and the true ratio at
saturation is of order 10⁴. The qualitative claim, that the augmented objective
is dominated by violation avoidance, survives and is considerably stronger than
stated. It is also the most likely reason the policies of §6.5 remain close to
their initialisation.

### 6.5 Structure of the learned policies

For the selected checkpoint of every full-budget run we rolled out the stochastic
policy on ten test episodes and recorded, at every decision, the entropy of the
action distribution and its largest probability. Table 5 reports the entropy
normalised by ln(number of routes), the effective number of routes exp(H), and
the mean top-1 probability. A uniform router has normalised entropy 1.0, effective
routes equal to the number of routes, and top-1 probability 1/12 = 0.083 on
electronics or 0.25 on bakery.

**Table 5.** Action-distribution statistics of the selected checkpoints, ranges
across five seeds. Uniform reference: normalised entropy 1.00; effective routes
12 (electronics) or 4 (bakery); top-1 probability 0.083 or 0.25.

| Testbed | Cell | normalised entropy | effective routes | mean top-1 probability |
|---|---|---|---|---|
| electronics | corrected | 0.94–0.97 | 10.4–11.3 of 12 | 0.13–0.18 |
| electronics | original signal | 0.96–0.98 | 10.9–11.5 of 12 | 0.13–0.18 |
| bakery | corrected | 0.91–0.98 | 3.5–3.9 of 4 | 0.31–0.44 |
| bakery | original signal | 0.95–0.99 | 3.7–3.9 of 4 | 0.29–0.40 |

The learned policies are near-uniform routers with a modest state-dependent
tilt. On electronics the most probable route at a typical decision carries 13–18%
of the mass against 8.3% for uniform, and the marginal distribution over routes
across an episode has normalised entropy 0.94–0.99. This accounts for the results
of §6.2 and §6.3 in one stroke. Stochastic evaluation of a near-uniform router is
close to uniform-random routing, which Table 1 shows satisfies the constraints on
78% of episodes at $79.91 on electronics; the learned tilt buys about 16 further
points of satisfaction and no measurable cost. Greedy evaluation of a
near-uniform router selects whichever route has the largest of twelve nearly
equal logits, which differs from seed to seed and from checkpoint to checkpoint
and has no reason to be feasible; the wide argmax confidence intervals of Table 3
follow.

Two robustness checks. First, single-draw stochastic evaluation introduces
sampling variance of its own. Re-evaluating the electronics corrected seed 42
checkpoint five times with different sampling seeds gives cost-per-unit
79.11–81.63 (SD 0.96) and joint satisfaction 88–96% (SD 0.036); the seed-level
confidence intervals of Table 3 are wider than this, so the between-seed
comparison is not an artefact of within-run sampling noise. Second, the control
cell is, if anything, closer to uniform than the corrected cell, consistent with
§6.1: a saturated penalty term four orders of magnitude larger than the base
reward leaves the policy gradient with almost nothing to say about routing.

## 7. Discussion

### 7.1 Two artefacts

The original submission's electronics result decomposes into two artefacts,
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

The second artefact is an evaluation error: reporting the argmax of a policy that
is close to uniform. The argmax of such a policy is an essentially arbitrary
deterministic router, and the original submission's finding of severe
seed-to-seed instability on electronics was largely a measurement of that
arbitrariness. Of the two artefacts this is the one that reverses the reported
verdict, and §6.3 shows that it does so on the original checkpoints without any
retraining. It would have gone unnoticed had the first artefact not prompted a
re-examination of the pipeline.

### 7.2 What the method actually learned

Correcting both artefacts does not produce a competitive agent. It produces a
near-uniform router. On electronics the corrected policies match RoundRobin and
UniformRandom on cost-per-unit and beat them only on constraint reliability, by
8 and 16 points of joint satisfaction respectively. On bakery they match both
on every metric. ShortestQueue, which reads the queue state and requires no
training, is 7.4% cheaper per unit than the best learned policy on electronics
with 100% constraint satisfaction, and is indistinguishable from it on bakery.

The most likely reason is the one §4.2 and §6.4 quantify. With the penalty term
four orders of magnitude larger than the base reward, and with the constraint
satisfiable by any policy that spreads load, the policy gradient has no
incentive to move far from the maximum-entropy initialisation: spreading load
satisfies the constraint, and the cost signal that would distinguish good routing
from indifferent routing is too small to register. The entropy coefficient of
0.01, retained from the original submission for comparability, works in the same
direction. Whether a smaller entropy coefficient, a rescaled base reward, or a
ratio objective (§3.4) would let PPO learn state-aware routing is the obvious next
experiment and is outside this paper's scope.

The practical recommendation of the original submission therefore stands, for a
different reason. Tuned dispatching rules remain the baseline to beat on
problems of this size and structure. The constrained agent no longer fails to
satisfy its constraints; it satisfies them the way random routing does, and is
still more expensive than a rule that looks at the queues.

### 7.3 Deterministic deployment

Feasible deterministic policies exist on both testbeds; ShortestQueue is one.
What we did not obtain is a feasible deterministic policy *from the learned
agent* on electronics: greedy extraction from a near-uniform distribution yields
an arbitrary route, and no configuration we tested produced a learned policy
concentrated enough for its argmax to be reliable. In plants that cannot deploy a
stochastic controller, and there are good reasons for such a requirement,
including auditability and operator trust, the learned policies of this paper
are not deployable, and the correction described here does not change that. The
argmax column is reported throughout so that this remains visible.

### 7.4 Implications for evaluation practice

Two systematic reviews have criticised this field for weak baselines and
simulation-only validation. This paper suggests three further items for the
checklist. Where a constrained method is reported to fail its constraints, the
reader should be able to verify that the constraint optimised is the constraint
stated: the exact augmented per-step reward, the units of the slack signal, and
its measured value under a known-good reference policy should be reported.
Where a stochastic policy is evaluated, the evaluation mode should be stated and
both modes reported, together with a measure of how far the learned distribution
is from uniform. And where a learned router is compared against dispatching
rules, uniform-random and round-robin routing should be among them, evaluated
under the same protocol; without that control this paper would have reported a
recovery where there was only a near-random policy satisfying easy constraints.

## 8. Threats to validity

**Two testbeds.** Bakery and electronics differ simultaneously in action-space
size, stage count, capacity ratios and reward scale. We can attribute the
original failure to the implementation, because that is manipulated directly,
but we cannot attribute the difference in difficulty between testbeds to any
single structural feature. A controlled feature ablation remains outstanding.

**Five seeds.** Stochastic-mode confidence intervals are tight (±2.1–3.4 on
cost-per-unit at full budget), which makes n = 5 more defensible than it was
under argmax evaluation. It remains a small sample; the bakery null in §6.2 and
the null comparisons against stateless routing in §6.2 are failures to detect a
difference at this n, not evidence of equality. Four hypothesis tests were
performed; under Bonferroni adjustment one of the two significant electronics
results survives. One V6 seed was not archived, so the V6 rows of Table 4 are
n = 4.

**Single-draw stochastic evaluation.** Each validation and test episode samples
one action trajectory. The repeated-draw check in §6.5 bounds the resulting
variance at roughly a third of the between-seed interval on one checkpoint; it
was not repeated for every run.

**Selection bias in the archival comparison.** Q2 in §6.3 re-selects checkpoints
on the same stochastic validation episodes now used for reporting. This is the
protocol the amendment specifies, applied identically to every run, but it is a
different selection from the one the original submission used. The Q1 columns,
which evaluate the originally selected checkpoint, are the cleaner like-for-like
comparison, and the principal claim rests on them.

**The amendment was informed by the pilot.** The change to stochastic selection
was motivated by the 400K ablation, which was run under argmax selection. The
amendment was committed before the full-budget runs it governs, but it is not
independent of data from the same study.

**Reduced-budget ablation and omitted cells.** §6.1 is at 400K steps and the
two off-diagonal cells were not run at full budget; see §5.3 for the
justification. The attribution should be treated as established at 400K only.

**Segmented training.** Full-budget runs flush the rollout buffer three times per
run. We regard the effect as negligible but have not measured it directly.

**Simulation only.** As in the original submission, there is no physical
validation.

## 9. Conclusion

A negative result about constrained reinforcement learning for flow-shop
routing, reported in an earlier submission by the author, reflected the
instrumentation used to obtain it. A per-step cumulative-rate slack signal
guaranteed multiplier saturation for any policy, including an oracle, and greedy
evaluation of a near-uniform policy measured an arbitrary deterministic router
rather than the policy that was trained. Both artefacts are reproduced and then
removed on the original checkpoints. With both corrected, the method satisfies
its constraints on every seed of both testbeds. It does so, however, in the way
that random routing satisfies them: the learned policies are near-uniform
routers, indistinguishable from round-robin on cost-per-unit and 7.4% more
expensive than ShortestQueue on the harder testbed. The corrected negative result
is narrower than the original and, we think, more useful. The method did not
fail to meet its constraints; it failed to learn. Separating those two claims
required reporting the constraint actually implemented, the evaluation mode
actually used, and a random-routing control, and we would encourage all three
as routine practice.

## Data availability

FlexFlowSim-CPPO (Author, 2026b), the pre-registered protocol (`protocol.md`,
commit 391eb0d),
Protocol Amendment R1 (`protocol_amendment_r1.md`), the corrected wrapper
(`lagrangian_slack.py`), the ablation and archival runners, the policy-entropy
analysis, and all per-seed results, λ histories and per-episode test records are
available at [URL withheld for review].

## Funding

This research did not receive any specific grant from funding agencies in the
public, commercial, or not-for-profit sectors.

## Declaration of competing interest

The author declares no known competing financial interests or personal
relationships that could have appeared to influence the work reported in this
paper.

## Declaration of generative AI use

During the preparation of this work the author used Claude (Anthropic) to audit
the original submission's source code against its text, to implement the
corrected slack wrapper and the evaluation and analysis scripts, to execute the
re-evaluation and re-training runs described in §5–6, and to draft and edit the
manuscript. All code was reviewed by the author, every reported figure was
recomputed from the archived result files, and all scientific claims and
interpretations are the author's own. The author takes full responsibility for
the content of the article.

## CRediT authorship contribution statement

**[Author name]:** Conceptualization, Methodology, Software, Investigation,
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
§6.3 by archival_reeval.py; and the stateless baselines, action-distribution
statistics and repeated-draw check of §5.2 and §6.5 by
analysis_stateless_and_entropy.py. The pre-committed protocol is protocol.md
(commit 391eb0d); its amendment is protocol_amendment_r1.md; deviations are in
protocol_deviations.md. All per-seed summaries, validation caches, λ histories
and per-episode test records are under results_r1/.

### A.2 Hyperparameters

PPO inner loop (Stable-Baselines3, MlpPolicy with two hidden layers of 64 tanh
units; defaults unless noted): learning rate 3 × 10⁻⁴; n_steps 2048; batch size
64; n_epochs 10; γ = 0.99; GAE λ = 0.95; clip range 0.2; entropy coefficient
0.01; value-function coefficient 0.5; max gradient norm 0.5. Lagrangian outer
loop, identical in both slack modes: U_min = 0.50; T_min = 18 (bakery) or 50
(electronics); λ_U initial 5, step η_U = 20, cap 500; λ_T initial 200, step
η_T = 4000, cap 20,000. The cumulative-rate mode applies a 100-step warm-up; the
episode mode applies none (§4.3). Multipliers are updated once per episode in
both modes.

### A.3 Budgets and seeds

Training seeds [42, 7, 2024, 123, 999] throughout. Mechanism ablation (§6.1):
400K timesteps per run, four cells, electronics only. Full-budget comparison
(§6.2): 1.6M timesteps on electronics and 1.5M on bakery, two cells, executed in
400K-step segments (§5.3). Checkpoints every 100K timesteps in all runs.
Validation seeds [10000, 10020); test seeds [11000, 11050); both disjoint from
training. Stateless baselines (Table 1) are evaluated on the 50 test seeds, with
UniformRandom's action sampling seeded by the episode seed. The
action-distribution statistics of Table 5 use the first ten test seeds.

### A.4 Evaluation protocol

Every checkpoint is evaluated on the 20 validation episodes in both modes:
argmax (`deterministic=True`) and stochastic (one action sample per step from
the policy distribution, one trajectory per episode). Selection follows §5.3 on
the stochastic validation results; argmax selection is also computed and stored.
The selected checkpoint is evaluated on the 50 test episodes in both modes.
Cost-per-unit is total episode cost divided by episode departures, averaged over
episodes. Joint satisfaction is the fraction of episodes in which every
constrained server's realised utilisation is at least U_min and departures are
at least T_min.

### A.5 Statistical tests

Between-cell comparisons in §6.2 use Welch's unequal-variance t-test on the five
seed-level means. Comparisons against a stateless baseline use a one-sample
t-test of the five seed-level means against the baseline's 50-episode mean.
Confidence intervals across seeds use the t-distribution with four degrees of
freedom; across episodes, the normal approximation. Four between-cell tests are
reported; the Bonferroni-adjusted threshold for α = 0.05 is 0.0125.

### A.6 Hardware and runtime

R1 training and evaluation ran on a two-vCPU Linux container at approximately
800–1,000 environment steps per second. A 400K-step segment takes 7–8 minutes;
a full 1.6M-step run about 30 minutes plus 10–15 minutes of dual-mode
validation over 16 checkpoints. The full R1 matrix (20 pilot runs, 20 full-budget
runs, 18 archival re-evaluations of 279 checkpoints, baselines and analyses)
consumed approximately 16 compute-hours. The original submission's runs were
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
| V4, V6 | Original submission's one-sided Lagrangian variant; its PID-Lagrangian comparator |
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
| ū_{i,t}, ū_i(τ) | Cumulative utilisation of server i up to step t; realised utilisation over episode τ |
| busy_{i,t} | In-service indicator of server i at step t |
| departures_t | Jobs completed at step t |
| H | Episode horizon, 480 one-minute steps |
| κ, w_c, w_t, w_w | Reward scale and shaping weights of Eq. (1) |
| π | Policy |
