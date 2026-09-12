Khaled R. Alrashdan  
Department of Manufacturing Engineering Technology  
College of Technological Studies  
Public Authority for Applied Education and Training (PAAET), Kuwait  
kr.alrashdan@paaet.edu.kw

12 September 2026

The Editors  
*International Journal of Modelling and Simulation*

Dear Editors,

I am submitting for your consideration a Research Article entitled "Objective,
constraint and evaluation alignment in constrained reinforcement learning for
multi-server flow-shop routing".

The paper is a simulation study of what happens when the links in a constrained
learning pipeline disagree with one another. Reinforcement learning is now
widely proposed for routing and sequencing in production systems, and two recent
systematic reviews have criticised the field for weak baselines and
simulation-only validation. This work identifies a class of problem upstream of
both, in the instrumentation of the constrained formulation itself, and shows
that it is capable of producing a reported result that is a property of the
pipeline rather than of the method.

Using Lagrangian proximal policy optimisation (PPO) on two discrete-event
flow-shop models with 4 and 12 routes, the study measures three such breaks.
Translating an episode-level throughput constraint into a per-step penalty on the
cumulative rate is shown, first by arithmetic on the slack measured under a fixed
dispatching rule and then empirically, to drive the multiplier to its cap for
that rule as well as for every trained policy: 10 of 10 seeds saturate with that
signal and 0 of 10 with an episode-level slack. Evaluating near-uniform
stochastic policies by their mode is shown to measure a router selected by
unstable logit differences, and correcting that alone reverses the reported
verdict on the archived checkpoints without retraining. With both corrected, the
agent satisfies the constraint criterion on every seed yet is not distinguishable
from round-robin routing on cost per unit under a paired hierarchical bootstrap.
A final configuration with dual multipliers on the scale of the reward makes the
dual behave as a dual, and the resulting policy moves towards the throughput
floor exactly as a total-cost objective directs, where it is no better on the
reported metric than the stateless rules. Across the seven load-spreading policies
studied, total episode cost varies by 4.6% while throughput varies by 15.5%, so a
total-cost formulation is the wrong instrument for a cost-per-unit objective
however well its dual behaves.

I believe the work suits the journal on three grounds. It is a simulation study
of a manufacturing system, built on an open discrete-event simulator that is
released with the paper. Its contribution is methodological and transferable:
the diagnostics it proposes, in particular reporting the measured slack under a
known-good reference policy and the ratio of constraint terms to objective, cost
nothing to compute and would have exposed each break immediately. And it reports
a carefully quantified negative result, with pre-committed protocols, disjoint
seed ranges, tuned dispatching baselines including uniform-random and
round-robin controls, and uncertainty propagated through a paired bootstrap with
multiplicity adjustment.

The simulator, all protocol documents including a recorded deviation, every
analysis script and all per-seed and per-episode result records are openly
available at https://github.com/alrashdank/flexflowsim-cppo. The "initial study"
whose result the paper re-examines is my own constrained-learning study, which
has not been published or submitted anywhere; its protocol, checkpoints and
result files are held in that repository and are cited as such. The two
flow-shop instances were introduced in an earlier, published study of mine,
cited in the manuscript [Alrashdan KR. J King Saud Univ Eng Sci. 2026;38(7):55],
which addressed disruption robustness rather than constraints or evaluation
mode; the present work shares those instances and the simulator with it but no
experiments, results or text.

The manuscript is original, is not under consideration elsewhere, and has not
been published previously. I have no competing interests to declare, and the
work involved no human subjects.

Thank you for your consideration.

Yours sincerely,

Khaled R. Alrashdan
