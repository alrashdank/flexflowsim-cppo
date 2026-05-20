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

