# Supplementary Material S2: anonymised code and results snapshot

This is a snapshot of the companion repository (branch `ablation-slack-fix`, anonymised for
double-blind review) at the state used for the submitted manuscript.

Contents
- Simulator, wrappers and runners: `env.py`, `baselines.py`, `pilot_constrained_v4_auto.py`
  (original one-sided wrapper), `lagrangian_slack.py` (corrected and control slack modes, symmetric
  option), `run_ablation_2x2.py` (ablation, full-budget and Amendment R2 cells), `archival_reeval.py`.
- Analysis: `analysis_stateless_and_entropy.py`, `analysis_r2.py`, `analysis_r2_stats.py`,
  `analysis_r2_tables.py`, `make_fig_r2.py`.
- Protocol documents: `protocol.md` (pre-registered, original study), `protocol_amendment_r1.md`,
  `protocol_amendment_r2.md`, `protocol_deviations.md`.
- Results: `results/` (the original submission's archived checkpoints, all seeds, used for the
  archival re-evaluation of Section 6.3); `results_ablation/` (400K ablation summaries, lambda
  histories, validation caches, selected checkpoints); `results_r1/` (full-budget summaries, lambda
  histories, validation caches, per-episode test records, selected checkpoints; `results_r1/r2/`
  holds the seeded per-episode records, bootstrap outputs, argmax-stability metrics and severity
  re-scoring); `results_r2/` (Amendment R2 cell: summaries, lambda histories, validation caches,
  per-episode records, selected checkpoints).
- Intermediate (non-selected) checkpoints of the R1 and R2 runs are omitted to respect the 100 MB
  limit; they will be available in the public repository on publication.

Reproducing the tables from the archived records (no training required)
    python3 analysis_r2_tables.py            # Tables 2, 3, 5 (seeded re-evaluation vs pipeline)
    python3 analysis_r2_stats.py electronics # Table 4 (bootstrap), Table 7 (severity), electronics
    python3 analysis_r2_stats.py bakery
    python3 analysis_r2_stats.py electronics r2   # Table 4 block and Table 7 column for the symmetric cell
    python3 run_ablation_2x2.py --aggregate-only --outdir results_r1/electronics --testbed electronics
Re-running the full-budget cells: see `r1_tick.sh`, `r2_tick.sh` (segmented, resumable).
Environment: Python 3.11, stable-baselines3, gymnasium, simpy, numpy, torch (CPU).
