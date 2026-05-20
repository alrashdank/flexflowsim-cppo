# FlexFlowSim-CPPO

Companion code for the paper **"Failure Modes of Constrained PPO in Multi-Server Flow-Shop Routing: A Benchmark Against Dispatching Rules"** (Alrashdan, 2026).

This repository contains the simulator, training and evaluation scripts, pre-committed experimental protocol, and per-seed results for the V4 and V6 multi-seed experiments reported in the paper. Trained PPO checkpoints for the four main reported tables (Tables 3, 5, 7, 8) are included so that the reported numbers can be reproduced without re-running training.

## Citation

If you use this code, please cite:

```
Alrashdan, K. R. (2026). Failure Modes of Constrained PPO in Multi-Server
Flow-Shop Routing: A Benchmark Against Dispatching Rules.
[Journal name, volume/pages once accepted].
```

## License

MIT License — see `LICENSE`.

---

# Paper 6 — Fresh-Folder Experimental Package

Self-contained directory for running the final experimental package for Paper 6
("Failure Modes of Constrained PPO in Multi-Server Flow-Shop Routing: A Benchmark Against Dispatching Rules").

Two ways to run: via the **FlexFlowSim dashboard** (recommended — has the Lagrangian
tab integrated alongside Configure / Train / Evaluate / etc.) or via the
**command-line launcher** for terminal/SSH use.

## Quick start (web — recommended)

```bash
# 1. Install dependencies (Python 3.9–3.12 recommended; 3.13+ may have edge cases)
py -m pip install -r requirements.txt

# 2. Initialise git and commit the protocol BEFORE running any experiments
git init
git add protocol.md
git commit -m "Pre-commit Paper 6 experimental protocol"

# 3. Launch the dashboard
py -m streamlit run app.py
```

A browser opens at `http://localhost:8501`. The dashboard has the usual
**Configure / Train / Evaluate / Simulate / Sensitivity / Compare** tabs plus a new
**🧪 Lagrangian** tab in the sidebar.

The Lagrangian tab has three sub-tabs:

- **▶ Run** — checkboxes for each pipeline stage, Quick-mode toggle for a 10-minute
  smoke test, live output streaming
- **📊 Results** — interactive tables of final results with download buttons
- **📋 Protocol** — renders `protocol.md` so you can review the pre-committed
  selection rules without leaving the UI

The launcher is **resumable**. Already-complete stages get a ✅ marker and are skipped.

## Quick start (command-line)

If you prefer the terminal:

```bash
py -m pip install -r requirements.txt

# Quick smoke test (~10 minutes)
py run_all.py --quick

# Full run (~40-55 CPU-hours, resumable)
py run_all.py

# Run only specific stages
py run_all.py --stages baselines bakery_v4
```

## What the launcher does

`run_all.py` runs seven stages in order. Each stage writes to `results/`
and is skipped if already complete (so you can safely Ctrl-C and resume).

| Stage | What it runs | Time |
|-------|-------------|------|
| `baselines` | LeastUtilised, ShortestQueue, CostMinimising, FastServerFirst on both testbeds, 50 episodes each | ~30 min |
| `bakery_v4` | V4 (Violation-Driven Lagrangian PPO) on bakery, 5 seeds × 1.5M timesteps, with checkpoint validation and 50-episode test eval | 5–7 hours |
| `bakery_unconstrained` | Unconstrained PPO on bakery, same protocol | 4–5 hours |
| `sensitivity` | T_min × U_min 9-cell sweep on bakery, single seed | 10–15 hours |
| `electronics_v4` | V4 on electronics, 5 seeds × 1.6M | 10–16 hours |
| `electronics_unconstrained` | Unconstrained PPO on electronics | 8–12 hours |
| `aggregate` | Combine all CSVs into final results tables with mean ± 95% CI | <1 min |

## Running individual stages

```bash
# Just baselines:
python run_all.py --stages baselines

# Just bakery (both V4 and unconstrained):
python run_all.py --stages bakery_v4 bakery_unconstrained

# Aggregate after partial completion:
python run_all.py --stages aggregate
```

## Quick mode

```bash
python run_all.py --quick
```

Runs 1 seed × 200K timesteps for the RL stages, 3 sensitivity cells (instead of 9),
and the full baselines/aggregation. Total: ~10 minutes. Use this once to confirm
the pipeline works on your machine before committing to overnight runs.

**Important:** Do NOT use `--quick` for paper results. Quick mode is for verification only.

## Files in this folder

### Core simulator (do not modify)

| File | Purpose |
|------|---------|
| `env.py` | FlexFlowSim simulator (multi-stage, multi-server flow shop) |
| `baselines.py` | Dispatching-rule policies |
| `pilot_constrained_v4_auto.py` | V4 Lagrangian wrapper |
| `configs/bakery_bk50.json` | Bakery testbed config (4 actions, BK50 calibrated) |
| `configs/electronics_3stage.json` | Electronics testbed config (12 actions) |
| `requirements.txt` | Python dependencies |

### Experiment package

| File | Purpose |
|------|---------|
| `protocol.md` | **Pre-committed experimental protocol. Commit BEFORE running.** |
| `app.py` | **FlexFlowSim dashboard with integrated 🧪 Lagrangian tab** |
| `run_all.py` | Command-line launcher — same pipeline, terminal interface |
| `run_experiment.py` | Trains V4 with checkpoint validation + test evaluation |
| `run_unconstrained.py` | Unconstrained PPO baseline (same protocol, no Lagrangian) |
| `run_baselines.py` | Dispatching-rule baselines (50 episodes each) |
| `run_sensitivity.py` | T_min × U_min 9-cell wrapper |
| `aggregate_results.py` | Combines all per-seed CSVs into final result tables |

## What goes in `results/` after a full run

```
results/
├── baselines/
│   ├── bakery/            (4 baseline CSVs + summary.csv)
│   └── electronics/
├── bakery_v4/             (5 per-seed CSVs + summary.csv + checkpoints)
├── bakery_unconstrained/
├── electronics_v4/
├── electronics_unconstrained/
├── sensitivity/           (9 sub-folders, one per (T_min, U_min) cell)
├── final_results_bakery.csv
├── final_results_electronics.csv
└── final_sensitivity.csv
```

The three `final_*.csv` files are the headline output. Each row is one method
with mean ± 95% CI on throughput, cost-per-unit, total cost, and constraint
satisfaction rates.

## What to do if something fails

1. **A single seed crashes** — the launcher continues with the next seed. Note
   the failure in `protocol_deviations.md` (create this file alongside
   `protocol.md`). Per protocol §11, n=4 with one documented failure is more
   defensible than n=5 with one silently re-run.

2. **A stage times out or you Ctrl-C** — re-run `python run_all.py`. Already-complete
   stages are skipped automatically.

3. **Out of disk space** — checkpoints are ~150 KB each, ~30 MB total for the
   full pipeline. Should be fine on any modern machine.

4. **Dependency error on import** — run `pip install -r requirements.txt` and
   retry.

## When all runs finish

Send these files back to me:

1. `results/final_results_bakery.csv`
2. `results/final_results_electronics.csv`
3. `results/final_sensitivity.csv`
4. `results/baselines/*/summary.csv`
5. Any `protocol_deviations.md` entries

That's enough for me to regenerate the paper with statistical evidence.

You do NOT need to send the per-seed CSVs or checkpoints unless I ask.

## Compute notes

The runtimes above assume a single CPU core. If you have multiple cores, you can
parallelise across seeds by running `run_experiment.py` with `--seeds <N>` in
separate terminals:

```bash
# Terminal 1:
python run_experiment.py --testbed bakery --method v4 --seeds 42 --output-dir results/bakery_v4

# Terminal 2:
python run_experiment.py --testbed bakery --method v4 --seeds 7 --output-dir results/bakery_v4

# ... etc for seeds 2024, 123, 999
```

If you have a GPU, PPO will use it automatically via Stable-Baselines3. Roughly
2-3× speedup expected.
