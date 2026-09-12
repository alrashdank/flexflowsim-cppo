#!/usr/bin/env python3
"""Figure 1: throughput multiplier over training under the two slack signals.

Reads the archived per-episode multiplier histories of the full-budget cells
(results_r1/{bakery,electronics}/{shaped-cumrate,cost-episode}/seed_*/
lambda_history.csv) and draws both testbeds on a log scale, five seeds per cell.
Same house style as make_fig_r2.py (Figure 2) so the two figures match.

The cells are labelled as the manuscript labels them - "original signal" and
"corrected" - and not by any publication status.
"""
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Liberation Serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8a85", "axes.labelcolor": "#2b2b28",
    "xtick.color": "#5c5c58", "ytick.color": "#5c5c58",
    "axes.grid": True, "grid.color": "#e6e6e2", "grid.linewidth": 0.6,
})

H = 480                     # minutes per episode, the paper's step convention
CAP = 20000.0
CELLS = [("shaped-cumrate", "#d9603b", "original signal (cumulative-rate slack)"),
         ("cost-episode",   "#3b7fd9", "corrected (episode-level slack)")]
TESTBEDS = [("bakery", "(a) bakery, 1.5M steps"), ("electronics", "(b) electronics, 1.6M steps")]

fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.7), dpi=200, sharey=True)
for ax, (tb, title) in zip(axes, TESTBEDS):
    for cell, colour, label in CELLS:
        for i, f in enumerate(sorted(glob.glob(f"results_r1/{tb}/{cell}/seed_*/lambda_history.csv"))):
            rows = list(csv.DictReader(open(f)))
            lam = [float(r["lam_tp"]) for r in rows]
            steps = [(k + 1) * H / 1e6 for k in range(len(lam))]
            ax.plot(steps, lam, color=colour, lw=1.0, alpha=0.85, label=label if i == 0 else None)
    ax.axhline(CAP, color="#8a8a85", lw=0.8, ls=(0, (3, 3)))
    ax.text(0.02, CAP * 1.12, "cap 20,000", fontsize=7.5, color="#5c5c58", transform=ax.get_yaxis_transform())
    ax.set_yscale("log")
    ax.set_xlabel("training steps (millions)")
    ax.set_title(title, loc="left", fontsize=9, fontweight="bold", color="#2b2b28")
axes[0].set_ylabel(r"$\lambda_T$ (throughput multiplier, log scale)")
axes[0].set_ylim(150, 40000)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(rect=(0, 0.04, 1, 1))
os.makedirs("/home/claude/draft", exist_ok=True)
fig.savefig("/home/claude/draft/fig_lambda_T.png", bbox_inches="tight")
fig.savefig("/home/claude/draft/fig_lambda_T.pdf", bbox_inches="tight")
print("wrote fig_lambda_T.png / .pdf")
