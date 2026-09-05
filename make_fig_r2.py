"""Figure 2: the Amendment R2 cell (symmetric, reward-scaled dual update), electronics, 5 seeds.
Left: lambda_T per episode (linear scale). Right: throughput slack TP - T_min per episode,
smoothed over 50 episodes. Direct labels; validated palette slot 3 (#1baf7a)."""
import csv, glob, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8a85", "axes.labelcolor": "#2b2b28", "xtick.color": "#5c5c58", "ytick.color": "#5c5c58", "axes.grid": True, "grid.color": "#e6e6e2", "grid.linewidth": 0.6})
C = "#1baf7a"; H = 480; T_MIN = 50
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.7, 2.6), dpi=200)
files = sorted(glob.glob("results_r2/electronics/cost-episode-sym/seed_*/lambda_history.csv"))
alphas = [0.9, 0.75, 0.6, 0.45, 0.35]
for f, al in zip(files, alphas):
    rows = list(csv.DictReader(open(f)))
    ep = np.array([int(r["episode"]) for r in rows]) * H / 1e6
    lt = np.array([float(r["lam_tp"]) for r in rows]); slack = -np.array([float(r["tp_gap"]) for r in rows]) * H
    a1.plot(ep, lt, color=C, lw=0.7, alpha=al)
    k = 50; sm = np.convolve(slack, np.ones(k) / k, mode="valid")
    a2.plot(ep[k - 1:], sm, color=C, lw=1.2, alpha=al)
a1.set_xlabel("training steps (millions)"); a1.set_ylabel(r"$\lambda_T$ (reward units per unit throughput)")
a1.axhline(0.1, color="#8a8a85", lw=0.8, ls=(0, (3, 3))); a1.text(1.62, 0.1, "shadow-price\nestimate 0.1", va="center", ha="left", fontsize=7.5, color="#5c5c58")
a1.set_xlim(0, 1.62); a1.set_ylim(0, 1.55)
a2.axhline(0, color="#8a8a85", lw=0.8); a2.text(1.62, 0, r"$T_{\min}$", va="center", ha="left", fontsize=7.5, color="#5c5c58")
a2.axhline(4.4, color="#8a8a85", lw=0.8, ls=(0, (3, 3))); a2.text(1.62, 4.4, "uniform-random\nrouting (+4.4)", va="center", ha="left", fontsize=7.5, color="#5c5c58")
a2.set_xlabel("training steps (millions)"); a2.set_ylabel("TP $-$ $T_{\\min}$ per training episode\n(50-episode moving average)")
a2.set_xlim(0, 1.62); a2.set_ylim(-4, 8)
for a, lab in ((a1, "(a)"), (a2, "(b)")): a.text(-0.18, 1.02, lab, transform=a.transAxes, fontsize=9, fontweight="bold")
fig.tight_layout(w_pad=3.5)
fig.savefig("/home/claude/draft/fig_symmetric.png", bbox_inches="tight"); fig.savefig("/home/claude/draft/fig_symmetric.pdf", bbox_inches="tight")
print("saved")
