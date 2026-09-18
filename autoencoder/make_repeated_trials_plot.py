import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

with open(os.path.join(RESULTS_DIR, "repeated_trials_results.json")) as f:
    r = json.load(f)

trials = r["trials"]
n = len(trials)
idx = np.arange(1, n + 1)
f1_scratch = np.array([t["scratch"]["f1"] for t in trials])
f1_mae = np.array([t["mae"]["f1"] for t in trials])
diff = f1_mae - f1_scratch

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# --- Panel 1: paired F1 per trial ---
ax = axes[0]
w = 0.35
ax.bar(idx - w / 2, f1_scratch, w, label="from-scratch", color="#4a5568")
ax.bar(idx + w / 2, f1_mae, w, label="MAE-pretrained", color="#3182ce")
ax.set_xlabel("trial (different random 7.5-min held-out window each time)")
ax.set_ylabel("F1 on that trial's held-out window")
ax.set_title(f"F1 per trial, {n} independent random test windows")
ax.set_xticks(idx)
ax.set_ylim(0, 1.05)
ax.legend(loc="lower left", fontsize=9)

# --- Panel 2: box plot of F1 distributions ---
ax2 = axes[1]
bp = ax2.boxplot([f1_scratch, f1_mae], tick_labels=["from-scratch", "MAE-pretrained"],
                  patch_artist=True, widths=0.5)
for patch, color in zip(bp["boxes"], ["#4a5568", "#3182ce"]):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)
for i, vals in enumerate([f1_scratch, f1_mae], start=1):
    xs = np.random.default_rng(0).normal(i, 0.04, size=len(vals))
    ax2.scatter(xs, vals, color="black", s=18, zorder=3, alpha=0.6)
ax2.set_ylabel("F1 across all 15 trials")
ax2.set_title(f"F1 spread across {n} trials\nmean±std: scratch {f1_scratch.mean():.3f}±{f1_scratch.std():.3f}, "
              f"MAE {f1_mae.mean():.3f}±{f1_mae.std():.3f}")
ax2.set_ylim(0, 1.05)

# --- Panel 3: paired difference (MAE - scratch) per trial ---
ax3 = axes[2]
colors = ["#2f855a" if d > 0 else "#c53030" for d in diff]
ax3.bar(idx, diff, color=colors)
ax3.axhline(0, color="black", linewidth=0.8)
ax3.axhline(diff.mean(), color="#666", linestyle="--", linewidth=1,
            label=f"mean diff = {diff.mean():+.3f}")
ax3.set_xlabel("trial")
ax3.set_ylabel("MAE F1 - from-scratch F1")
n_mae_wins = int((diff > 0).sum())
n_scratch_wins = int((diff < 0).sum())
ax3.set_title(f"Per-trial paired difference\nMAE better in {n_mae_wins}/{n}, from-scratch better in {n_scratch_wins}/{n}")
ax3.set_xticks(idx)
ax3.legend(loc="lower right", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "repeated_trials_comparison.png"), dpi=150)
print("saved plot")
