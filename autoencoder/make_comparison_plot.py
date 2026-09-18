import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

with open(os.path.join(RESULTS_DIR, "option_a_results.json")) as f:
    r = json.load(f)

# --- Panel 1: the headline supervised comparison (the actual ask) ---
labels = ["From-scratch trunk\n+ 30-min supervised\nfine-tune", "MAE-pretrained trunk\n+ 30-min supervised\nfine-tune\n(Option A, this run)"]
precision = [r["scratch_supervised"]["precision"], r["mae_supervised"]["precision"]]
recall = [r["scratch_supervised"]["recall"], r["mae_supervised"]["recall"]]
f1 = [r["scratch_supervised"]["f1"], r["mae_supervised"]["f1"]]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

x = np.arange(2)
w = 0.25
ax = axes[0]
ax.bar(x - w, precision, w, label="precision", color="#4a5568")
ax.bar(x, recall, w, label="recall", color="#3182ce")
ax.bar(x + w, f1, w, label="F1", color="#2f855a")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0, 1.0)
ax.set_ylabel("score (held-out ~7.6 min of real Chelten clicks)")
ax.set_title("Does Option A pretraining help the 30-min fine-tune?")
ax.legend(loc="lower right", fontsize=9)
for i, (p, rc, f) in enumerate(zip(precision, recall, f1)):
    ax.text(i - w, p + 0.02, f"{p:.3f}", ha="center", fontsize=8)
    ax.text(i, rc + 0.02, f"{rc:.3f}", ha="center", fontsize=8)
    ax.text(i + w, f + 0.02, f"{f:.3f}", ha="center", fontsize=8)

# --- Panel 2: all four numbers side by side on a common precision/recall-ish axis ---
# weak-supervision numbers use a different metric basis (recall_conditional / recall_overall,
# per-BEAT rather than per-candidate) -- shown separately, clearly labeled, not merged into
# the same bars as the supervised precision/recall to avoid a misleading direct comparison.
ax2 = axes[1]
methods = [
    "ShapeMIL baseline\n(from-scratch, weak\nsupervision only)",
    "MAE-pretrained\n(weak supervision\nonly, blind)",
    "From-scratch +\n30-min supervised",
    "MAE-pretrained +\n30-min supervised",
]
# from-scratch weak baseline @thr=0.5 from the separate baseline run
precision_all = [0.804, r["weak_mae"]["precision"], r["scratch_supervised"]["precision"], r["mae_supervised"]["precision"]]
recall_all = [0.947, r["weak_mae"]["recall_conditional"], r["scratch_supervised"]["recall"], r["mae_supervised"]["recall"]]
recall_kind = ["recall_conditional\n(per-beat)", "recall_conditional\n(per-beat)", "recall\n(per-candidate)", "recall\n(per-candidate)"]

x2 = np.arange(4)
ax2.bar(x2 - 0.15, precision_all, 0.3, label="precision", color="#4a5568")
ax2.bar(x2 + 0.15, recall_all, 0.3, label="recall (see labels)", color="#3182ce")
ax2.set_xticks(x2)
ax2.set_xticklabels(methods, fontsize=8)
ax2.set_ylim(0, 1.05)
ax2.set_title("All four approaches (note: recall metric basis differs\nbetween weak-supervision and supervised columns)")
ax2.legend(loc="lower center", fontsize=9)
for i, k in enumerate(recall_kind):
    ax2.text(i, 1.0, k, ha="center", fontsize=6.5, color="#666")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "option_a_comparison.png"), dpi=150)
print("saved plot")
