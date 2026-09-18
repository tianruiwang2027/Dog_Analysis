import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

with open(os.path.join(RESULTS_DIR, "unet_results.json")) as f:
    u = json.load(f)

STAGE1_CEILING = 157 / 309  # fraction of held-out window's real clicks Stage-1 even proposes a candidate for

rows = {
    "From-scratch trunk\n+ 30-min supervised\n(candidate classifier)": dict(
        precision=0.7972972972972973, recall_conditional=0.7515923566878981),
    "MAE-pretrained trunk\n+ 30-min supervised\n(candidate classifier)": dict(
        precision=0.8033707865168539, recall_conditional=0.910828025477707),
}

names, precisions, recalls_overall, f1s_overall, recalls_raw = [], [], [], [], []
for name, r in rows.items():
    recall_overall = r["recall_conditional"] * STAGE1_CEILING
    f1_overall = 2 * r["precision"] * recall_overall / (r["precision"] + recall_overall)
    names.append(name)
    precisions.append(r["precision"])
    recalls_overall.append(recall_overall)
    recalls_raw.append(r["recall_conditional"])
    f1s_overall.append(f1_overall)

names.append("1D U-Net\n(this experiment,\nno Stage-1 needed)")
precisions.append(u["held_out_result"]["precision"])
recalls_overall.append(u["held_out_result"]["recall"])
recalls_raw.append(u["held_out_result"]["recall"])  # same number -- U-Net has no conditional/overall gap
f1s_overall.append(u["held_out_result"]["f1"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

x = np.arange(len(names))
w = 0.25
ax = axes[0]
ax.bar(x - w, precisions, w, label="precision", color="#4a5568")
ax.bar(x, recalls_overall, w, label="recall (overall, vs ALL real clicks)", color="#3182ce")
ax.bar(x + w, f1s_overall, w, label="F1 (overall)", color="#2f855a")
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=8.5)
ax.set_ylim(0, 1.0)
ax.set_ylabel("score on held-out ~7.6 min (all 309 real clicks)")
ax.set_title("Fair comparison: every real click counts,\nnot just the ones Stage-1 proposed a candidate for")
ax.legend(loc="upper left", fontsize=8)
for i, (p, rc, f) in enumerate(zip(precisions, recalls_overall, f1s_overall)):
    ax.text(i - w, p + 0.02, f"{p:.3f}", ha="center", fontsize=7.5)
    ax.text(i, rc + 0.02, f"{rc:.3f}", ha="center", fontsize=7.5)
    ax.text(i + w, f + 0.02, f"{f:.3f}", ha="center", fontsize=7.5)

ax2 = axes[1]
ax2.bar(x, recalls_raw, 0.5, color=["#a0aec0", "#a0aec0", "#3182ce"])
ax2.set_xticks(x)
ax2.set_xticklabels(names, fontsize=8.5)
ax2.set_ylim(0, 1.0)
ax2.set_title("Why this matters: the candidate classifiers' reported\n\"recall\" only ever covered candidates Stage-1 proposed\n"
              f"(Stage-1 ceiling on this window: {STAGE1_CEILING*100:.1f}% of real clicks). The\nU-Net has no such ceiling -- it sees the raw signal directly.")
ax2.set_ylabel("recall as originally reported (candidate classifiers)\nvs. as directly achieved (U-Net)")
for i, v in enumerate(recalls_raw):
    ax2.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
ax2.axhline(STAGE1_CEILING, color="#c53030", linestyle="--", linewidth=1)
ax2.text(2.15, STAGE1_CEILING + 0.015, "Stage-1 ceiling", color="#c53030", fontsize=7.5, ha="right")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "unet_comparison.png"), dpi=150)
print("saved plot")
