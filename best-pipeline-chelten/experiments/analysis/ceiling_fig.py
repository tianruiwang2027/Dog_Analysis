import pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/jitter_gold_pairs.pkl","rb") as f:
    J = pickle.load(f)
offsets_ms = J["offsets_ms"]

fig, axs = plt.subplots(1, 2, figsize=(13,5))

ax = axs[0]
labels = ["official\n(good data only)\nr=0.897", "window-count-matched\n(87.8% of pts, blunt\n±2.5s exclusion)\nr=0.943", "strict 1:1 beat-matched\n(99% of pts, surgical\nexclusion)\nr=0.903"]
vals = [0.897, 0.943, 0.903]
colors = ["0.5","tab:orange","tab:green"]
bars = ax.bar(labels, vals, color=colors)
ax.set_ylim(0.85, 0.97)
ax.set_ylabel("r vs hand ECG")
ax.set_title("How much of the gap is really\n\"fixable\" by excluding mismatched beats?")
ax.tick_params(axis='x', labelsize=8)
for b,v in zip(bars,vals): ax.annotate(f"{v:.3f}", (b.get_x()+b.get_width()/2, v), ha="center", va="bottom", fontsize=9)

ax = axs[1]
ax.hist(offsets_ms, bins=40, color="tab:blue", alpha=0.75)
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("SCG click time − matched ECG click time (ms)")
ax.set_ylabel("count")
ax.set_title(f"Per-beat timing jitter between the two\nhand-labeled channels (n={len(offsets_ms)})\nstd={offsets_ms.std():.0f}ms -- tiny vs a ~800-1000ms RR interval")

fig.suptitle("Chasing r from 0.90 toward 0.999: neither mismatch nor jitter has much room left", fontsize=13)
plt.tight_layout()
out = "/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_gold_ceiling.png"
plt.savefig(out, dpi=140)
print("->", out)
