import pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/illustrate_15missed.pkl","rb") as f:
    I = pickle.load(f)

xa, yb, bad_idx, keep = I["xa"], I["yb"], I["bad_idx"], I["keep"]
s_all, s_removed = I["s_all"], I["s_removed"]

fig, axes = plt.subplots(1, 2, figsize=(12,5.8))
lim = [30,150]

ax = axes[0]
ax.scatter(yb[keep], xa[keep], s=8, alpha=0.35, color="tab:green", label="kept points")
ax.scatter(yb[bad_idx], xa[bad_idx], s=45, facecolors="none", edgecolors="red", linewidths=1.6, label="the 15 missed-beat points")
ax.plot(lim, lim, "k-", lw=1, alpha=0.6)
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
ax.set_xlabel("Hand ECG HR (bpm)"); ax.set_ylabel("Best-pipeline HR (bpm)")
ax.set_title(f"WITH the 15 points\nr={s_all['r']:.3f}  MAE={s_all['mae']:.2f}  n={s_all['n']}")
ax.legend(loc="lower right", fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.scatter(yb[keep], xa[keep], s=8, alpha=0.35, color="tab:green")
ax.plot(lim, lim, "k-", lw=1, alpha=0.6)
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
ax.set_xlabel("Hand ECG HR (bpm)"); ax.set_ylabel("Best-pipeline HR (bpm)")
ax.set_title(f"WITHOUT the 15 points (just 15 of {s_all['n']} removed)\nr={s_removed['r']:.3f}  MAE={s_removed['mae']:.2f}  n={s_removed['n']}")
ax.grid(True, alpha=0.3)

fig.suptitle("Effect on r of removing only the 15 missed-beat artifact points", fontsize=13)
plt.tight_layout()
out = "/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_r_with_without_15.png"
plt.savefig(out, dpi=140)
print("->", out)
print(s_all, s_removed)
