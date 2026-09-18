import pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/cnn_recover_eval.pkl","rb") as f:
    R = pickle.load(f)
res_new = R["res_new"]
s = res_new["s"]
xa, yb = res_new["xa"], res_new["yb"]   # xa = SCG HR, yb = ECG HR (matched pairs)

fig, ax = plt.subplots(figsize=(6.5,6.5))
ax.scatter(yb, xa, s=8, alpha=0.35, color="tab:purple", edgecolors="none")
lo = min(yb.min(), xa.min()) - 3
hi = max(yb.max(), xa.max()) + 3
ax.plot([lo,hi],[lo,hi], "--", color="0.4", lw=1.2, label="perfect agreement")
ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
ax.set_xlabel("ECG HR (bpm, gold standard)")
ax.set_ylabel("SCG HR (bpm, current algorithm)")
ax.set_title(f"SCG vs ECG heart rate\nr = {s['r']:.3f}   n = {s['n']}   MAE = {s['mae']:.2f} bpm   (7.0s lag applied)")
ax.set_aspect("equal")
ax.legend(fontsize=9, loc="upper left")
plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_current_r_scatter.png", dpi=140)
print("saved scatter")
