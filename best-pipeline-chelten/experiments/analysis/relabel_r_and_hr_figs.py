import pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/relabel_failure_points.pkl","rb") as f:
    F = pickle.load(f)
xa, yb, ta, r = F["xa"], F["yb"], F["ta"], F["r"]
n = len(xa)
mae = np.abs(xa-yb).mean()

# ---------- r coefficient scatter ----------
fig, ax = plt.subplots(figsize=(6.5,6.5))
ax.scatter(yb, xa, s=8, alpha=0.35, color="tab:purple", edgecolors="none")
lo = min(yb.min(), xa.min()) - 3
hi = max(yb.max(), xa.max()) + 3
ax.plot([lo,hi],[lo,hi], "--", color="0.4", lw=1.2, label="perfect agreement")
ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
ax.set_xlabel("ECG HR (bpm, gold standard)")
ax.set_ylabel("SCG HR (bpm, CNN-SCG+ECG relabeled)")
ax.set_title(f"SCG vs ECG heart rate -- CNN (envelope, SCG+ECG relabeled)\nr = {r:.3f}   n = {n}   MAE = {mae:.2f} bpm   (7.0s lag applied)")
ax.set_aspect("equal")
ax.legend(fontsize=9, loc="upper left")
plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_relabel_r_scatter.png", dpi=140)
print("saved scatter")

# ---------- HR over time ----------
with open("/tmp/final_4way_compare.pkl","rb") as f:
    D = pickle.load(f)
ecg_tmid = D["ecg_tmid"]; ecg_hr_sm = D["ecg_hr_sm"]; ecg_v = D["ecg_v"]

fig2, ax2 = plt.subplots(figsize=(15,5.5))
esel = ecg_v
t0ref = ecg_tmid[esel][0]
ax2.plot((ecg_tmid[esel]-t0ref)/1e6/60, ecg_hr_sm[esel], "-", lw=1.0, color="tab:blue", alpha=0.8, label="ECG hand-picked (gold standard)")
ax2.plot((ta-t0ref)/1e6/60, xa, ".", ms=3.5, color="tab:purple", alpha=0.55, label="SCG (CNN-SCG+ECG relabeled)")
ax2.set_xlabel("time (minutes)")
ax2.set_ylabel("HR (bpm)")
ax2.set_title(f"Heart rate over time: CNN (envelope, SCG+ECG relabeled) vs ECG (r={r:.3f}, n={n}, 7.0s lag applied)")
ax2.legend(fontsize=10, loc="upper right")
plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_relabel_hr_over_time.png", dpi=130)
print("saved hr-over-time")
