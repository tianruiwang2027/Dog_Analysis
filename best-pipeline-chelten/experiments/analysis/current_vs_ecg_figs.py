import pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/cnn_recover_eval.pkl","rb") as f:
    R = pickle.load(f)
res_new = R["res_new"]
s = res_new["s"]

with open("/tmp/final_4way_compare.pkl","rb") as f:
    D = pickle.load(f)
ecg_tmid = D["ecg_tmid"]; ecg_hr_sm = D["ecg_hr_sm"]; ecg_v = D["ecg_v"]

print(f"current algo (per-beat confirm + S2 recovery) vs ECG hand-picked: "
      f"r={s['r']:.4f}  n={s['n']}  MAE={s['mae']:.2f} bpm  bias={s['bias']:+.2f} bpm  "
      f"(7.0s SCG->ECG lag already applied)")

# ---------- 1) r coefficient bar ----------
fig, ax = plt.subplots(figsize=(5,5.5))
ax.bar(["SCG (current algorithm)\nvs ECG hand-picked"], [s["r"]], color="tab:purple", alpha=0.85, width=0.5)
ax.text(0, s["r"]+0.02, f"r = {s['r']:.3f}\nn = {s['n']}", ha="center", fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_ylabel("r (Pearson correlation vs ECG HR)")
ax.set_title("SCG algorithm vs gold-standard ECG\n(fully automatic, 7.0s lag applied)")
plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_current_r.png", dpi=140)
print("saved r bar chart")

# ---------- 2) HR over time, full session ----------
fig2, ax2 = plt.subplots(figsize=(15,5.5))
esel = ecg_v
ax2.plot((ecg_tmid[esel]-ecg_tmid[esel][0])/1e6/60, ecg_hr_sm[esel], "-", lw=1.0,
         color="tab:blue", alpha=0.8, label="ECG hand-picked (gold standard)")
t0ref = ecg_tmid[esel][0]
ax2.plot((res_new["ta"]-t0ref)/1e6/60, res_new["xa"], ".", ms=3.5,
         color="tab:purple", alpha=0.55, label="SCG (current algorithm)")
ax2.set_xlabel("time (minutes)")
ax2.set_ylabel("HR (bpm)")
ax2.set_title(f"Heart rate over time: current SCG algorithm vs ECG (r={s['r']:.3f}, n={s['n']}, 7.0s lag applied)")
ax2.legend(fontsize=10, loc="upper right")
plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_current_hr_over_time.png", dpi=130)
print("saved hr-over-time")
