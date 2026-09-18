import pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/fully_auto_pipeline.pkl","rb") as f:
    F = pickle.load(f)
with open("/tmp/final_4way_compare.pkl","rb") as f:
    D = pickle.load(f)

s_cnn = F["s_cnn"]     # our best: detector + CNN mask, fully automatic
s_cor = D["s_cor"]     # CORAL, the filter

print(f"Our best algorithm (detector + CNN mask): r={s_cnn['r']:.4f}  MAE={s_cnn['mae']:.2f}  bias={s_cnn['bias']:+.2f}  n={s_cnn['n']}")
print(f"CORAL (the filter):                        r={s_cor['r']:.4f}  MAE={s_cor['mae']:.2f}  bias={s_cor['bias']:+.2f}  n={s_cor['n']}")

# ---------- 1) comparison bar chart ----------
fig, ax = plt.subplots(figsize=(6.5,5.5))
names = ["Our algorithm\n(detector + CNN mask)", "CORAL\n(the filter)"]
rs = [s_cnn["r"], s_cor["r"]]
colors = ["tab:purple", "tab:orange"]
bars = ax.bar(names, rs, color=colors, alpha=0.85, width=0.55)
for b, r, n in zip(bars, rs, [s_cnn["n"], s_cor["n"]]):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f"r = {r:.3f}\nn = {n}", ha="center", fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_ylabel("r (vs ECG hand-picked HR)")
ax.set_title("Fully-automatic pipelines vs ECG hand-picked\n(no SCG hand labels used by either)")
plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_r_comparison.png", dpi=140)
print("saved bar chart")

# ---------- 2) HR over time: all three overlaid ----------
ta_cnn = F["ta_cnn"]/1e6
xa_cnn = F["xa_cnn"]   # our algorithm's HR
yb_ecg_at_cnn = F["yb_cnn"]  # ECG matched to our algorithm's points

ta_cor = D["ta_cor"]/1e6
xa_cor = D["xa_cor"]   # CORAL's HR
yb_ecg_at_cor = D["yb_cor"]

t0 = min(ta_cnn.min(), ta_cor.min())

fig2, ax2 = plt.subplots(figsize=(13,5))
ax2.plot(ta_cnn - t0, xa_cnn, ".", ms=3.5, color="tab:purple", alpha=0.65, label="Our algorithm (detector + CNN mask)")
ax2.plot(ta_cor - t0, xa_cor, ".", ms=3.5, color="tab:orange", alpha=0.45, label="CORAL (the filter)")
# ECG hand-picked reference -- use the full continuous hand series, not just matched subsets
ecg_tmid = D["ecg_tmid"]/1e6; ecg_hr_sm = D["ecg_hr_sm"]; ecg_v = D["ecg_v"]
ax2.plot(ecg_tmid[ecg_v]-t0, ecg_hr_sm[ecg_v], "-", lw=1.1, color="tab:blue", alpha=0.85, label="ECG hand-picked (reference)")

ax2.set_xlabel("time (s, session-relative)")
ax2.set_ylabel("HR (bpm)")
ax2.set_title(f"HR over time: our algorithm (r={s_cnn['r']:.3f}) vs CORAL (r={s_cor['r']:.3f}) vs ECG hand-picked")
ax2.legend(fontsize=9, markerscale=3, loc="upper right")
plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_hr_over_time.png", dpi=140)
print("saved time series")
