import pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/cnn_confirm_final.pkl","rb") as f:
    C = pickle.load(f)
with open("/tmp/fully_auto_pipeline.pkl","rb") as f:
    F = pickle.load(f)
with open("/tmp/final_4way_compare.pkl","rb") as f:
    D = pickle.load(f)

s_old = F["s_cnn"]          # old: window-gated (keep all peaks, gate windows by gap/min_n)
s_new = C["balanced"]["s"]  # new: per-beat CNN confirm + RR-gap sanity check
s_new2 = C["aggressive"]["s"]
s_cor = D["s_cor"]

print(f"OLD  (window-gated):      r={s_old['r']:.4f}  MAE={s_old['mae']:.2f}  n={s_old['n']}")
print(f"NEW  (per-beat confirm):  r={s_new['r']:.4f}  MAE={s_new['mae']:.2f}  n={s_new['n']}")
print(f"NEW, stricter:            r={s_new2['r']:.4f}  MAE={s_new2['mae']:.2f}  n={s_new2['n']}")
print(f"CORAL (the filter):       r={s_cor['r']:.4f}  MAE={s_cor['mae']:.2f}  n={s_cor['n']}")

# ---------- 1) comparison bar chart ----------
fig, ax = plt.subplots(figsize=(8,5.5))
names = ["CORAL\n(the filter)", "Old approach\n(window-gated)", "New approach\n(per-beat CNN confirm)", "New, stricter\n(cutoff .70)"]
rs = [s_cor["r"], s_old["r"], s_new["r"], s_new2["r"]]
ns = [s_cor["n"], s_old["n"], s_new["n"], s_new2["n"]]
colors = ["tab:orange", "0.6", "tab:purple", "tab:green"]
bars = ax.bar(names, rs, color=colors, alpha=0.85, width=0.6)
for b, r, n in zip(bars, rs, ns):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f"r = {r:.3f}\nn = {n}", ha="center", fontsize=10)
ax.set_ylim(0, 1.05)
ax.set_ylabel("r (vs ECG hand-picked HR)")
ax.set_title("Confirming each detected peak with the CNN directly\nvs the old window-gating approach")
plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_confirm_vs_old.png", dpi=140)
print("saved bar chart")

# ---------- 2) HR over time ----------
ta_old = F["ta_cnn"]/1e6; xa_old = F["xa_cnn"]
ta_new = C["balanced"]["ta"]/1e6; xa_new = C["balanced"]["xa"]

ecg_tmid = D["ecg_tmid"]/1e6; ecg_hr_sm = D["ecg_hr_sm"]; ecg_v = D["ecg_v"]
t0 = min(ta_old.min(), ta_new.min())

fig2, ax2 = plt.subplots(figsize=(13,5))
ax2.plot(ecg_tmid[ecg_v]-t0, ecg_hr_sm[ecg_v], "-", lw=1.1, color="tab:blue", alpha=0.8, label="ECG hand-picked (reference)")
ax2.plot(ta_old - t0, xa_old, ".", ms=3, color="0.6", alpha=0.5, label=f"Old (window-gated), r={s_old['r']:.3f}")
ax2.plot(ta_new - t0, xa_new, ".", ms=3.5, color="tab:purple", alpha=0.65, label=f"New (per-beat confirm), r={s_new['r']:.3f}")
ax2.set_xlabel("time (s, session-relative)")
ax2.set_ylabel("HR (bpm)")
ax2.set_title("HR over time: per-beat CNN confirmation vs the old window-gated mask")
ax2.legend(fontsize=9, markerscale=3, loc="upper right")
plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_confirm_hr_over_time.png", dpi=140)
print("saved time series")
