import pickle, datetime
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/remove_mismatches_result.pkl","rb") as f:
    R = pickle.load(f)
s_h=R["s_h"]; s_c=R["s_c"]; s_c2=R["s_c2"]
xa_h=R["xa_h"]; yb_h=R["yb_h"]; ta_h=R["ta_h"]
xa_c=R["xa_c"]; yb_c=R["yb_c"]; ta_c=R["ta_c"]
mismatch_times=R["mismatch_times"]; RADIUS_US=R["RADIUS_US"]; RESTRICT=R["RESTRICT"]

fig = plt.figure(figsize=(14,9))
gs = fig.add_gridspec(2,2, height_ratios=[1,1.1])

axA = fig.add_subplot(gs[0,0])
labels=["BASELINE\n(hand labels)","cleaned\n(+-1.0s tight)","cleaned\n(+-3.0s, smoothing\nradius)"]
rvals=[s_h["r"], s_c2["r"], s_c["r"]]
ns=[s_h["n"], s_c2["n"], s_c["n"]]
colors=["black","tab:orange","tab:green"]
axA.bar(labels, rvals, color=colors)
for i,(v,n) in enumerate(zip(rvals,ns)):
    axA.text(i, v+0.012, f"r={v:.4f}\nn={n}", ha="center", fontsize=9)
axA.set_ylim(0,1.0)
axA.set_ylabel("r")
axA.set_title("A. Removing the 24 mismatch points + neighbors\n(3.7% of data dropped at +-3.0s radius)")

axB = fig.add_subplot(gs[0,1])
err_h = xa_h-yb_h
th_h = (ta_h-RESTRICT[0])/1e6/60
axB.scatter(th_h, err_h, s=6, color="lightcoral", alpha=0.5, label=f"removed (n={s_h['n']-s_c['n']})")
err_c = xa_c-yb_c
th_c = (ta_c-RESTRICT[0])/1e6/60
axB.scatter(th_c, err_c, s=6, color="tab:blue", alpha=0.5, label=f"kept (n={s_c['n']})")
axB.axhline(0, color="gray", lw=0.6)
axB.set_xlabel("minutes from window start")
axB.set_ylabel("SCG HR - ECG HR (bpm)")
axB.set_title("B. Error over time: removed vs kept points")
axB.legend(fontsize=8)

axC = fig.add_subplot(gs[1,:])
mt_min = (mismatch_times-RESTRICT[0])/1e6/60
th_all = (ta_h-RESTRICT[0])/1e6/60
axC.scatter(th_all, err_h, s=8, color="lightgray", zorder=1, label="all baseline points")
kept_set = set(ta_c.tolist())
removed_mask = np.array([t not in kept_set for t in ta_h])
axC.scatter(th_h[removed_mask], err_h[removed_mask], s=14, color="tab:red", zorder=3, label="removed (near a mismatch)")
for t in mt_min:
    axC.axvline(t, color="tab:purple", lw=0.6, alpha=0.4)
axC.axhline(0, color="gray", lw=0.6)
axC.set_xlabel("minutes from window start")
axC.set_ylabel("SCG HR - ECG HR (bpm)")
axC.set_title("C. All 24 mismatch locations (purple lines) and which baseline points got swept up (red) within +-3.0s\nremaining large errors elsewhere (e.g. ~17:48:39-53) are NOT explained by any of the 24 -- a separate issue")
axC.legend(fontsize=8, loc="upper right")

plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_mismatch_removal_result.png", dpi=130)
print("saved")
