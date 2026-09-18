import pickle, datetime
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/remove_badinterval_result.pkl","rb") as f:
    R = pickle.load(f)
s_h=R["s_h"]; s_1=R["s_1"]; s_2=R["s_2"]
xa_h=R["xa_h"]; yb_h=R["yb_h"]; ta_h=R["ta_h"]
xa_2=R["xa_2"]; yb_2=R["yb_2"]; ta_2=R["ta_2"]
RESTRICT=R["RESTRICT"]

fig = plt.figure(figsize=(14,8.5))
gs = fig.add_gridspec(2,2, height_ratios=[1,1.1])

axA = fig.add_subplot(gs[0,0])
labels=["BASELINE\n(hand labels)","+ 24 mismatch\npoints removed","+ bad-interval-edge\ncontamination removed"]
rvals=[s_h["r"], s_1["r"], s_2["r"]]
ns=[s_h["n"], s_1["n"], s_2["n"]]
colors=["black","tab:green","tab:blue"]
axA.bar(labels, rvals, color=colors)
for i,(v,n) in enumerate(zip(rvals,ns)):
    axA.text(i, v+0.012, f"r={v:.4f}\nn={n}", ha="center", fontsize=9)
axA.set_ylim(0,1.0)
axA.set_ylabel("r")
axA.set_title("A. Two-stage cleanup\n(12.1% of data dropped total)")

axB = fig.add_subplot(gs[0,1])
axB.scatter(xa_h, yb_h, s=6, color="lightcoral", alpha=0.4, label=f"baseline (r={s_h['r']:.3f})")
axB.scatter(xa_2, yb_2, s=6, color="tab:blue", alpha=0.5, label=f"cleaned (r={s_2['r']:.3f})")
lims = [30,140]
axB.plot(lims, lims, color="gray", ls="--", lw=1)
axB.set_xlim(lims); axB.set_ylim(lims)
axB.set_xlabel("SCG smoothed HR (bpm)")
axB.set_ylabel("ECG smoothed HR (bpm)")
axB.set_title("B. Scatter: baseline vs fully cleaned")
axB.legend(fontsize=8)

axC = fig.add_subplot(gs[1,:])
err_h = xa_h-yb_h; th_h = (ta_h-RESTRICT[0])/1e6/60
err_2 = xa_2-yb_2; th_2 = (ta_2-RESTRICT[0])/1e6/60
axC.scatter(th_h, err_h, s=8, color="lightgray", zorder=1, label="removed")
axC.scatter(th_2, err_2, s=10, color="tab:blue", zorder=2, label=f"kept (n={s_2['n']})")
axC.axhline(0, color="gray", lw=0.6)
axC.set_xlabel("minutes from window start")
axC.set_ylabel("SCG HR - ECG HR (bpm)")
axC.set_title("C. Error over time after both cleanup stages -- remaining errors are small (~7-9bpm) and scattered,\nno more concentrated clusters (diminishing returns from this 'remove bad regions' approach)")
axC.legend(fontsize=8, loc="upper right")

plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_badinterval_cleanup.png", dpi=130)
print("saved")
