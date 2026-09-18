import pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/mask_v9_apply.pkl","rb") as f:
    V9 = pickle.load(f)
with open("/tmp/mask_v5_apply.pkl","rb") as f:
    V5 = pickle.load(f)
with open("/tmp/mask_v7_window_sweep.pkl","rb") as f:
    V7 = pickle.load(f)

s_h = V9["s_h"]
rms_r = V5["s_a"]["r"]; rms_n = V5["s_a"]["n"]
trend_best = max(V7["rows"], key=lambda r: r["r"])
shape_best_name = "cutoff=0.7 gap=2.0s min_n=2 (best J)"
shape_best = V9["results"][shape_best_name]["s_a"]
shape_hi_name = "cutoff=0.8 gap=2.0s min_n=2"
shape_hi = V9["results"][shape_hi_name]["s_a"]

fig, axs = plt.subplots(1,2, figsize=(13,5.5))

labels = ["HAND\n(baseline)", "RMS-only\n(amplitude)", "TREND\n(rolling-avg,\nbest of sweep)", "SHAPE\n(template NCC,\nbalanced)", "SHAPE\n(template NCC,\nstricter)"]
rvals  = [s_h["r"], rms_r, trend_best["r"], shape_best["r"], shape_hi["r"]]
ns     = [s_h["n"], rms_n, trend_best["n"], shape_best["n"], shape_hi["n"]]
colors = ["black","tab:gray","tab:orange","tab:blue","tab:blue"]
bars = axs[0].bar(labels, rvals, color=colors)
for i,(v,n) in enumerate(zip(rvals,ns)):
    axs[0].text(i, v+0.012, f"r={v:.3f}\nn={n}", ha="center", fontsize=8.5)
axs[0].set_ylim(0,1.0)
axs[0].set_ylabel("r  (SCG smoothed-HR vs ECG smoothed-HR)")
axs[0].set_title("A. Shape (template-match) mask vs everything tried before\nsame data-volume ballpark as hand labels (n~1300-1600)")

names = ["cutoff=0.7 gap=2.0s min_n=2 (best J)","cutoff=0.8 gap=2.0s min_n=2","cutoff=0.8 gap=1.5s min_n=2",
         "cutoff=0.7 gap=2.5s min_n=3","cutoff=0.6 gap=1.5s min_n=3","cutoff=0.7 gap=3.0s min_n=1 (best acc)"]
xs = [V9["results"][n]["s_a"]["n"] for n in names]
ys = [V9["results"][n]["r"] if "r" in V9["results"][n] else V9["results"][n]["s_a"]["r"] for n in names]
ys = [V9["results"][n]["s_a"]["r"] for n in names]
ags = [V9["results"][n]["agree"]*100 for n in names]
sc = axs[1].scatter(xs, ys, c=ags, cmap="viridis", s=90, edgecolor="black", zorder=3)
for n,x,y in zip(names,xs,ys):
    short = n.split(" (")[0]
    axs[1].annotate(short, (x,y), fontsize=7, xytext=(5,5), textcoords="offset points")
axs[1].scatter([s_h["n"]],[s_h["r"]], marker="*", s=350, color="red", edgecolor="black", zorder=5, label="HAND label")
axs[1].axhline(0.73, color="gray", ls="--", lw=1, label="amplitude-only ceiling (~0.73)")
cb = plt.colorbar(sc, ax=axs[1]); cb.set_label("beat-level agreement w/ hand (%)")
axs[1].set_xlabel("n (points retained)")
axs[1].set_ylabel("r")
axs[1].set_title("B. Shape-mask configs tested\nall clear the old amplitude-only ceiling")
axs[1].legend(fontsize=8, loc="lower right")
axs[1].set_xlim(right=max(xs+[s_h["n"]])+220)

plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_shape_mask.png", dpi=130)
print("saved")
