import pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/fully_auto_pipeline.pkl","rb") as f:
    F = pickle.load(f)

# full spectrum: from "no quality control at all" up to "fully hand-labeled gold standard"
names =  ["Raw detector\n(no mask at all)",
          "Detector + NCC mask\n(fully automatic)",
          "Detector + CNN mask\n(fully automatic)",
          "Detector\n(hand-label-gated)*",
          "Hand-clicked SCG\n(gold standard)*",
          "Hand-clicked SCG\n(valid regions)*"]
rs =     [F["s_none"]["r"], F["s_ncc"]["r"], F["s_cnn"]["r"], 0.8217, 0.8959, 0.9465]
ns =     [F["s_none"]["n"], F["s_ncc"]["n"], F["s_cnn"]["n"], 1563, 1645, 1446]
colors = ["tab:red", "tab:orange", "tab:purple", "0.6", "0.6", "0.6"]
hatches= ["", "", "", "//", "//", "//"]

fig, ax = plt.subplots(figsize=(11.5, 6))
x = np.arange(len(names))
bars = ax.bar(x, rs, color=colors, alpha=0.85, hatch=hatches, edgecolor="black", linewidth=0.6)
for i,b in enumerate(bars):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.015, f"r={rs[i]:.3f}\nn={ns[i]}", ha="center", fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
ax.set_ylim(0,1.05)
ax.set_ylabel("r (vs ECG hand-picked HR)")
ax.set_title("Fully-automatic SCG pipeline (no hand SCG quality labels at all) vs ECG hand-picked\n"
              "solid bars = zero SCG hand-labels used  |  hatched bars = uses hand SCG quality labels (shown for context)")
plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_fully_automatic.png", dpi=130)
print("saved")

# second panel: coverage / time-series view of what the CNN mask actually kept
fig2, ax2 = plt.subplots(figsize=(12,4))
ta = F["ta_cnn"]/1e6
xa = F["xa_cnn"]; yb = F["yb_cnn"]
t0 = min(ta)
ax2.plot((ta-t0), xa, ".", ms=3, color="tab:purple", alpha=0.6, label="fully-automatic SCG (detector+CNN mask)")
ax2.plot((ta-t0), yb, ".", ms=3, color="tab:blue", alpha=0.4, label="ECG hand-picked")
ax2.set_xlabel("time (s, session-relative)")
ax2.set_ylabel("HR (bpm)")
ax2.set_title(f"Fully-automatic SCG HR vs ECG hand-picked over the session (r={F['s_cnn']['r']:.3f}, n={F['s_cnn']['n']})")
ax2.legend(fontsize=9, markerscale=3)
plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_fully_automatic_timeseries.png", dpi=130)
print("saved 2")
