import pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/mask_v6_trend_ratio.pkl","rb") as f:
    feats = pickle.load(f)
with open("/tmp/mask_v6_apply.pkl","rb") as f:
    V = pickle.load(f)
with open("/tmp/mask_v5_apply.pkl","rb") as f:
    V5 = pickle.load(f)  # previous RMS-only mask, for reference bar

good = [f for f in feats if f["label"]=="good"]
bad  = [f for f in feats if f["label"]=="bad"]
g_tr = np.array([f["trend_range"] for f in good]); b_tr = np.array([f["trend_range"] for f in bad])
g_cv = np.array([f["ratio_cv"] for f in good]); b_cv = np.array([f["ratio_cv"] for f in bad])

fig, axs = plt.subplots(1,3, figsize=(16,4.5))

bins = np.logspace(np.log10(0.3), np.log10(300), 55)
axs[0].hist(g_tr, bins=bins, color="tab:green", alpha=0.5, label=f"hand GOOD (n={len(g_tr)})")
axs[0].hist(b_tr, bins=bins, color="tab:red", alpha=0.5, label=f"hand BAD (n={len(b_tr)})")
axs[0].axvline(5, color="black", ls="--", lw=1.3, label="thr=5")
axs[0].axvline(10, color="black", ls=":", lw=1.3, label="thr=10")
axs[0].set_xscale("log")
axs[0].set_xlabel("trend_range: peak-to-peak of a 200ms rolling average\nof the respiration-removed signal, within each 3s window")
axs[0].set_title("A. Feature 1: rolling-average trend flatness\n(discriminates about as well as plain RMS did)")
axs[0].legend(fontsize=8)

bins2 = np.linspace(0,1.5,40)
axs[1].hist(g_cv[~np.isnan(g_cv)], bins=bins2, color="tab:green", alpha=0.5, density=True, label="hand GOOD")
axs[1].hist(b_cv[~np.isnan(b_cv)], bins=bins2, color="tab:red", alpha=0.5, density=True, label="hand BAD")
axs[1].axvline(np.nanmedian(g_cv), color="tab:green", ls="--", lw=1.3)
axs[1].axvline(np.nanmedian(b_cv), color="tab:red", ls="--", lw=1.3)
axs[1].set_xlabel("ratio_cv: coefficient of variation of (cardiac amplitude / noise floor)\nacross six 0.5s sub-segments of each 3s window")
axs[1].set_title(f"B. Feature 2: cardiac/noise proportion constancy\nGOOD median={np.nanmedian(g_cv):.2f}  BAD median={np.nanmedian(b_cv):.2f}  -- barely separates")
axs[1].legend(fontsize=8)

labels = ["HAND\n(baseline)", "RMS-only\n(last mask,\nthr=20)", "TREND\nthr=5", "TREND\nthr=10"]
rvals = [V["s_h"]["r"], V5["s_a"]["r"], V["results"][5.0]["s_a"]["r"], V["results"][10.0]["s_a"]["r"]]
colors = ["black","tab:gray","tab:blue","tab:blue"]
axs[2].bar(labels, rvals, color=colors)
for i,v in enumerate(rvals):
    axs[2].text(i, v+0.01, f"{v:.3f}", ha="center", fontsize=9)
axs[2].set_ylim(0,1)
axs[2].set_ylabel("r  (SCG smoothed-HR vs ECG smoothed-HR)")
axs[2].set_title("C. Effect on the gold-standard comparison\nsame failure mode as before: r still craters")

plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_trend_ratio_mask.png", dpi=130)
print("saved")
