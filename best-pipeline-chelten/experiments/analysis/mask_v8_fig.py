import pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/mask_v8_grid_search.pkl","rb") as f:
    D = pickle.load(f)
grid = D["grid"]; s_h = D["s_h"]; WIN_LIST_S = D["WIN_LIST_S"]; PCTS = D["PCTS"]

R = np.full((len(WIN_LIST_S), len(PCTS)), np.nan)
N = np.full((len(WIN_LIST_S), len(PCTS)), np.nan)
for g in grid:
    i = WIN_LIST_S.index(g["WIN_S"]); j = list(PCTS).index(g["pct"])
    R[i,j] = g["r"]; N[i,j] = g["n"]

fig, axs = plt.subplots(1,2, figsize=(15,6))

im = axs[0].imshow(R, aspect="auto", cmap="viridis", vmin=0.4, vmax=0.9)
axs[0].set_xticks(range(len(PCTS))); axs[0].set_xticklabels(PCTS.astype(int), fontsize=7)
axs[0].set_yticks(range(len(WIN_LIST_S))); axs[0].set_yticklabels(WIN_LIST_S)
axs[0].set_xlabel("threshold percentile (of that window length's RMS distribution)")
axs[0].set_ylabel("window length (s)")
axs[0].set_title(f"A. r across the full (window length x threshold) grid\nHAND baseline r={s_h['r']:.3f} (n=1645) -- nothing here reaches it")
cb = plt.colorbar(im, ax=axs[0]); cb.set_label("r")
# annotate best cell
best = max(grid, key=lambda g: g["r"])
bi = WIN_LIST_S.index(best["WIN_S"]); bj = list(PCTS).index(best["pct"])
axs[0].scatter([bj],[bi], marker="*", s=250, color="red", edgecolor="white", linewidth=1,
               label=f"best r={best['r']:.3f}\n(WIN={best['WIN_S']}s, n={best['n']})")
axs[0].legend(fontsize=8, loc="upper right")

colors = plt.cm.plasma(np.linspace(0,1,len(WIN_LIST_S)))
for i,WIN_S in enumerate(WIN_LIST_S):
    pts = [g for g in grid if g["WIN_S"]==WIN_S]
    pts = sorted(pts, key=lambda g:g["n"])
    ns = [g["n"] for g in pts]; rs = [g["r"] for g in pts]
    axs[1].plot(ns, rs, "o-", color=colors[i], ms=4, lw=1, label=f"{WIN_S}s")
axs[1].scatter([s_h["n"]],[s_h["r"]], marker="*", s=350, color="red", edgecolor="black", zorder=5, label="HAND label")
axs[1].axvline(s_h["n"], color="gray", ls=":", lw=1)
axs[1].set_xlabel("n (points retained in the gold-standard comparison)")
axs[1].set_ylabel("r")
axs[1].set_title("B. r vs. how much data is kept, by window length\n(same tradeoff at every window length: more data kept -> lower r)")
axs[1].legend(fontsize=7, ncol=2, title="window", loc="lower left")

plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_grid_search.png", dpi=130)
print("saved")
