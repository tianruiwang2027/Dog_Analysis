import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
ts_sd = d["ts_sd"]; env_sd = d["env_sd"]

with open("/tmp/three_model_r_compare.pkl","rb") as f:
    prev = pickle.load(f)
with open("/tmp/snap_r_compare.pkl","rb") as f:
    SNAP = pickle.load(f)
with open("/tmp/snapped_candidates.pkl","rb") as f:
    SN = pickle.load(f)

names = ["CNN-SCG\n(original)", "CNN-SCG+ECG\n(relabeled)", "CNN-raw\n(relabeled)", "CNN-SCG+ECG\n(snapped+relabeled)"]
rs = [prev["CNN-SCG (original)"]["s"]["r"],
      prev["CNN-SCG+ECG (envelope, relabeled)"]["s"]["r"],
      prev["CNN-raw (raw signal, relabeled)"]["s"]["r"],
      SNAP["best"]["s"]["r"]]
ns = [prev["CNN-SCG (original)"]["s"]["n"],
      prev["CNN-SCG+ECG (envelope, relabeled)"]["s"]["n"],
      prev["CNN-raw (raw signal, relabeled)"]["s"]["n"],
      SNAP["best"]["s"]["n"]]

fig = plt.figure(figsize=(13, 9))
gs = fig.add_gridspec(3, 3, height_ratios=[1.1, 1, 1])

# --- panel 1: bar chart of best r, all 4 models ---
ax0 = fig.add_subplot(gs[0, :])
colors = ["tab:blue","tab:green","tab:orange","tab:red"]
bars = ax0.bar(names, rs, color=colors)
for b, r, n in zip(bars, rs, ns):
    ax0.text(b.get_x()+b.get_width()/2, r+0.01, f"r={r:.3f}\nn={n}", ha="center", va="bottom", fontsize=9)
ax0.set_ylabel("best downstream r (SCG HR vs ECG HR)")
ax0.set_ylim(0, 1.0)
ax0.set_title("Snap strategy result: does re-anchoring S2-locked candidates onto their true S1\n"
              "improve downstream HR correlation? (best cutoff per model, whole session)")
ax0.axhline(prev["CNN-SCG+ECG (envelope, relabeled)"]["s"]["r"], color="green", ls="--", lw=1, alpha=0.5)

# --- panels 2-7: 6 examples of actual snaps, showing envelope before/after ---
final_t = SN["final_t"]; was_snapped = SN["final_was_snapped"]
snapped_t = final_t[was_snapped]
cand_t_orig = np.sort(SN["cand_t_orig"])
# for display, recover each snapped candidate's PRE-snap (original) position by nearest-orig-candidate search
example_idx = np.linspace(0, len(snapped_t)-1, 6).astype(int)

for k, ei in enumerate(example_idx):
    ax = fig.add_subplot(gs[1 + k//3, k%3])
    t_new = snapped_t[ei]
    # find the original (pre-snap) candidate whose snap lands at t_new: search cand_t_orig for one
    # within [t_new+100ms, t_new+300ms] (since snap moves candidate EARLIER by 100-300ms)
    cand_mask = (cand_t_orig >= t_new + 90_000) & (cand_t_orig <= t_new + 310_000)
    cand_near = cand_t_orig[cand_mask]
    t_old = cand_near[0] if len(cand_near) else t_new + 200_000

    win = 700_000
    lo, hi = t_new - win, t_new + win
    i_lo, i_hi = np.searchsorted(ts_sd, lo), np.searchsorted(ts_sd, hi)
    tt = (ts_sd[i_lo:i_hi] - t_new) / 1000.0
    ax.plot(tt, env_sd[i_lo:i_hi], color="black", lw=1)
    ax.axvline(0, color="tab:red", lw=2, label="snapped position\n(recovered true S1)")
    ax.axvline((t_old - t_new)/1000.0, color="tab:gray", ls="--", lw=2, label="original candidate\n(was anchored on S2)")
    ax.set_title(f"example {k+1}", fontsize=9)
    ax.set_xlabel("ms")
    if k == 0:
        ax.legend(fontsize=7, loc="upper right")

fig.suptitle("", fontsize=1)
plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_snap_compare.png", dpi=130)
print("saved chelten_snap_compare.png")
print("bar values:", list(zip(names, rs, ns)))
