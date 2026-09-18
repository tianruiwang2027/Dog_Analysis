import pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/ncc_cutoff_sweep.pkl","rb") as f:
    R = pickle.load(f)

cutoffs = sorted(R.keys())
r_vals = [R[c]["s"]["r"] for c in cutoffs]
mae_vals = [R[c]["s"]["mae"] for c in cutoffs]
n_vals = [R[c]["s"]["n"] for c in cutoffs]
tp_pct = [R[c]["tp_pct"] for c in cutoffs]
fp_pct = [R[c]["fp_pct"] for c in cutoffs]

fig, axs = plt.subplots(1, 2, figsize=(13,5))

ax = axs[0]
ax.plot(cutoffs, r_vals, "o-", color="tab:green", label="r (vs hand ECG)")
ax2 = ax.twinx()
ax2.plot(cutoffs, mae_vals, "s--", color="tab:orange", label="MAE (bpm)")
ax.set_xlabel("NCC cutoff (lower = more tolerant, keeps more candidates)")
ax.set_ylabel("r", color="tab:green")
ax2.set_ylabel("MAE (bpm)", color="tab:orange")
ax.set_title("r and MAE vs. NCC cutoff")
ax.grid(True, alpha=0.3)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labels1+labels2, loc="lower left", fontsize=8)
for c,r in zip(cutoffs, r_vals):
    ax.annotate(f"{r:.3f}", (c,r), textcoords="offset points", xytext=(0,8), fontsize=8, ha="center", color="tab:green")

ax = axs[1]
ax.plot(cutoffs, tp_pct, "o-", color="tab:green", label="% of true beats kept")
ax.plot(cutoffs, fp_pct, "o-", color="tab:red", label="% of false positives kept")
ax.set_xlabel("NCC cutoff")
ax.set_ylabel("% kept")
ax.set_title("Composition tradeoff as cutoff loosens")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig.suptitle("Effect of a more tolerant NCC cutoff on the smoothed-HR comparison", fontsize=13)
plt.tight_layout()
out = "/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_ncc_cutoff_sweep.png"
plt.savefig(out, dpi=140)
print("->", out)
