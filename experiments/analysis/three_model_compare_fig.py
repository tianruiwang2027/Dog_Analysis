import pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/full_session_scores_all_models.pkl","rb") as f:
    S = pickle.load(f)
with open("/tmp/three_model_r_compare.pkl","rb") as f:
    RES = pickle.load(f)

exec(open("/tmp/dog-test-ecg/dog-test-ecg-code/analysis/three_model_r_compare.py").read().split('models = {')[0])

models = {
    "CNN-SCG (original)": (S["scores_scg"], "0.55"),
    "CNN-SCG+ECG (relabeled)": (S["scores_relabel"], "tab:purple"),
    "CNN-raw signal (relabeled)": (S["scores_raw"], "tab:orange"),
}

fig, axs = plt.subplots(1,2, figsize=(14,5.5))
axA = axs[0]
for name,(scores,color) in models.items():
    cutoffs = np.arange(0.05,0.96,0.05)
    rs = []
    for c in cutoffs:
        r = confirm_and_run(scores, c)
        rs.append(r["s"]["r"] if r else np.nan)
    axA.plot(cutoffs, rs, "o-", color=color, label=name, ms=4)
axA.set_xlabel("confirm cutoff")
axA.set_ylabel("downstream r (vs ECG hand-picked HR)")
axA.set_title("Downstream r vs cutoff")
axA.legend(fontsize=8.5)
axA.grid(alpha=0.25)

axB = axs[1]
names = list(RES.keys())
rs_best = [RES[n]["s"]["r"] for n in names]
ns_best = [RES[n]["s"]["n"] for n in names]
colors = ["0.55","tab:purple","tab:orange"]
bars = axB.bar(range(3), rs_best, color=colors, alpha=0.85, width=0.55)
for i,(r,n) in enumerate(zip(rs_best,ns_best)):
    axB.text(i, r+0.015, f"r={r:.3f}\nn={n}", ha="center", fontsize=9)
axB.set_xticks(range(3)); axB.set_xticklabels([n.replace(" (","\n(") for n in names], fontsize=8)
axB.set_ylim(0,1.0)
axB.set_ylabel("best downstream r")
axB.set_title("Best operating point per model")

fig.suptitle("Three CNN variants: original SCG-only labels vs ECG-corroborated relabeling vs raw-signal input", fontsize=11)
plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_three_model_compare.png", dpi=135)
print("saved")
