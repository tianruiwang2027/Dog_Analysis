import pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/cnn_train_result.pkl","rb") as f:
    TR = pickle.load(f)
prob = np.asarray(TR["test_prob"]).ravel()
y = np.asarray(TR["y_test"]).ravel()

n_pos = int((y==1).sum())
n_neg = int((y==0).sum())
print(f"held-out test set: n={len(y)}  positives(true beats)={n_pos}  negatives(not-beats)={n_neg}")

thresholds = np.linspace(0.0, 1.0, 201)
tpr = np.empty_like(thresholds)
fpr = np.empty_like(thresholds)
for i, thr in enumerate(thresholds):
    pred_pos = prob >= thr
    tp = np.sum(pred_pos & (y==1))
    fp = np.sum(pred_pos & (y==0))
    tpr[i] = tp / n_pos * 100
    fpr[i] = fp / n_neg * 100

# mark a few reference cutoffs used elsewhere in this analysis
refs = [0.40, 0.55, 0.70]

fig, ax = plt.subplots(figsize=(9,6))
ax.plot(thresholds, tpr, color="tab:green", lw=2, label="true positive rate (% of real beats confirmed)")
ax.plot(thresholds, fpr, color="tab:red", lw=2, label="false positive rate (% of non-beats confirmed)")
for r in refs:
    j = np.argmin(np.abs(thresholds-r))
    ax.axvline(r, color="0.75", lw=0.8, ls=":")
    ax.annotate(f"cutoff={r:.2f}\nTPR={tpr[j]:.1f}%\nFPR={fpr[j]:.1f}%", xy=(r, max(tpr[j],fpr[j])),
                xytext=(r+0.02, 60), fontsize=8.5, color="0.25",
                arrowprops=dict(arrowstyle="-", color="0.6", lw=0.6))
ax.set_xlabel("CNN confirm cutoff")
ax.set_ylabel("% ")
ax.set_xlim(0,1)
ax.set_ylim(0,105)
ax.set_title(f"True positive % and false positive % vs CNN cutoff\n(held-out chronological test set, n={len(y)})")
ax.legend(fontsize=10, loc="center left")
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_tpr_fpr_vs_cutoff.png", dpi=140)
print("saved")

for r in refs:
    j = np.argmin(np.abs(thresholds-r))
    print(f"cutoff={r:.2f}  TPR={tpr[j]:.1f}%  FPR={fpr[j]:.1f}%")
