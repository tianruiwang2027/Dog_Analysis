import pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

with open("/tmp/cnn_train_result.pkl","rb") as f: TR = pickle.load(f)
with open("/tmp/cnn_context_train_result.pkl","rb") as f: CTX = pickle.load(f)
with open("/tmp/cnn_context_dataset.pkl","rb") as f: CTXD = pickle.load(f)
with open("/tmp/cnn_fooling_negs.pkl","rb") as f: FN = pickle.load(f)

label = CTXD["label"]; bg = CTXD["bg"]; fool = FN["fool"]

fpr_o, tpr_o, _ = roc_curve(TR["y_test"], TR["test_prob"])
fpr_c, tpr_c, _ = roc_curve(CTX["y_test"], CTX["test_prob"])
auc_o = roc_auc_score(TR["y_test"], TR["test_prob"])
auc_c = roc_auc_score(CTX["y_test"], CTX["test_prob"])

fig, axs = plt.subplots(1, 2, figsize=(13,5.5))

axA = axs[0]
axA.plot(fpr_o, tpr_o, color="tab:gray", lw=2, label=f"shape only  (AUC={auc_o:.3f})")
axA.plot(fpr_c, tpr_c, color="tab:purple", lw=2, label=f"shape + background context  (AUC={auc_c:.3f})")
axA.plot([0,1],[0,1],"--",color="0.85",lw=1)
axA.set_xlabel("false positive rate")
axA.set_ylabel("true positive rate")
axA.set_title("ROC: adding background-noise context\nimproves overall discrimination")
axA.legend(fontsize=9, loc="lower right")

axB = axs[1]
groups = [
    ("real beats\n(positives)", bg[label==1]),
    ("easy negatives\n(shape alone already rejects)", bg[(label==0)&~fool]),
    ("hard negatives\n(fool the shape-only model)", bg[(label==0)&fool]),
]
data = [np.log10(g[1]+1e-9) for g in groups]
bp = axB.boxplot(data, labels=[g[0] for g in groups], showfliers=False, patch_artist=True)
colors = ["tab:green","tab:gray","tab:red"]
for patch,c in zip(bp["boxes"], colors):
    patch.set_facecolor(c); patch.set_alpha(0.5)
axB.set_ylabel("log10(background RMS)")
axB.set_title("Why it only half-works:\nhard negatives already look as 'calm' as real beats")
plt.setp(axB.get_xticklabels(), fontsize=8.5)

plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_context_model_summary.png", dpi=135)
print("saved")
print(f"AUC: shape-only={auc_o:.4f}  shape+context={auc_c:.4f}")
