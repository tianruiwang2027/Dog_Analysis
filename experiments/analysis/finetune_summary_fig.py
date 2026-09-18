import pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/cnn_train_result.pkl","rb") as f: TR = pickle.load(f)
with open("/tmp/cnn_finetune_v2b_result.pkl","rb") as f: V2 = pickle.load(f)   # HARD_MULT=2
with open("/tmp/cnn_finetune_v2_result.pkl","rb") as f: V6 = pickle.load(f)    # HARD_MULT=6

def eval_at(prob, y, thr=0.5):
    pred=(prob>=thr).astype(int); y=y.astype(int)
    tp=((pred==1)&(y==1)).sum(); fp=((pred==1)&(y==0)).sum(); fn=((pred==0)&(y==1)).sum()
    prec = tp/(tp+fp)*100 if (tp+fp)>0 else float("nan")
    rec = tp/(tp+fn)*100 if (tp+fn)>0 else float("nan")
    return prec, rec

hard_test = V2["hard_test"].astype(bool)   # the 60 test-set candidates that ALREADY fool the original model

configs = [
    ("original\n(no reweighting)", TR["test_prob"]),
    ("fine-tuned\n(hard-neg x2)",   V2["test_prob"]),
    ("fine-tuned\n(hard-neg x6)",   V6["test_prob"]),
]

fig, axs = plt.subplots(1, 2, figsize=(12,5.5))
names = [c[0] for c in configs]
precs = []; recs = []; hard_scores = []
for name, prob in configs:
    p, r = eval_at(prob, TR["y_test"])
    precs.append(p); recs.append(r)
    hard_scores.append(prob[hard_test].mean())

x = np.arange(len(names))
axA = axs[0]
axA.bar(x-0.18, precs, width=0.34, color="tab:blue", label="precision")
axA.bar(x+0.18, recs, width=0.34, color="tab:orange", label="recall")
for i,(p,r) in enumerate(zip(precs,recs)):
    axA.text(i-0.18, p+1.5, f"{p:.0f}%", ha="center", fontsize=9)
    axA.text(i+0.18, r+1.5, f"{r:.0f}%", ha="center", fontsize=9)
axA.set_xticks(x); axA.set_xticklabels(names, fontsize=9)
axA.set_ylim(0,105)
axA.set_ylabel("%")
axA.set_title("Overall precision / recall\n(held-out test set, cutoff=0.5)")
axA.legend(fontsize=9)

axB = axs[1]
bars = axB.bar(x, hard_scores, color="tab:red", alpha=0.8, width=0.5)
axB.axhline(0.40, color="0.4", ls="--", lw=1, label="confirm cutoff (0.40)")
for i,s in enumerate(hard_scores):
    axB.text(i, s+0.02, f"{s:.2f}", ha="center", fontsize=10)
axB.set_xticks(x); axB.set_xticklabels(names, fontsize=9)
axB.set_ylim(0,1)
axB.set_ylabel("mean CNN score")
axB.set_title("Mean score on the 60 held-out negatives\nthat ALREADY fool the original model\n(shape-coincidence false positives)")
axB.legend(fontsize=9)

fig.suptitle("Trying to fine-tune away the hardest false positives: precision barely moves,\nrecall collapses, and the specific hard cases stay above cutoff regardless", fontsize=11)
plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_finetune_summary.png", dpi=135)
print("saved")
for n,p,r,h in zip(names,precs,recs,hard_scores):
    print(f"{n.replace(chr(10),' '):30s} precision={p:.1f}%  recall={r:.1f}%  hard_neg_mean_score={h:.3f}")
