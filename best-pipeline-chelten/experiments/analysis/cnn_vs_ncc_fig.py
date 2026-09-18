import pickle, sqlite3, datetime
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# ---------- load everything ----------
with open("/tmp/cnn_dataset.pkl","rb") as f:
    D = pickle.load(f)
with open("/tmp/cnn_train_result.pkl","rb") as f:
    TR = pickle.load(f)
n_train = TR["n_train"]; test_prob = TR["test_prob"]; y_test = TR["y_test"]
t_test = D["t"][n_train:]

with open("/tmp/env_template_wide.pkl","rb") as f:
    tpl = pickle.load(f)
template = tpl["template"]; HALF_N = tpl["HALF_N"]
template_energy = np.sqrt((template**2).sum())
with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
ts_sd = d["ts_sd"]; env_sd = d["env_sd"]; xf_s = d["xf_s"]; ts_s = d["ts_s"]

def ncc_score(t):
    i = np.searchsorted(ts_sd, t)
    if i-HALF_N < 0 or i+HALF_N+1 > len(env_sd): return np.nan
    s = env_sd[i-HALF_N:i+HALF_N+1]; s = s - s.mean()
    denom = np.sqrt((s**2).sum()) * template_energy
    if denom < 1e-12: return np.nan
    return float((s*template).sum() / denom)

ncc_test = np.array([ncc_score(t) for t in t_test])
valid = ~np.isnan(ncc_test)
auc_cnn = roc_auc_score(y_test[valid], test_prob[valid])
auc_ncc = roc_auc_score(y_test[valid], ncc_test[valid])
fpr_c, tpr_c, _ = roc_curve(y_test[valid], test_prob[valid])
fpr_n, tpr_n, _ = roc_curve(y_test[valid], ncc_test[valid])

with open("/tmp/cnn_mask_apply.pkl","rb") as f:
    CA = pickle.load(f)
with open("/tmp/mask_v9_apply.pkl","rb") as f:
    NA = pickle.load(f)

s_h = CA["s_h"]
cnn_best = dict(name="CNN mask\n(cutoff=0.55, gap=3.5s)", n=1549, r=0.8711, mae=2.60, agree=93.6)
ncc_bal = NA["results"]["cutoff=0.7 gap=2.0s min_n=2 (best J)"]["s_a"]
ncc_str = NA["results"]["cutoff=0.8 gap=2.0s min_n=2"]["s_a"]

# ---------- panel A+B ----------
fig = plt.figure(figsize=(13, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])

axA = fig.add_subplot(gs[0,0])
axA.plot(fpr_c, tpr_c, color="tab:purple", lw=2, label=f"CNN (AUC={auc_cnn:.3f})")
axA.plot(fpr_n, tpr_n, color="tab:orange", lw=2, label=f"NCC template (AUC={auc_ncc:.3f})")
axA.plot([0,1],[0,1], color="gray", lw=1, ls="--")
axA.set_xlabel("false positive rate"); axA.set_ylabel("true positive rate")
axA.set_title("A. Per-candidate identification: CNN vs NCC template\n(identical held-out, chronologically-later test set, n="+str(valid.sum())+")")
axA.legend(fontsize=9, loc="lower right")
axA.set_xlim(0,1); axA.set_ylim(0,1)

axB = fig.add_subplot(gs[0,1])
names = ["HAND\n(gold std)", "NCC template\n(balanced)", "NCC template\n(stricter)", "CNN mask\n(best)"]
rs = [s_h["r"], ncc_bal["r"], ncc_str["r"], cnn_best["r"]]
ns = [s_h["n"], ncc_bal["n"], ncc_str["n"], cnn_best["n"]]
colors = ["tab:gray","tab:orange","tab:orange","tab:purple"]
bars = axB.bar(names, rs, color=colors, alpha=0.85)
for b,n in zip(bars, ns):
    axB.text(b.get_x()+b.get_width()/2, b.get_height()+0.015, f"r={b.get_height():.3f}\nn={n}", ha="center", fontsize=8.5)
axB.set_ylim(0,1.05); axB.set_ylabel("r (vs ECG smoothed HR)")
axB.set_title("B. Downstream mask performance\n(same HR-agreement pipeline used throughout)")

# ---------- panel C: 3 known hard cases ----------
known = [
    ("2026-06-26 17:26:58", "hand: BAD", "NCC: correct (gap rule)", "CNN: correct (low score)"),
    ("2026-06-26 17:31:07", "hand: BAD", "NCC: WRONG (accepted)", "CNN: correct (low score)"),
    ("2026-06-26 17:47:11", "hand: GOOD", "NCC: WRONG (rejected)", "CNN: WRONG (rejected)"),
]

axC = fig.add_subplot(gs[1,:])
axC.axis("off")
axC.set_title("C. Three previously-identified hard cases, re-checked with the CNN", fontsize=11, loc="left")

# mini time-series panels embedded via inset axes
n_cases = len(known)
for i,(ks, hand_lab, ncc_v, cnn_v) in enumerate(known):
    tc = int(datetime.datetime.strptime(ks, "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc).timestamp()*1e6)
    inset = axC.inset_axes([i/n_cases+0.02, 0.05, 1/n_cases-0.06, 0.85])
    lo, hi = tc-2_500_000, tc+2_500_000
    sel = (ts_s>=lo)&(ts_s<hi)
    tms = (ts_s[sel]-tc)/1000
    inset.plot(tms, xf_s[sel], color="black", lw=0.6)
    # mark candidates in range with CNN/NCC scores
    cand_t = CA["cand_t"]; cand_cnn = CA["cand_cnn"]
    csel = (cand_t>=lo)&(cand_t<hi)
    for t in cand_t[csel]:
        cv = cand_cnn[cand_t==t][0]
        nv = ncc_score(t)
        tm = (t-tc)/1000
        inset.axvline(tm, color="tab:purple" if cv>=0.55 else "tab:purple", alpha=0.9 if cv>=0.55 else 0.25, lw=1.6, ls="-")
        inset.axvline(tm, color="tab:orange", alpha=0.9 if nv>=0.7 else 0.25, lw=1.0, ls=(0,(3,2)))
        inset.text(tm, inset.get_ylim()[1] if False else 0, "", fontsize=0)
    inset.set_title(f"{ks[11:]}\n{hand_lab}\n{ncc_v}\n{cnn_v}", fontsize=8)
    inset.set_xlabel("ms", fontsize=7)
    inset.tick_params(labelsize=7)
    inset.set_yticks([])

fig.text(0.5, 0.485, "solid purple = CNN-validated candidate (≥cutoff)   |   dashed orange = NCC-validated candidate (≥cutoff)   |   faint = below cutoff",
          ha="center", fontsize=8.5, style="italic")

plt.tight_layout(rect=[0,0,1,0.98])
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_cnn_vs_ncc.png", dpi=130)
print("saved")
print(f"AUC: CNN={auc_cnn:.4f}  NCC={auc_ncc:.4f}")
