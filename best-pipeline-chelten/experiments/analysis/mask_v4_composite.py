import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

with open("/tmp/mask_v3_all_features.pkl","rb") as f:
    feats = pickle.load(f)

tc = np.array([f["tc"] for f in feats], dtype="int64")
y = np.array([1 if f["label"]=="bad" else 0 for f in feats])  # 1 = bad

drift = np.array([f["drift_amp"] for f in feats], dtype=float)
cbp   = np.array([f["cardiac_bp_amp"] for f in feats], dtype=float)
camp  = np.array([f["cardiac_amp"] for f in feats], dtype=float)
s1s2  = np.array([f["s1s2_std_ms"] for f in feats], dtype=float)
has_s1s2 = (~np.isnan(s1s2)).astype(float)
s1s2_filled = np.where(np.isnan(s1s2), 100.0, s1s2)  # missing S1S2 structure -> treat as "very irregular"

log_drift = np.log10(drift + 1e-6)
log_cbp   = np.log10(cbp + 1e-6)
log_camp  = np.log10(camp + 1e-6)

X = np.column_stack([log_drift, log_cbp, log_camp, s1s2_filled, has_s1s2])
feat_names = ["log10(drift_amp)", "log10(cardiac_bp_amp)", "log10(cardiac_amp)", "s1s2_std_ms(filled)", "has_s1s2_structure"]

# standardize for interpretable-ish coefficients
mu = X.mean(axis=0); sd = X.std(axis=0)
Xs = (X-mu)/sd

clf = LogisticRegression(max_iter=2000, class_weight="balanced")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
proba_cv = cross_val_predict(clf, Xs, y, cv=skf, method="predict_proba")[:,1]

clf.fit(Xs, y)
print("=== Logistic regression coefficients (standardized features, +=more BAD) ===")
for name, c in zip(feat_names, clf.coef_[0]):
    print(f"  {name:28s} {c:+.3f}")
print(f"  intercept {clf.intercept_[0]:+.3f}")

print()
print("=== 5-fold CV performance across thresholds ===")
print(f"{'thr':>5} {'acc':>6} {'sens(bad)':>10} {'spec(good)':>11} {'flagged%':>9}")
for thr in [0.3,0.4,0.5,0.6,0.7]:
    pred = (proba_cv>=thr).astype(int)
    tp = ((pred==1)&(y==1)).sum(); tn=((pred==0)&(y==0)).sum()
    fp = ((pred==1)&(y==0)).sum(); fn=((pred==0)&(y==1)).sum()
    acc = (tp+tn)/len(y)
    sens = tp/(tp+fn)  # correctly flags bad
    spec = tn/(tn+fp)  # correctly keeps good
    flagged = pred.mean()*100
    print(f"{thr:5.2f} {acc*100:5.1f}% {sens*100:9.1f}% {spec*100:10.1f}% {flagged:8.1f}%")

# choose a working threshold: prioritize keeping GOOD data (high specificity), like NCC cutoff logic,
# since this mask will be used to RESTRICT analysis to trustworthy windows
THR = 0.5
pred_cv = (proba_cv>=THR).astype(int)
tp = ((pred_cv==1)&(y==1)).sum(); tn=((pred_cv==0)&(y==0)).sum()
fp = ((pred_cv==1)&(y==0)).sum(); fn=((pred_cv==0)&(y==1)).sum()
print()
print(f"=== Confusion matrix at thr={THR} (cross-validated, out-of-fold) ===")
print(f"                 pred GOOD   pred BAD")
print(f"  actual GOOD    {tn:8d}   {fp:8d}")
print(f"  actual BAD     {fn:8d}   {tp:8d}")
print(f"  accuracy={ (tp+tn)/len(y)*100:.1f}%  sensitivity(bad caught)={tp/(tp+fn)*100:.1f}%  specificity(good kept)={tn/(tn+fp)*100:.1f}%")

with open("/tmp/mask_v4_composite.pkl","wb") as f:
    pickle.dump(dict(tc=tc, y=y, proba_cv=proba_cv, X=X, feat_names=feat_names,
                      mu=mu, sd=sd, coef=clf.coef_[0], intercept=clf.intercept_[0],
                      THR=THR), f)
print("\nsaved /tmp/mask_v4_composite.pkl")
