import pickle, datetime
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/mask_v3_all_features.pkl","rb") as f:
    feats = pickle.load(f)
with open("/tmp/mask_v4_composite.pkl","rb") as f:
    M = pickle.load(f)
with open("/tmp/mask_v4_apply_and_validate.pkl","rb") as f:
    V = pickle.load(f)

good = [f for f in feats if f["label"]=="good"]
bad  = [f for f in feats if f["label"]=="bad"]
g_drift = np.array([f["drift_amp"] for f in good]); g_cbp = np.array([f["cardiac_bp_amp"] for f in good])
b_drift = np.array([f["drift_amp"] for f in bad]);  b_cbp = np.array([f["cardiac_bp_amp"] for f in bad])
g_s1s2 = np.array([f["s1s2_std_ms"] for f in good], dtype=float)
b_s1s2 = np.array([f["s1s2_std_ms"] for f in bad], dtype=float)

fig = plt.figure(figsize=(13,10))
gs = fig.add_gridspec(3, 2, height_ratios=[1.3,1,0.7])

# Panel A: feature-space separation (log-log)
axA = fig.add_subplot(gs[0,0])
axA.scatter(g_cbp, g_drift, s=8, alpha=0.35, color="tab:green", label=f"hand GOOD (n={len(good)})")
axA.scatter(b_cbp, b_drift, s=8, alpha=0.35, color="tab:red", label=f"hand BAD (n={len(bad)})")
axA.set_xscale("log"); axA.set_yscale("log")
axA.set_xlabel("cardiac_bp_amp = bandpassed(10-100Hz) p99-p1 amplitude (raw units)")
axA.set_ylabel("drift_amp = low-pass(<1Hz) baseline peak-to-peak (raw units)")
axA.set_title("A. Raw amplitude features separate GOOD vs BAD windows\n(bad = louder in every band -- motion is broadband)")
axA.legend(fontsize=8, loc="upper left")

# Panel B: s1s2 timing consistency
axB = fig.add_subplot(gs[0,1])
bins = np.linspace(0,100,41)
axB.hist(g_s1s2[~np.isnan(g_s1s2)], bins=bins, color="tab:green", alpha=0.5, density=True, label="hand GOOD")
axB.hist(b_s1s2[~np.isnan(b_s1s2)], bins=bins, color="tab:red", alpha=0.5, density=True, label="hand BAD")
axB.axvline(np.nanmedian(g_s1s2), color="tab:green", ls="--", lw=1.5)
axB.axvline(np.nanmedian(b_s1s2), color="tab:red", ls="--", lw=1.5)
axB.set_xlabel("s1s2_std_ms  (std-dev of S1->S2 [100-320ms] inter-peak gaps within window)")
axB.set_title(f"B. Timing consistency\nGOOD median={np.nanmedian(g_s1s2):.0f}ms   BAD median={np.nanmedian(b_s1s2):.0f}ms")
axB.legend(fontsize=8)

# Panel C: threshold sweep (sensitivity/specificity/accuracy) using cross-validated probabilities
y = M["tc"]*0  # placeholder, recompute below from pkl
with open("/tmp/mask_v3_all_features.pkl","rb") as f:
    feats2 = pickle.load(f)
yb = np.array([1 if f["label"]=="bad" else 0 for f in feats2])
proba = M["proba_cv"]
thrs = np.linspace(0.1,0.9,33)
accs, senss, specs = [], [], []
for thr in thrs:
    pred = (proba>=thr).astype(int)
    tp=((pred==1)&(yb==1)).sum(); tn=((pred==0)&(yb==0)).sum()
    fp=((pred==1)&(yb==0)).sum(); fn=((pred==0)&(yb==1)).sum()
    accs.append((tp+tn)/len(yb)); senss.append(tp/(tp+fn)); specs.append(tn/(tn+fp))
axC = fig.add_subplot(gs[1,0])
axC.plot(thrs, accs, label="accuracy", color="black")
axC.plot(thrs, senss, label="sensitivity (bad correctly caught)", color="tab:red")
axC.plot(thrs, specs, label="specificity (good correctly kept)", color="tab:green")
axC.axvline(M["THR"], color="tab:blue", ls="--", lw=1, label=f"chosen thr={M['THR']}")
axC.set_xlabel("classifier probability threshold (flag as BAD if p >= thr)")
axC.set_ylabel("rate")
axC.set_title("C. 5-fold cross-validated performance vs threshold\n(logistic regression on log-amplitude + timing features)")
axC.legend(fontsize=7.5, loc="lower left")
axC.set_ylim(0.5,1.0)

# Panel D: coefficients bar chart
axD = fig.add_subplot(gs[1,1])
names = M["feat_names"]; coefs = M["coef"]
ypos = np.arange(len(names))
colors = ["tab:red" if c>0 else "tab:green" for c in coefs]
axD.barh(ypos, coefs, color=colors)
axD.set_yticks(ypos); axD.set_yticklabels(names, fontsize=8)
axD.axvline(0, color="black", lw=0.7)
axD.set_xlabel("standardized coefficient  (+ = pushes toward BAD)")
axD.set_title("D. What the mask weighs")

# Panel E: timeline ribbon, hand vs auto, across full recording
axE = fig.add_subplot(gs[2,:])
scg_tmid = V["scg_tmid"]; scg_v_hand = V["scg_v_hand"]; scg_v_auto = V["scg_v_auto"]
t0 = scg_tmid.min();
th = (scg_tmid-t0)/1e6/60  # minutes
order = np.argsort(scg_tmid)
th_s = th[order]; hv = scg_v_hand[order].astype(int); av = scg_v_auto[order].astype(int)
axE.fill_between(th_s, 1.0, 1.9, where=hv.astype(bool), color="tab:green", step="mid", alpha=0.8)
axE.fill_between(th_s, 1.0, 1.9, where=~hv.astype(bool), color="tab:red", step="mid", alpha=0.4)
axE.fill_between(th_s, 0.0, 0.9, where=av.astype(bool), color="tab:green", step="mid", alpha=0.8)
axE.fill_between(th_s, 0.0, 0.9, where=~av.astype(bool), color="tab:red", step="mid", alpha=0.4)
axE.set_yticks([0.45,1.45]); axE.set_yticklabels(["AUTO mask","HAND label"])
axE.set_xlabel("minutes from start of comparison window")
axE.set_title("E. Hand-labeled good/bad (top) vs automated composite mask (bottom) across the full session  (green=good, red=bad)")

plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_quality_mask_v4.png", dpi=130)
print("saved")
