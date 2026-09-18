import pickle
import numpy as np

with open("/tmp/mask_v1_all_features.pkl","rb") as f:
    feats = pickle.load(f)

good = [f for f in feats if f["label"]=="good"]
bad = [f for f in feats if f["label"]=="bad"]
print(f"n good windows: {len(good)}   n bad windows: {len(bad)}")

for key in ["snr","drift_ratio","n_peaks"]:
    gv = np.array([f[key] for f in good])
    bv = np.array([f[key] for f in bad])
    print(f"\n{key}:")
    print(f"  GOOD: median={np.median(gv):.3f}  p10={np.percentile(gv,10):.3f}  p90={np.percentile(gv,90):.3f}")
    print(f"  BAD:  median={np.median(bv):.3f}  p10={np.percentile(bv,10):.3f}  p90={np.percentile(bv,90):.3f}")

gt = np.array([f["timing_ok"] for f in good])
bt = np.array([f["timing_ok"] for f in bad])
print(f"\ntiming_ok: GOOD={100*gt.mean():.1f}%   BAD={100*bt.mean():.1f}%")
