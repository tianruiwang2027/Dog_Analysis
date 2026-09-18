import pickle
import numpy as np

with open("/tmp/singles_labels.pkl","rb") as f:
    lab = pickle.load(f)
with open("/tmp/env_template_wide.pkl","rb") as f:
    tpl = pickle.load(f)
with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)

ts_sd = d["ts_sd"]; env_sd = d["env_sd"]
template = tpl["template"]; HALF_N = tpl["HALF_N"]
template_energy = np.sqrt((template**2).sum())
def ncc_score(t):
    i = np.searchsorted(ts_sd, t)
    if i-HALF_N < 0 or i+HALF_N+1 > len(env_sd): return np.nan
    s = env_sd[i-HALF_N:i+HALF_N+1]; s=s-s.mean()
    denom = np.sqrt((s**2).sum())*template_energy
    return float((s*template).sum()/denom) if denom>1e-12 else np.nan

fp_times = np.sort(lab["fp_times"])
scores = np.array([ncc_score(t) for t in fp_times])
CUTOFF = 0.5
survived = fp_times[scores>=CUTOFF]
survived_scores = scores[scores>=CUTOFF]
caught = fp_times[scores<CUTOFF]
print(f"Surviving (undetected) FPs: {len(survived)} of {len(fp_times)} ({len(survived)/len(fp_times)*100:.0f}%)")
print(f"Caught FPs: {len(caught)}")
print(f"\nSurviving FP score stats: mean={survived_scores.mean():.3f} median={np.median(survived_scores):.3f} min={survived_scores.min():.3f} max={survived_scores.max():.3f}")

# distance to nearest hand-clicked SCG beat (regardless of the +-100ms match tolerance used for TP/FP labeling)
hand_good = np.sort(lab["hand_good"])
def nearest_dist(t):
    idx = np.searchsorted(hand_good, t)
    cands = []
    if idx>0: cands.append(abs(hand_good[idx-1]-t))
    if idx<len(hand_good): cands.append(abs(hand_good[idx]-t))
    return min(cands) if cands else np.nan

dists = np.array([nearest_dist(t) for t in survived])/1000  # ms
print(f"\nDistance from surviving FPs to nearest hand click (ms):")
for lo,hi in [(0,100),(100,200),(200,300),(300,500),(500,1000),(1000,1e9)]:
    m = (dists>=lo)&(dists<hi)
    print(f"  {lo:5.0f}-{hi:5.0f}ms: {m.sum():3d}  ({m.sum()/len(dists)*100:.0f}%)")

print(f"\nmedian distance to nearest hand click: {np.median(dists):.0f}ms")
print(f"fraction within 100-300ms (plausible 'near miss'/duplicate of a real beat): {np.mean((dists>=100)&(dists<300))*100:.0f}%")

with open("/tmp/surviving_fp.pkl","wb") as f:
    pickle.dump(dict(survived=survived, survived_scores=survived_scores, dists=dists, caught=caught), f)
