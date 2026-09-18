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
    if i-HALF_N < 0 or i+HALF_N+1 > len(env_sd):
        return np.nan
    s = env_sd[i-HALF_N:i+HALF_N+1]
    s = s - s.mean()
    denom = np.sqrt((s**2).sum()) * template_energy
    if denom < 1e-12: return np.nan
    return float((s*template).sum() / denom)

singles_good = np.sort(lab["singles_good"])
tp_set = set(lab["tp_times"].tolist())
fp_set = set(lab["fp_times"].tolist())
scores = np.array([ncc_score(t) for t in singles_good])

CUTOFF = 0.5
removed_mask = scores < CUTOFF
kept_mask = ~removed_mask

removed = singles_good[removed_mask]
is_tp = np.array([t in tp_set for t in removed])
is_fp = np.array([t in fp_set for t in removed])
print(f"Candidates REMOVED by template filter (score<{CUTOFF}): {len(removed)}")
print(f"  of which TRUE positives removed (cost): {is_tp.sum()}")
print(f"  of which FALSE positives removed (benefit): {is_fp.sum()}")

kept = singles_good[kept_mask]
is_tp_k = np.array([t in tp_set for t in kept])
is_fp_k = np.array([t in fp_set for t in kept])
print(f"\nCandidates KEPT: {len(kept)}  TP kept={is_tp_k.sum()}  FP kept={is_fp_k.sum()}")

print(f"\nNet effect: removed {is_fp.sum()} of {len(fp_set)} false positives ({is_fp.sum()/len(fp_set)*100:.0f}%),")
print(f"            at the cost of {is_tp.sum()} of {len(tp_set)} true positives ({is_tp.sum()/len(tp_set)*100:.1f}%)")

import datetime
removed_fp_times = removed[is_fp]
removed_tp_times = removed[is_tp]
print("\nFirst 15 removed FALSE POSITIVES (times):")
for t in removed_fp_times[:15]:
    print(" ", datetime.datetime.utcfromtimestamp(t/1e6), " score=", round(ncc_score(t),3))
print("\nFirst 15 removed TRUE POSITIVES (times) -- these are the 'cost':")
for t in removed_tp_times[:15]:
    print(" ", datetime.datetime.utcfromtimestamp(t/1e6), " score=", round(ncc_score(t),3))

with open("/tmp/failure_breakdown.pkl","wb") as f:
    pickle.dump(dict(removed_fp_times=removed_fp_times, removed_tp_times=removed_tp_times,
                      kept=kept, removed=removed, scores=scores, singles_good=singles_good), f)
