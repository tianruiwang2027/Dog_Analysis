import pickle, sqlite3
import numpy as np
from scipy.signal import find_peaks

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
with open("/tmp/singles_labels.pkl","rb") as f:
    lab = pickle.load(f)
with open("/tmp/env_template.pkl","rb") as f:
    tpl = pickle.load(f)

ts_sd = d["ts_sd"]; env_sd = d["env_sd"]; fsd_s = d["fsd_s"]
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

singles_good = lab["singles_good"]
hand_good = lab["hand_good"]
tp_times = set(lab["tp_times"].tolist())
fp_times = set(lab["fp_times"].tolist())

scores = np.array([ncc_score(t) for t in singles_good])
is_tp = np.array([t in tp_times for t in singles_good])
is_fp = np.array([t in fp_times for t in singles_good])
print("sanity: is_tp+is_fp count", is_tp.sum()+is_fp.sum(), "vs n", len(singles_good))

for cutoff in [0.0, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]:
    keep = scores >= cutoff
    n_keep = keep.sum()
    tp_kept = (is_tp & keep).sum()
    fp_kept = (is_fp & keep).sum()
    fp_rate = fp_kept/n_keep*100 if n_keep else float('nan')
    match_rate = tp_kept/len(hand_good)*100
    print(f"cutoff={cutoff:.2f}  n_kept={n_keep:4d}  TP_kept={tp_kept:4d}  FP_kept={fp_kept:3d}  FP_rate={fp_rate:5.1f}%  match_rate={match_rate:5.1f}%")
