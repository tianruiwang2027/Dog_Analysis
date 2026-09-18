import pickle
import numpy as np

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
with open("/tmp/singles_labels.pkl","rb") as f:
    lab = pickle.load(f)
with open("/tmp/env_template_wide.pkl","rb") as f:
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

order = np.argsort(lab["tp_times"])
tp_sorted = lab["tp_times"][order]
n_train = len(tp_sorted)//2
test_tp = tp_sorted[n_train:]
fp_times = lab["fp_times"]

tp_scores = np.array([ncc_score(t) for t in test_tp])
fp_scores = np.array([ncc_score(t) for t in fp_times])
print("WIDE (+-400ms, two-lobe) template:")
print(f"held-out TP: mean={np.nanmean(tp_scores):.3f} median={np.nanmedian(tp_scores):.3f}")
print(f"FP:          mean={np.nanmean(fp_scores):.3f} median={np.nanmedian(fp_scores):.3f}")
for cutoff in [0.5,0.6,0.65,0.7,0.75,0.8]:
    tp_pass = np.nanmean(tp_scores>=cutoff)*100
    fp_pass = np.nanmean(fp_scores>=cutoff)*100
    print(f"cutoff={cutoff:.2f}  TP pass={tp_pass:5.1f}%  FP pass={fp_pass:5.1f}%")
