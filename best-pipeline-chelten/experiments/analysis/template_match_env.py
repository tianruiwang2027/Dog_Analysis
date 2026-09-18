import pickle
import numpy as np
from scipy.signal import find_peaks, fftconvolve
from scipy.ndimage import uniform_filter1d

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
with open("/tmp/singles_labels.pkl","rb") as f:
    lab = pickle.load(f)

ts_sd = d["ts_sd"]; env_sd = d["env_sd"]; fsd_s = d["fsd_s"]
tp_times = lab["tp_times"]; fp_times = lab["fp_times"]

HALF_S = 0.20  # +-200ms window around each candidate, in envelope-sample space
HALF_N = int(HALF_S*fsd_s)
L = 2*HALF_N + 1

def snippet(t):
    i = np.searchsorted(ts_sd, t)
    if i-HALF_N < 0 or i+HALF_N+1 > len(env_sd):
        return None
    return env_sd[i-HALF_N:i+HALF_N+1].copy()

# build template from a training subset of TPs (first half by time), evaluate on held-out second half + all FPs
order = np.argsort(tp_times)
tp_sorted = tp_times[order]
n_train = len(tp_sorted)//2
train_tp = tp_sorted[:n_train]
test_tp = tp_sorted[n_train:]

snips = [snippet(t) for t in train_tp]
snips = [s for s in snips if s is not None]
snips = np.array(snips)
print("n template-building snippets:", len(snips))

# normalize each snippet (unit peak) before averaging, to avoid amplitude-dominant beats swamping shape
snips_norm = snips / (snips.max(axis=1, keepdims=True) + 1e-12)
template = snips_norm.mean(axis=0)
template = template - template.mean()
template_energy = np.sqrt((template**2).sum())
print("template built, length", len(template), "energy", template_energy)

with open("/tmp/env_template.pkl","wb") as f:
    pickle.dump(dict(template=template, HALF_N=HALF_N, fsd_s=fsd_s), f)

def ncc_score(t):
    s = snippet(t)
    if s is None: return np.nan
    s = s - s.mean()
    denom = np.sqrt((s**2).sum()) * template_energy
    if denom < 1e-12: return np.nan
    return float((s*template).sum() / denom)

test_tp_scores = np.array([ncc_score(t) for t in test_tp])
fp_scores = np.array([ncc_score(t) for t in fp_times])
print(f"\nheld-out TP NCC score: mean={np.nanmean(test_tp_scores):.3f} median={np.nanmedian(test_tp_scores):.3f}  n={len(test_tp_scores)}")
print(f"FP NCC score:          mean={np.nanmean(fp_scores):.3f} median={np.nanmedian(fp_scores):.3f}  n={len(fp_scores)}")

for cutoff in [0.3,0.4,0.5,0.6,0.7,0.8]:
    tp_pass = np.nanmean(test_tp_scores>=cutoff)*100
    fp_pass = np.nanmean(fp_scores>=cutoff)*100
    print(f"cutoff={cutoff:.1f}  TP pass={tp_pass:5.1f}%   FP pass={fp_pass:5.1f}%")
