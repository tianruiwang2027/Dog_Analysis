import pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/raw_template.pkl","rb") as f:
    rt = pickle.load(f)
with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
with open("/tmp/singles_labels.pkl","rb") as f:
    lab = pickle.load(f)

template = rt["template"]; HALF_N = rt["HALF_N"]; fs_s = rt["fs_s"]; LAG_MAX_N = rt["LAG_MAX_N"]
ts_s = d["ts_s"]; xf_s = d["xf_s"]

fig, ax = plt.subplots(figsize=(10,4))
t_ms = (np.arange(len(template))-HALF_N)/fs_s*1000
ax.plot(t_ms, template, color="tab:purple")
ax.set_title(f"Raw-domain (bandpassed 10-100Hz, native {fs_s:.0f}Hz) matched-filter template\n(iteratively realigned average of 745 confirmed beats)")
ax.set_xlabel("ms from candidate peak (after alignment)")
ax.axvline(0, color="gray", ls="--", lw=0.7)
plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_raw_template_shape.png", dpi=130)
print("saved shape figure")

# --- evaluate discriminative power: NCC with small-lag search, held-out TP vs FP ---
tp_times = np.sort(lab["tp_times"]); fp_times = lab["fp_times"]
n_train = len(tp_times)//2
test_tp = tp_times[n_train:]

ref = template - template.mean()
ref_norm = np.linalg.norm(ref)

def best_ncc(t):
    i = np.searchsorted(ts_s, t)
    if i-HALF_N-LAG_MAX_N < 0 or i+HALF_N+LAG_MAX_N+1 > len(xf_s):
        return np.nan
    best = -1e18
    for lag in range(-LAG_MAX_N, LAG_MAX_N+1):
        seg = xf_s[i-HALF_N+lag : i+HALF_N+1+lag]
        seg0 = seg - seg.mean()
        denom = np.linalg.norm(seg0)*ref_norm
        if denom < 1e-9: continue
        score = (seg0*ref).sum()/denom
        if score > best: best = score
    return best

tp_scores = np.array([best_ncc(t) for t in test_tp])
fp_scores = np.array([best_ncc(t) for t in fp_times])

print(f"\nRAW-DOMAIN template (best-lag NCC within +-{LAG_MAX_N/fs_s*1000:.0f}ms):")
print(f"held-out TP: mean={np.nanmean(tp_scores):.3f} median={np.nanmedian(tp_scores):.3f}  n={len(tp_scores)}")
print(f"FP:          mean={np.nanmean(fp_scores):.3f} median={np.nanmedian(fp_scores):.3f}  n={len(fp_scores)}")
for cutoff in [0.1,0.2,0.3,0.4,0.5,0.6]:
    tp_pass = np.nanmean(tp_scores>=cutoff)*100
    fp_pass = np.nanmean(fp_scores>=cutoff)*100
    print(f"cutoff={cutoff:.2f}  TP pass={tp_pass:5.1f}%  FP pass={fp_pass:5.1f}%")

with open("/tmp/raw_template_scores.pkl","wb") as f:
    pickle.dump(dict(tp_scores=tp_scores, fp_scores=fp_scores, test_tp=test_tp, fp_times=fp_times), f)
