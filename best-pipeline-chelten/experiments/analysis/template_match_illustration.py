import pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/env_template.pkl","rb") as f:
    tpl = pickle.load(f)
with open("/tmp/singles_labels.pkl","rb") as f:
    lab = pickle.load(f)
with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)

template = tpl["template"]; HALF_N = tpl["HALF_N"]; fsd_s = tpl["fsd_s"]
ts_sd = d["ts_sd"]; env_sd = d["env_sd"]
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

tp_scores = np.array([ncc_score(t) for t in lab["tp_times"]])
fp_scores = np.array([ncc_score(t) for t in lab["fp_times"]])

fig, axs = plt.subplots(1,2, figsize=(13,4.5))
t_ms = (np.arange(len(template))-HALF_N)/fsd_s*1000
axs[0].plot(t_ms, template, color="tab:purple")
axs[0].set_title("Learned canonical single-beat\nenvelope template (avg of 745 confirmed beats)")
axs[0].set_xlabel("ms from candidate peak")
axs[0].axvline(0, color="gray", lw=0.7, ls="--")

bins = np.linspace(-1,1,41)
axs[1].hist(tp_scores[~np.isnan(tp_scores)], bins=bins, alpha=0.6, label=f"true positives (n={len(tp_scores)})", color="tab:green", density=True)
axs[1].hist(fp_scores[~np.isnan(fp_scores)], bins=bins, alpha=0.6, label=f"false positives (n={len(fp_scores)})", color="tab:red", density=True)
axs[1].axvline(0.65, color="black", ls="--", lw=1, label="best cutoff (0.65)")
axs[1].set_title("Matched-filter (NCC) score:\ntrue vs false positive candidates")
axs[1].set_xlabel("normalized cross-correlation score")
axs[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_template_match_illustration.png", dpi=130)
print("saved")
print("TP score mean/median:", np.nanmean(tp_scores), np.nanmedian(tp_scores))
print("FP score mean/median:", np.nanmean(fp_scores), np.nanmedian(fp_scores))
