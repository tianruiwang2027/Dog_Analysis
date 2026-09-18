import pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/raw_template_scores.pkl","rb") as f:
    rs = pickle.load(f)
with open("/tmp/env_template_wide.pkl","rb") as f:
    tpl = pickle.load(f)
with open("/tmp/singles_labels.pkl","rb") as f:
    lab = pickle.load(f)
with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)

ts_sd = d["ts_sd"]; env_sd = d["env_sd"]
template = tpl["template"]; HALF_N = tpl["HALF_N"]
template_energy = np.sqrt((template**2).sum())
def ncc_score(t):
    i = np.searchsorted(ts_sd, t)
    if i-HALF_N < 0 or i+HALF_N+1 > len(env_sd): return np.nan
    s = env_sd[i-HALF_N:i+HALF_N+1]; s = s-s.mean()
    denom = np.sqrt((s**2).sum())*template_energy
    return float((s*template).sum()/denom) if denom>1e-12 else np.nan

tp_times = np.sort(lab["tp_times"]); n_train=len(tp_times)//2
test_tp = tp_times[n_train:]
fp_times = lab["fp_times"]
env_tp = np.array([ncc_score(t) for t in test_tp])
env_fp = np.array([ncc_score(t) for t in fp_times])

fig, axs = plt.subplots(1,2, figsize=(12,4.5), sharey=False)
bins = np.linspace(-0.2,1,49)
axs[0].hist(rs["tp_scores"], bins=bins, alpha=0.6, color="tab:green", density=True, label=f"true positives (n={len(rs['tp_scores'])})")
axs[0].hist(rs["fp_scores"], bins=bins, alpha=0.6, color="tab:red", density=True, label=f"false positives (n={len(rs['fp_scores'])})")
axs[0].set_title(f"RAW-DOMAIN template\nTP median={np.nanmedian(rs['tp_scores']):.2f}  FP median={np.nanmedian(rs['fp_scores']):.2f}")
axs[0].set_xlabel("best-lag NCC score"); axs[0].legend(fontsize=8)

axs[1].hist(env_tp, bins=bins, alpha=0.6, color="tab:green", density=True, label=f"true positives (n={len(env_tp)})")
axs[1].hist(env_fp, bins=bins, alpha=0.6, color="tab:red", density=True, label=f"false positives (n={len(env_fp)})")
axs[1].set_title(f"ENVELOPE-DOMAIN template\nTP median={np.nanmedian(env_tp):.2f}  FP median={np.nanmedian(env_fp):.2f}")
axs[1].set_xlabel("NCC score"); axs[1].legend(fontsize=8)

fig.suptitle("Same true/false-positive candidates, scored two ways: raw waveform vs Shannon envelope")
plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_raw_vs_env_comparison.png", dpi=130)
print("saved")
