import pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/env_template_wide.pkl","rb") as f:
    tpl = pickle.load(f)
with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
with open("/tmp/singles_labels.pkl","rb") as f:
    lab = pickle.load(f)

template = tpl["template"]; HALF_N = tpl["HALF_N"]; fsd_s = tpl["fsd_s"]
ts_sd = d["ts_sd"]; env_sd = d["env_sd"]
template_energy = np.sqrt((template**2).sum())

def ncc_score(t):
    i = np.searchsorted(ts_sd, t)
    if i-HALF_N < 0 or i+HALF_N+1 > len(env_sd): return np.nan
    s = env_sd[i-HALF_N:i+HALF_N+1]; s = s - s.mean()
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

t_ms = (np.arange(len(template))-HALF_N)/fsd_s*1000

fig, axs = plt.subplots(1, 2, figsize=(13,5))

ax = axs[0]
ax.plot(t_ms, template, color="tab:blue", lw=1.8)
ax.axvline(0, color="gray", ls=":", lw=1)
ax.annotate("main lobe (S1)", (0, template.max()), textcoords="offset points", xytext=(-10,8), fontsize=9, ha="center")
# find secondary lobe peak in 150-280ms range
mask2 = (t_ms>150)&(t_ms<280)
i2 = np.argmax(template[mask2])
t2 = t_ms[mask2][i2]; v2 = template[mask2][i2]
ax.annotate(f"secondary lobe (S2)\n~{t2:.0f}ms later", (t2, v2), textcoords="offset points", xytext=(30,15), fontsize=9,
            arrowprops=dict(arrowstyle="->", lw=1))
ax.set_xlabel("ms from beat center")
ax.set_ylabel("template amplitude (normalized, mean-subtracted)")
ax.set_title(f"Learned two-lobe template (±400ms window,\ntrained on {n_train} true-positive beats)")

ax = axs[1]
bins = np.linspace(-0.4, 1.0, 40)
ax.hist(tp_scores[~np.isnan(tp_scores)], bins=bins, alpha=0.6, color="tab:green", label=f"held-out TRUE beats (n={len(test_tp)})\nmedian={np.nanmedian(tp_scores):.2f}")
ax.hist(fp_scores[~np.isnan(fp_scores)], bins=bins, alpha=0.6, color="tab:red", label=f"FALSE positives (n={len(fp_times)})\nmedian={np.nanmedian(fp_scores):.2f}")
ax.axvline(0.5, color="black", ls="--", lw=1.3, label="cutoff used so far (0.50)")
ax.set_xlabel("NCC score (match to two-lobe template)")
ax.set_ylabel("count")
ax.set_title("Score separation: true beats vs. false positives")
ax.legend(fontsize=8)

fig.suptitle("What the two-lobe (±400ms) template comparison looks like", fontsize=13)
plt.tight_layout()
out = "/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_wide_template_explained.png"
plt.savefig(out, dpi=140)
print("->", out)
print("TP median:", np.nanmedian(tp_scores), "FP median:", np.nanmedian(fp_scores))
