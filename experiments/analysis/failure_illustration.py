import pickle, datetime
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
with open("/tmp/failure_breakdown.pkl","rb") as f:
    fb = pickle.load(f)
with open("/tmp/env_template_wide.pkl","rb") as f:
    tpl = pickle.load(f)

ts_s = d["ts_s"]; xf_s = d["xf_s"]; fs_s = d["fs_s"]
ts_sd = d["ts_sd"]; env_sd = d["env_sd"]; fsd_s = d["fsd_s"]
template = tpl["template"]; HALF_N = tpl["HALF_N"]

fp_example = fb["removed_fp_times"][0]   # 17:28:52.319401
tp_example = fb["removed_tp_times"][0]   # 17:58:39.488809

def plot_event(ax_raw, ax_env, t_center, label, color):
    # raw bandpassed, +-400ms
    lo = np.searchsorted(ts_s, t_center - int(0.4e6))
    hi = np.searchsorted(ts_s, t_center + int(0.4e6))
    t_ms = (ts_s[lo:hi]-t_center)/1e3
    ax_raw.plot(t_ms, xf_s[lo:hi], color=color, lw=0.7)
    ax_raw.axvline(0, color="gray", ls="--", lw=0.6)
    ax_raw.set_title(f"{label}\nraw bandpassed c1, {datetime.datetime.utcfromtimestamp(t_center/1e6)}")

    # envelope +- 400ms, overlay scaled template for comparison
    i = np.searchsorted(ts_sd, t_center)
    lo2 = i - HALF_N; hi2 = i + HALF_N + 1
    t_ms2 = (np.arange(lo2,hi2)-i)/fsd_s*1000
    snippet = env_sd[lo2:hi2]
    ax_env.plot(t_ms2, snippet/ (snippet.max()+1e-9), color=color, lw=1.3, label="this candidate (normalized)")
    ax_env.plot(t_ms2, (template-template.min())/(template.max()-template.min()), color="black", lw=1, ls="--", alpha=0.6, label="canonical template shape")
    ax_env.axvline(0, color="gray", ls="--", lw=0.6)
    ax_env.legend(fontsize=7, loc="upper right")

fig, axs = plt.subplots(2,2, figsize=(13,7.5))
plot_event(axs[0,0], axs[0,1], fp_example, "Removed FALSE POSITIVE (score=0.479) -- correctly rejected", "tab:red")
plot_event(axs[1,0], axs[1,1], tp_example, "Removed TRUE POSITIVE (score=0.344) -- incorrectly rejected (cost)", "tab:orange")
axs[1,0].set_xlabel("ms from candidate peak")
axs[1,1].set_xlabel("ms from candidate peak")
plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_failure_examples.png", dpi=130)
print("saved")
