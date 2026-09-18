import pickle, datetime
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
with open("/tmp/singles_labels.pkl","rb") as f:
    lab = pickle.load(f)
with open("/tmp/env_template_wide.pkl","rb") as f:
    tpl = pickle.load(f)

ts_s = d["ts_s"]; x_s = d["x_s"]; xf_s = d["xf_s"]; fs_s = d["fs_s"]
ts_sd = d["ts_sd"]; env_sd = d["env_sd"]; fsd_s = d["fsd_s"]
template = tpl["template"]; HALF_N = tpl["HALF_N"]

# pick a clean, "typical" confirmed true-positive example (not the noisy one used before)
tp_times = np.sort(lab["tp_times"])
t_center = tp_times[len(tp_times)//2]   # a representative middle-of-the-pack example
print("illustrating candidate at", datetime.datetime.utcfromtimestamp(t_center/1e6))

WIN_MS = 500

fig, axs = plt.subplots(4, 1, figsize=(11,11), sharex=False)

# Panel 1: RAW
lo = np.searchsorted(ts_s, t_center - int(WIN_MS*1e3))
hi = np.searchsorted(ts_s, t_center + int(WIN_MS*1e3))
t_ms = (ts_s[lo:hi]-t_center)/1e3
axs[0].plot(t_ms, x_s[lo:hi], color="tab:gray", lw=0.6)
axs[0].set_title("1. RAW c1 signal (native 2000 Hz)")
axs[0].axvline(0, color="red", ls="--", lw=0.7)

# Panel 2: BANDPASSED 10-100Hz
axs[1].plot(t_ms, xf_s[lo:hi], color="tab:purple", lw=0.7)
axs[1].set_title("2. BANDPASSED 10-100Hz c1 (still native 2000 Hz) -- NOT what the template matching used")
axs[1].axvline(0, color="red", ls="--", lw=0.7)

# Panel 3: Shannon energy envelope, decimated to 200Hz + smoothed  (env_sd)
lo2 = np.searchsorted(ts_sd, t_center - int(WIN_MS*1e3))
hi2 = np.searchsorted(ts_sd, t_center + int(WIN_MS*1e3))
t_ms2 = (ts_sd[lo2:hi2]-t_center)/1e3
axs[2].plot(t_ms2, env_sd[lo2:hi2], color="tab:blue", lw=1.3, marker="o", ms=3)
axs[2].set_title("3. SHANNON ENERGY ENVELOPE, decimated to 200Hz + smoothed (env_sd)\nTHIS is the signal the template & matched-filter (NCC) operate on")
axs[2].axvline(0, color="red", ls="--", lw=0.7)

# Panel 4: this candidate's envelope snippet vs the learned template, overlaid
i = np.searchsorted(ts_sd, t_center)
snip = env_sd[i-HALF_N:i+HALF_N+1]
t_ms3 = (np.arange(-HALF_N,HALF_N+1))/fsd_s*1000
axs[3].plot(t_ms3, snip/snip.max(), color="tab:blue", lw=1.5, label="this candidate's envelope (normalized)")
axs[3].plot(t_ms3, (template-template.min())/(template.max()-template.min()), color="black", ls="--", lw=1.2, label="learned canonical template")
axs[3].set_title("4. Matched-filter comparison: candidate envelope shape vs template (this is where the NCC score comes from)")
axs[3].axvline(0, color="red", ls="--", lw=0.7)
axs[3].legend(fontsize=8)
axs[3].set_xlabel("ms from candidate peak")

for ax in axs:
    ax.set_xlim(-WIN_MS, WIN_MS)

plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_template_pipeline_illustration.png", dpi=130)
print("saved")
