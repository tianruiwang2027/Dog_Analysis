import pickle, datetime
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
with open("/tmp/singles_labels.pkl","rb") as f:
    lab = pickle.load(f)

ts_s = d["ts_s"]; xf_s = d["xf_s"]; fs_s = d["fs_s"]
ts_sd = d["ts_sd"]; env_sd = d["env_sd"]; fsd_s = d["fsd_s"]

fp_t = 1782494924171854
click_t = 1782494923989193   # the missed hand click, ~183ms earlier

t_center = (fp_t + click_t)//2
WIN_MS = 500

fig, axs = plt.subplots(2,1, figsize=(11,7))

lo = np.searchsorted(ts_s, t_center - int(WIN_MS*1e3))
hi = np.searchsorted(ts_s, t_center + int(WIN_MS*1e3))
t_ms = (ts_s[lo:hi]-t_center)/1e3
axs[0].plot(t_ms, xf_s[lo:hi], color="tab:purple", lw=0.8)
axs[0].axvline((click_t-t_center)/1e3, color="black", ls="-", lw=1.5, label="hand-clicked beat time (this click was MISSED by our detector)")
axs[0].axvline((fp_t-t_center)/1e3, color="tab:red", ls="-", lw=1.5, label="detector's beat time (counted as a 'false positive')")
axs[0].set_title("Raw bandpassed c1: the hand-clicker and the detector picked TWO DIFFERENT real\nbursts, ~183ms apart, within the SAME cardiac cycle")
axs[0].legend(fontsize=8, loc="upper right")

lo2 = np.searchsorted(ts_sd, t_center - int(WIN_MS*1e3))
hi2 = np.searchsorted(ts_sd, t_center + int(WIN_MS*1e3))
t_ms2 = (ts_sd[lo2:hi2]-t_center)/1e3
axs[1].plot(t_ms2, env_sd[lo2:hi2], color="tab:blue", lw=1.3, marker="o", ms=3)
axs[1].axvline((click_t-t_center)/1e3, color="black", ls="-", lw=1.5, label="hand click (this cycle's chosen beat)")
axs[1].axvline((fp_t-t_center)/1e3, color="tab:red", ls="-", lw=1.5, label="detector's pick (same cycle, other heart sound)")
axs[1].set_title("Shannon envelope: BOTH bursts are real, comparable-sized peaks -- this is\nexactly why the shape/template test scores the 'false positive' highly")
axs[1].set_xlabel("ms")
axs[1].legend(fontsize=8, loc="upper right")

plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_surviving_fp_illustration.png", dpi=130)
print("saved")
