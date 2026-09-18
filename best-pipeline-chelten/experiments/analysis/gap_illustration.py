import pickle, datetime
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)

ts_s = d["ts_s"]; x_s = d["x_s"]; xf_s = d["xf_s"]; fs_s = d["fs_s"]
ts_sd = d["ts_sd"]; sharp_s = d["sharp_s"]; fsd_s = d["fsd_s"]

def to_us(dt):
    return int(dt.replace(tzinfo=datetime.timezone.utc).timestamp()*1e6)

t0 = to_us(datetime.datetime(2026,6,26,17,58,13))
t1 = to_us(datetime.datetime(2026,6,26,17,58,29))

fig, axs = plt.subplots(3,1, figsize=(13,8), sharex=True)

a = np.searchsorted(ts_s, t0); b = np.searchsorted(ts_s, t1)
t_s_axis = (ts_s[a:b]-t0)/1e6
axs[0].plot(t_s_axis, x_s[a:b], color="tab:gray", lw=0.5)
axs[0].set_title("Raw c1 -- a genuine ~12s stretch with essentially no heart-sound activity\n(inside a nominally 'good data' labeled span)")

axs[1].plot(t_s_axis, xf_s[a:b], color="tab:purple", lw=0.5)
axs[1].set_title("Bandpassed 10-100Hz -- flat, at noise floor, no cardiac bursts visible")

a2 = np.searchsorted(ts_sd, t0); b2 = np.searchsorted(ts_sd, t1)
t_sd_axis = (ts_sd[a2:b2]-t0)/1e6
axs[2].plot(t_sd_axis, sharp_s[a2:b2], color="tab:blue", lw=1.2)
axs[2].axhline(0.3, color="red", ls="--", lw=1, label="detection threshold (0.3)")
axs[2].set_title("Sharpened envelope -- never crosses threshold for ~12 seconds:\nthe algorithm has nothing to detect here, whatever the threshold or filter")
axs[2].legend(fontsize=8)
axs[2].set_xlabel("seconds from 17:58:13.000")

plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_signal_dropout_illustration.png", dpi=130)
print("saved")
