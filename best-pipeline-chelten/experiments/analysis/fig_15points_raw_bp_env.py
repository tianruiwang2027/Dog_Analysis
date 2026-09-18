import pickle, datetime
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
with open("/tmp/illustrate_15missed.pkl","rb") as f:
    I = pickle.load(f)
with open("/tmp/combined_detector.pkl","rb") as f:
    c = pickle.load(f)

ts_s = d["ts_s"]; x_s = d["x_s"]; xf_s = d["xf_s"]; fs_s = d["fs_s"]
ts_sd = d["ts_sd"]; sharp_s = d["sharp_s"]; fsd_s = d["fsd_s"]

det = np.sort(c["primary"])
ta = I["ta"]; bad_idx = I["bad_idx"]
pts = np.sort(ta[bad_idx])

UTC = datetime.timezone.utc
def fmt(us):
    return datetime.datetime.fromtimestamp(us/1e6, tz=UTC).strftime("%H:%M:%S.%f")[:-3]

WIN_US = int(2.5e6)  # +/-2.5s around each point

n = len(pts)
fig, axs = plt.subplots(n, 3, figsize=(15, 2.0*n))

for row, t_center in enumerate(pts):
    lo = np.searchsorted(ts_s, t_center-WIN_US); hi = np.searchsorted(ts_s, t_center+WIN_US)
    t_ms = (ts_s[lo:hi]-t_center)/1e3

    lo2 = np.searchsorted(ts_sd, t_center-WIN_US); hi2 = np.searchsorted(ts_sd, t_center+WIN_US)
    t_ms2 = (ts_sd[lo2:hi2]-t_center)/1e3

    # detected beats (best-pipeline) inside window, for reference ticks
    det_in = det[(det>=t_center-WIN_US)&(det<=t_center+WIN_US)]

    ax = axs[row,0]
    ax.plot(t_ms, x_s[lo:hi], color="tab:gray", lw=0.5)
    ax.set_ylabel(fmt(t_center), fontsize=8, rotation=0, ha="right", va="center")
    if row==0: ax.set_title("RAW", fontsize=10)
    for dt in det_in: ax.axvline((dt-t_center)/1e3, color="tab:green", lw=0.6, alpha=0.5)

    ax = axs[row,1]
    ax.plot(t_ms, xf_s[lo:hi], color="tab:purple", lw=0.5)
    if row==0: ax.set_title("BANDPASSED 10-100Hz", fontsize=10)
    for dt in det_in: ax.axvline((dt-t_center)/1e3, color="tab:green", lw=0.6, alpha=0.5)

    ax = axs[row,2]
    ax.plot(t_ms2, sharp_s[lo2:hi2], color="tab:blue", lw=0.9)
    ax.axhline(0.3, color="red", ls="--", lw=0.8)
    if row==0: ax.set_title("SHARPENED ENVELOPE (0.3 = detection threshold)", fontsize=10)
    for dt in det_in: ax.axvline((dt-t_center)/1e3, color="tab:green", lw=0.6, alpha=0.5)

    for ax in axs[row]:
        ax.set_xlim(-WIN_US/1e3, WIN_US/1e3)
        ax.tick_params(labelsize=7)

for ax in axs[-1]:
    ax.set_xlabel("ms from the point's timestamp", fontsize=8)

fig.suptitle("Raw / bandpassed / envelope around each of the 15 missed-beat artifact points\n(green vertical lines = beats the best-pipeline DID detect nearby; gaps between them = the miss)", fontsize=12)
plt.tight_layout(rect=[0,0,1,0.97])
out = "/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_15points_raw_bp_env.png"
plt.savefig(out, dpi=120)
print("->", out)
