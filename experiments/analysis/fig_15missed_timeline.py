import pickle, datetime
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

with open("/tmp/illustrate_15missed.pkl","rb") as f:
    I = pickle.load(f)
with open("/tmp/final_full_comparison_data.pkl","rb") as f:
    D = pickle.load(f)

UTC = datetime.timezone.utc
def dn(us):
    return mdates.date2num([datetime.datetime.fromtimestamp(t/1e6, tz=UTC) for t in np.atleast_1d(us)])

t0, t1, LAG_US = D["t0"], D["t1"], D["LAG_US"]
ecg_v, ecg_tmid, ecg_hr_sm = D["ecg_v"], D["ecg_tmid"], D["ecg_hr_sm"]
det_v, det_tmid, det_hr_sm = D["det_v"], D["det_tmid"], D["det_hr_sm"]

fig, ax = plt.subplots(figsize=(16,5.5))

m_ecg = ecg_v & (ecg_tmid-LAG_US>=t0-5_000_000) & (ecg_tmid-LAG_US<=t1+5_000_000)
ax.plot(dn(ecg_tmid[m_ecg]-LAG_US), ecg_hr_sm[m_ecg], color="black", lw=0.9, label="true ECG HR (3s-smoothed)")

m_det = det_v & (det_tmid>=t0) & (det_tmid<=t1)
ax.plot(dn(det_tmid[m_det]), det_hr_sm[m_det], color="tab:green", lw=0.9, alpha=0.85, label="best-pipeline HR (3s-smoothed)")

ta, bad_idx = I["ta"], I["bad_idx"]
ax.scatter(dn(ta[bad_idx]), I["xa"][bad_idx], s=55, facecolors="none", edgecolors="red", linewidths=1.8, zorder=5,
           label="the 15 missed-beat artifact points (err < -18 bpm)")

ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=UTC))
ax.set_ylabel("3s-smoothed HR (bpm)")
ax.set_title("[Chelten] Where the 15 missed-beat artifact points fall in the full recording (17:26:30-18:00:00)")
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(30,150)
plt.tight_layout()
out = "/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_15_missed_points.png"
plt.savefig(out, dpi=140)
print("->", out)
