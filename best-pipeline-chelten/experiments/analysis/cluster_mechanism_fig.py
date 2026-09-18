import pickle, datetime, sqlite3
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

with open("/tmp/combined_detector.pkl","rb") as f:
    c = pickle.load(f)
with open("/tmp/final_full_comparison_data.pkl","rb") as f:
    D = pickle.load(f)

UTC = datetime.timezone.utc
def dn(us):
    return mdates.date2num([datetime.datetime.fromtimestamp(t/1e6, tz=UTC) for t in np.atleast_1d(us)])

det = np.sort(c["primary"])
def beat_hr(peaks):
    rr = np.diff(peaks)/1e6; hr = 60.0/rr
    tmid = peaks[:-1] + np.diff(peaks)//2
    return tmid, hr
def smooth(tmid, hr, win_s=3.0):
    win_us = int(win_s*1e6)
    out = np.empty(len(tmid))
    for i,t in enumerate(tmid):
        sel = (tmid>=t-win_us)&(tmid<=t+win_us)
        out[i] = hr[sel].mean()
    return out

tmid, hr = beat_hr(det)
hr_sm = smooth(tmid, hr)

ecg_tmid = D["ecg_tmid"]; ecg_hr_sm = D["ecg_hr_sm"]; LAG_US = D["LAG_US"]

lo_dt = datetime.datetime(2026,6,26,17,27,12,tzinfo=UTC)
hi_dt = datetime.datetime(2026,6,26,17,27,37,tzinfo=UTC)
lo = int(lo_dt.timestamp()*1e6); hi = int(hi_dt.timestamp()*1e6)

sel = (tmid>=lo)&(tmid<=hi)
sel_ecg = (ecg_tmid-LAG_US>=lo)&(ecg_tmid-LAG_US<=hi)

fig, ax = plt.subplots(figsize=(12,6))
ax.plot(dn(ecg_tmid[sel_ecg]-LAG_US), ecg_hr_sm[sel_ecg], color="black", lw=1.3, label="true ECG HR (3s-smoothed)")
ax.plot(dn(tmid[sel]), hr_sm[sel], "o-", color="tab:green", ms=4, lw=1.1, label="best-pipeline HR (3s-smoothed) -- what gets compared")
ax.plot(dn(tmid[sel]), hr[sel], "x", color="tab:red", ms=7, mew=1.6, label="best-pipeline instantaneous HR (60/RR), one point per detected beat")

# annotate the long-RR / missed-beat points
rr_ms = np.diff(det)/1e3
for i in np.where(sel)[0]:
    if rr_ms[i] > 1200:
        ax.annotate(f"RR={rr_ms[i]:.0f}ms\n(missed beat)", (dn(tmid[i])[0], hr[i]),
                    textcoords="offset points", xytext=(0,-28), fontsize=8, color="tab:red", ha="center")

ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=UTC))
ax.set_ylabel("HR (bpm)")
ax.set_title("Cluster near 17:27:20 -- a couple of missed beats crash the instantaneous HR,\nand 3s box-car smoothing spreads that crash into several neighboring \"good\" points")
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(0,110)
plt.tight_layout()
out = "/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_cluster_mechanism.png"
plt.savefig(out, dpi=140)
print("->", out)
