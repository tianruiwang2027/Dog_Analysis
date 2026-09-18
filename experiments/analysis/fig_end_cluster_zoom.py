import pickle, datetime
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

with open("/tmp/final_4way_compare.pkl","rb") as f:
    D = pickle.load(f)
LAG_US = D["LAG_US"]

UTC = datetime.timezone.utc
def dn(us):
    return mdates.date2num([datetime.datetime.fromtimestamp(t/1e6, tz=UTC) for t in np.atleast_1d(us)])

lo = int(datetime.datetime(2026,6,26,17,59,15,tzinfo=UTC).timestamp()*1e6)
hi = int(datetime.datetime(2026,6,26,18,0,15,tzinfo=UTC).timestamp()*1e6)

ecg_tmid, ecg_hr_sm, ecg_v = D["ecg_tmid"], D["ecg_hr_sm"], D["ecg_v"]
scg_tmid, scg_hr_sm, scg_v = D["scg_tmid"], D["scg_hr_sm"], D["scg_v"]

fig, ax = plt.subplots(figsize=(12,5.5))
m_e = ecg_v & (ecg_tmid-LAG_US>=lo) & (ecg_tmid-LAG_US<=hi)
ax.plot(dn(ecg_tmid[m_e]-LAG_US), ecg_hr_sm[m_e], color="black", lw=1.3, marker="o", ms=3, label="hand-clicked ECG")
m_s = scg_v & (scg_tmid>=lo) & (scg_tmid<=hi)
ax.plot(dn(scg_tmid[m_s]), scg_hr_sm[m_s], color="tab:gray", lw=1.1, ls="--", marker="o", ms=3, label="hand-clicked SCG")
ax.axvline(dn(int(datetime.datetime(2026,6,26,17,59,30,tzinfo=UTC).timestamp()*1e6))[0], color="tab:red", ls=":", lw=1.5, label="17:59:30 cutoff")
ax.axvspan(dn(int(datetime.datetime(2026,6,26,17,59,30,tzinfo=UTC).timestamp()*1e6))[0], dn(hi)[0], color="red", alpha=0.08)

ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=UTC))
ax.set_ylabel("3s-smoothed HR (bpm)")
ax.set_title("The dense cluster of click-count anomalies at the very end of the recording\n(shaded region = what a 17:59:30 cutoff removes)")
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
out = "/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_end_cluster_zoom.png"
plt.savefig(out, dpi=140)
print("->", out)
