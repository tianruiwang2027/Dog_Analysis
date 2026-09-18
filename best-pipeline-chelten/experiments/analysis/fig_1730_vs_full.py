import pickle, datetime
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

with open("/tmp/compare_1730_1800.pkl","rb") as f:
    N = pickle.load(f)
with open("/tmp/final_4way_compare.pkl","rb") as f:
    D = pickle.load(f)

UTC = datetime.timezone.utc
def dn(us):
    return mdates.date2num([datetime.datetime.fromtimestamp(t/1e6, tz=UTC) for t in np.atleast_1d(us)])

t0_old, t1_old, LAG_US = D["t0"], D["t1"], D["LAG_US"]
t0_new, t1_new = N["t0"], N["t1"]

fig = plt.figure(figsize=(16,9))
gs = fig.add_gridspec(2, 2, height_ratios=[1.3,1], hspace=0.32, wspace=0.25)

ax0 = fig.add_subplot(gs[0,:])
ecg_tmid, ecg_hr_sm, ecg_v = D["ecg_tmid"], D["ecg_hr_sm"], D["ecg_v"]
m_ecg = ecg_v & (ecg_tmid-LAG_US>=t0_old-5_000_000) & (ecg_tmid-LAG_US<=t1_old+5_000_000)
ax0.plot(dn(ecg_tmid[m_ecg]-LAG_US), ecg_hr_sm[m_ecg], color="black", lw=1.0, label="hand-clicked ECG")
scg_tmid, scg_hr_sm, scg_v = D["scg_tmid"], D["scg_hr_sm"], D["scg_v"]
m_scg = scg_v & (scg_tmid>=t0_old) & (scg_tmid<=t1_old)
ax0.plot(dn(scg_tmid[m_scg]), scg_hr_sm[m_scg], color="tab:gray", lw=0.9, ls="--", label="hand-clicked SCG")
ax0.axvspan(dn(t0_old)[0], dn(t0_new)[0], color="red", alpha=0.12, label="dropped in the 17:30-18:00 comparison")
ax0.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=UTC))
ax0.set_ylabel("3s-smoothed HR (bpm)")
ax0.set_title("Full recording, with the 17:26:30-17:30:00 segment that gets dropped shaded")
ax0.legend(loc="upper right", fontsize=9)

lim=[40,140]
ax1 = fig.add_subplot(gs[1,0])
xa_o, yb_o = D["xa_scg"], D["yb_scg"]
ax1.scatter(yb_o, xa_o, s=6, alpha=0.3, color="0.4")
ax1.plot(lim,lim,"k-",lw=1,alpha=0.6)
ax1.set_xlim(lim); ax1.set_ylim(lim); ax1.set_aspect("equal")
ax1.set_xlabel("Hand ECG HR (bpm)"); ax1.set_ylabel("Hand SCG HR (bpm)")
s_o = D["s_scg"]
ax1.set_title(f"FULL window 17:26:30-18:00:00\nr={s_o['r']:.3f}  MAE={s_o['mae']:.2f}  bias={s_o['bias']:+.2f}  n={s_o['n']}")
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[1,1])
xa_n, yb_n, s_n = N["xa"], N["yb"], N["s"]
ax2.scatter(yb_n, xa_n, s=6, alpha=0.3, color="tab:blue")
ax2.plot(lim,lim,"k-",lw=1,alpha=0.6)
ax2.set_xlim(lim); ax2.set_ylim(lim); ax2.set_aspect("equal")
ax2.set_xlabel("Hand ECG HR (bpm)"); ax2.set_ylabel("Hand SCG HR (bpm)")
ax2.set_title(f"RESTRICTED window 17:30:00-18:00:00\nr={s_n['r']:.3f}  MAE={s_n['mae']:.2f}  bias={s_n['bias']:+.2f}  n={s_n['n']}")
ax2.grid(True, alpha=0.3)

fig.suptitle("[Chelten] Hand SCG vs hand ECG: full window vs. dropping the first 3.5 minutes", fontsize=13)
out = "/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_1730_vs_full.png"
fig.savefig(out, dpi=135)
print("->", out)
