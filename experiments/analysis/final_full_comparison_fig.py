import pickle, datetime
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

with open("/tmp/final_full_comparison_data.pkl","rb") as f:
    D = pickle.load(f)

UTC = datetime.timezone.utc
def dn(us):
    return mdates.date2num([datetime.datetime.fromtimestamp(t/1e6, tz=UTC) for t in np.atleast_1d(us)])

t0, t1, LAG_US = D["t0"], D["t1"], D["LAG_US"]

fig = plt.figure(figsize=(16,11))
gs = fig.add_gridspec(3, 2, height_ratios=[2.2,1,1], hspace=0.35, wspace=0.25)

# ---- Panel 1: HR over time, full window ----
ax0 = fig.add_subplot(gs[0,:])
# ECG (shifted by -LAG_US so it plots on SCG's own clock, matching how the algorithm's times compare)
ecg_v = D["ecg_v"]; ecg_tmid = D["ecg_tmid"]; ecg_hr_sm = D["ecg_hr_sm"]
m_ecg = ecg_v & (ecg_tmid-LAG_US>=t0-5_000_000) & (ecg_tmid-LAG_US<=t1+5_000_000)
ax0.plot(dn(ecg_tmid[m_ecg]-LAG_US), ecg_hr_sm[m_ecg], color="black", lw=0.9, label="hand-clicked ECG (clock-corrected onto SCG time axis)")

scg_v = D["scg_v"]; scg_tmid = D["scg_tmid"]; scg_hr_sm = D["scg_hr_sm"]
m_scg = scg_v & (scg_tmid>=t0) & (scg_tmid<=t1)
ax0.plot(dn(scg_tmid[m_scg]), scg_hr_sm[m_scg], color="0.6", lw=0.7, ls="--", label="hand-clicked SCG")

det_v = D["det_v"]; det_tmid = D["det_tmid"]; det_hr_sm = D["det_hr_sm"]
m_det = det_v & (det_tmid>=t0) & (det_tmid<=t1)
ax0.plot(dn(det_tmid[m_det]), det_hr_sm[m_det], color="tab:green", lw=0.9, alpha=0.85, label="BEST PIPELINE: envelope detector + two-lobe template filter")

ax0.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=UTC))
ax0.set_ylabel("3s-smoothed HR (bpm)")
ax0.set_title("[Chelten] Heart rate over time, 17:26:30 - 18:00:00 (good-data stretches only)")
ax0.legend(loc="upper right", fontsize=9)
ax0.set_ylim(30,150)

# shade bad-data stretches (SCG's own bad regions) lightly for context
for a,b in D["scg_ivs"]:
    pass  # good intervals; shade the gaps between them instead
scg_ivs_sorted = sorted(D["scg_ivs"])
prev_end = t0
for a,b in scg_ivs_sorted:
    if a > prev_end:
        ax0.axvspan(dn(prev_end)[0], dn(min(a,t1))[0], color="red", alpha=0.06)
    prev_end = max(prev_end, b)
if prev_end < t1:
    ax0.axvspan(dn(prev_end)[0], dn(t1)[0], color="red", alpha=0.06)

# ---- Panel 2/3 row: scatter, detector vs ECG ----
ax1 = fig.add_subplot(gs[1,0])
xa_det, yb_det, s_det = D["xa_det"], D["yb_det"], D["s_det"]
lim=[40,140]
ax1.scatter(yb_det, xa_det, s=6, alpha=0.3, color="tab:green")
ax1.plot(lim,lim,"k-",lw=1,alpha=0.6)
ax1.set_xlim(lim); ax1.set_ylim(lim); ax1.set_aspect("equal")
ax1.set_xlabel("Hand ECG HR (bpm)"); ax1.set_ylabel("Best-pipeline SCG HR (bpm)")
ax1.set_title(f"Best pipeline vs hand ECG\nr={s_det['r']:.3f}  MAE={s_det['mae']:.2f}  bias={s_det['bias']:+.2f}  n={s_det['n']}")
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[1,1])
xa_scg, yb_scg, s_scg = D["xa_scg"], D["yb_scg"], D["s_scg"]
ax2.scatter(yb_scg, xa_scg, s=6, alpha=0.3, color="0.4")
ax2.plot(lim,lim,"k-",lw=1,alpha=0.6)
ax2.set_xlim(lim); ax2.set_ylim(lim); ax2.set_aspect("equal")
ax2.set_xlabel("Hand ECG HR (bpm)"); ax2.set_ylabel("Hand SCG HR (bpm)")
ax2.set_title(f"Gold standard: hand SCG vs hand ECG\nr={s_scg['r']:.3f}  MAE={s_scg['mae']:.2f}  bias={s_scg['bias']:+.2f}  n={s_scg['n']}")
ax2.grid(True, alpha=0.3)

# ---- Panel row 3: error-over-time for both, so drift/local failures are visible ----
ax3 = fig.add_subplot(gs[2,:])
err_det = xa_det - yb_det
err_scg = xa_scg - yb_scg
ax3.axhline(0, color="gray", lw=0.8)
ax3.plot(dn(D["ta_det"]), err_det, ".", ms=3, color="tab:green", alpha=0.4, label="best pipeline error (SCG-time axis)")
ax3.plot(dn(D["ta_scg"]), err_scg, ".", ms=3, color="0.4", alpha=0.4, label="hand-SCG error (SCG-time axis)")
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=UTC))
ax3.set_ylabel("HR error vs hand ECG (bpm)")
ax3.set_title("Error over time (detector minus true ECG rate)")
ax3.legend(loc="upper right", fontsize=9)
ax3.set_ylim(-40,40)

fig.suptitle("[Chelten] Best automated SCG pipeline vs hand annotations, full 17:26:30-18:00:00 span", fontsize=14)
out = "/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_final_full_comparison.png"
fig.savefig(out, dpi=140)
print("->", out)
