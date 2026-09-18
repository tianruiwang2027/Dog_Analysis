import pickle, datetime
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

with open("/tmp/final_4way_compare.pkl","rb") as f:
    D = pickle.load(f)

UTC = datetime.timezone.utc
def dn(us):
    return mdates.date2num([datetime.datetime.fromtimestamp(t/1e6, tz=UTC) for t in np.atleast_1d(us)])

t0, t1, LAG_US = D["t0"], D["t1"], D["LAG_US"]

fig = plt.figure(figsize=(17,13))
gs = fig.add_gridspec(3, 3, height_ratios=[2.0,1,1], hspace=0.38, wspace=0.28)

# ---- Panel 1: HR over time, full window, all 4 series ----
ax0 = fig.add_subplot(gs[0,:])
ecg_v, ecg_tmid, ecg_hr_sm = D["ecg_v"], D["ecg_tmid"], D["ecg_hr_sm"]
m_ecg = ecg_v & (ecg_tmid-LAG_US>=t0-5_000_000) & (ecg_tmid-LAG_US<=t1+5_000_000)
ax0.plot(dn(ecg_tmid[m_ecg]-LAG_US), ecg_hr_sm[m_ecg], color="black", lw=1.0, label="hand-clicked ECG (reference)", zorder=5)

scg_v, scg_tmid, scg_hr_sm = D["scg_v"], D["scg_tmid"], D["scg_hr_sm"]
m_scg = scg_v & (scg_tmid>=t0) & (scg_tmid<=t1)
ax0.plot(dn(scg_tmid[m_scg]), scg_hr_sm[m_scg], color="0.55", lw=0.8, ls="--", label="hand-clicked SCG (gold standard)")

det_v, det_tmid, det_hr_sm = D["det_v"], D["det_tmid"], D["det_hr_sm"]
m_det = det_v & (det_tmid>=t0) & (det_tmid<=t1)
ax0.plot(dn(det_tmid[m_det]), det_hr_sm[m_det], color="tab:green", lw=1.0, alpha=0.9, label="best pipeline (envelope + two-lobe template, NCC≥0.40)")

c_v, c_ts, c_bpm = D["c_v"], D["c_ts"], D["c_bpm"]
m_cor = c_v & (c_ts>=t0) & (c_ts<=t1)
ax0.plot(dn(c_ts[m_cor]), c_bpm[m_cor], color="tab:orange", lw=0.9, alpha=0.85, label="CORAL (coral-st, SQI≥0.10)")

ax0.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=UTC))
ax0.set_ylabel("HR (bpm)")
ax0.set_title("[Chelten] HR over time -- CORAL vs. best pipeline vs. hand-clicked, 17:26:30-18:00:00 (good-data only)")
ax0.legend(loc="upper right", fontsize=9)
ax0.set_ylim(30,170)

scg_ivs_sorted = sorted(D["scg_ivs"])
prev_end = t0
for a,b in scg_ivs_sorted:
    if a > prev_end:
        ax0.axvspan(dn(prev_end)[0], dn(min(a,t1))[0], color="red", alpha=0.06)
    prev_end = max(prev_end, b)
if prev_end < t1:
    ax0.axvspan(dn(prev_end)[0], dn(t1)[0], color="red", alpha=0.06)

lim = [30,170]
specs = [("det","tab:green","Best pipeline"), ("scg","0.4","Hand SCG (gold standard)"), ("cor","tab:orange","CORAL")]
for j,(key,color,label) in enumerate(specs):
    ax = fig.add_subplot(gs[1,j])
    xa, yb, s = D[f"xa_{key}"], D[f"yb_{key}"], D[f"s_{key}"]
    ax.scatter(yb, xa, s=6, alpha=0.3, color=color)
    ax.plot(lim,lim,"k-",lw=1,alpha=0.6)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("Hand ECG HR (bpm)"); ax.set_ylabel(f"{label} HR (bpm)")
    ax.set_title(f"{label} vs hand ECG\nr={s['r']:.3f}  MAE={s['mae']:.2f}  bias={s['bias']:+.2f}  n={s['n']}")
    ax.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[2,:])
for key,color,label in specs:
    xa, yb, ta = D[f"xa_{key}"], D[f"yb_{key}"], D[f"ta_{key}"]
    err = xa-yb
    ax3.plot(dn(ta), err, ".", ms=3, color=color, alpha=0.4, label=f"{label} error")
ax3.axhline(0, color="gray", lw=0.8)
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=UTC))
ax3.set_ylabel("HR error vs hand ECG (bpm)")
ax3.set_title("Error over time")
ax3.legend(loc="upper right", fontsize=9)
ax3.set_ylim(-60,60)

fig.suptitle("[Chelten] CORAL vs. best custom pipeline vs. hand annotations, full 17:26:30-18:00:00 span", fontsize=14)
out = "/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_4way_compare.png"
fig.savefig(out, dpi=135)
print("->", out)
