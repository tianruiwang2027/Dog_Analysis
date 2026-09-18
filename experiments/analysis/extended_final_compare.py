import glob, pickle, datetime
import numpy as np
import polars as pl
from scipy import signal as sg
from scipy.ndimage import median_filter, uniform_filter1d
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Final step: gap-aware SCG HR (whole overlap: before+during+after, using the
# frozen/unchanged CNN+S2-chain pipeline, before-segment and after-segment envelopes
# rescaled against the SAME reference amplitude the models were calibrated on) vs the
# fully-tuned Pan-Tompkins ECG detector run on the full session. Compared directly against
# each other (algorithm vs algorithm) since ECG hand clicks don't exist outside 17:22-18:00,
# with the previously hand-click-validated window called out separately. ----------

HERE = "/tmp/dog-test-ecg/dog-test-ecg-code"
UTC = datetime.timezone.utc

with open("/tmp/extended_full_merged_FIXED.pkl","rb") as f:
    M = pickle.load(f)
full_merged = M["full_merged"]

with open("/tmp/cnn_mask_apply.pkl","rb") as f:
    CA = pickle.load(f)
LAG_US = CA["LAG_US"]

RESTRICT_PREV = (1782494790000000, 1782496808699088)

def beat_hr(peaks):
    rr = np.diff(peaks)/1e6; hr = 60.0/rr
    tmid = peaks[:-1] + np.diff(peaks)//2
    return tmid, hr, rr
def smooth_plain(tmid, hr, win_s=3.0):
    win_us2 = int(win_s*1e6); out = np.empty(len(tmid))
    for i,t in enumerate(tmid):
        sel = (tmid>=t-win_us2)&(tmid<=t+win_us2); out[i]=hr[sel].mean()
    return out
def stats(a,b):
    dd=a-b
    return dict(n=int(len(a)), bias=float(dd.mean()), mae=float(np.abs(dd).mean()),
                r=float(np.corrcoef(a,b)[0,1]) if len(a)>2 else float("nan"))
def align(tsA, hrA, tsB, hrB, lag_us=0, win_us=2_000_000):
    xa, yb, ta = [], [], []
    for i in range(len(tsA)):
        t = tsA[i] + lag_us
        sel = (tsB>=t-win_us)&(tsB<=t+win_us)
        if sel.sum()>=1:
            xa.append(hrA[i]); yb.append(hrB[sel].mean()); ta.append(tsA[i])
    return np.array(xa), np.array(yb), np.array(ta, dtype="int64")

# ---- SCG side: RR-sanity + gap-aware smoothing over the WHOLE merged (before+during+after) ----
tmid, hr_raw, rr = beat_hr(full_merged)
MAX_RR_S = 1.5
ok = rr <= MAX_RR_S
tmid_ok, hr_ok = tmid[ok], hr_raw[ok]

GAP_THRESH_S = 1.5; WIN_S = 3.0
win_us = int(WIN_S*1e6); gap_us = int(GAP_THRESH_S*1e6)
all_gaps_len = np.diff(full_merged)
has_gap = all_gaps_len >= gap_us
gap_windows = list(zip(full_merged[:-1][has_gap], full_merged[1:][has_gap]))
print(f"n detection gaps (>= {GAP_THRESH_S}s) over full session: {len(gap_windows)}")
def window_touches_a_gap(t):
    lo, hi = t-win_us, t+win_us
    for gs, ge in gap_windows:
        if ge >= lo and gs <= hi: return True
    return False
hr_sm = np.full(len(tmid_ok), np.nan)
n_suppressed = 0
for i, t in enumerate(tmid_ok):
    if window_touches_a_gap(t):
        n_suppressed += 1; continue
    sel = (tmid_ok>=t-win_us)&(tmid_ok<=t+win_us)
    hr_sm[i] = hr_ok[sel].mean()
keep = ~np.isnan(hr_sm)
scg_tmid_final, scg_hr_final = tmid_ok[keep], hr_sm[keep]
print(f"SCG: {len(tmid_ok)} raw beat-HR points, {n_suppressed} suppressed by gap-aware, {len(scg_tmid_final)} final")

# ---- ECG side: fully-tuned Pan-Tompkins, full session ----
FC_LO, FC_HI = 8.0, 45.0
REFRACT_S = 0.40
MWI_S = 0.010
FLOOR_S = 2.0
SNR_THR = 100.0
def detect_qrs(x, fs):
    b = sg.butter(2, [FC_LO/(fs/2), FC_HI/(fs/2)], btype="band")
    xf = sg.filtfilt(*b, x - np.mean(x))
    deriv = np.convolve(xf, np.array([1,2,0,-2,-1])*(fs/8.0), mode="same")
    sqd = deriv**2
    mwi = uniform_filter1d(sqd, max(1, int(MWI_S*fs)))
    floor = median_filter(mwi, max(3,int(FLOOR_S*fs))|1, mode="nearest") + 1e-12
    pk, _ = sg.find_peaks(mwi, distance=int(fs*REFRACT_S), height=None)
    snr = mwi[pk] / floor[pk]
    pk = pk[snr > SNR_THR]
    win = int(0.06*fs)
    R = []
    for p_ in pk:
        a, bnd = max(0, p_-win), min(len(xf), p_+win)
        R.append(a + int(np.argmax(np.abs(xf[a:bnd]))))
    return np.unique(R)

p = glob.glob(f"{HERE}/hive/username=ecg_polar/device=Chelten/stream=0/date=*/data_0.parquet")[0]
df = pl.read_parquet(p, columns=["ts", "c1"])
ts_raw = df["ts"].to_numpy().astype("int64")
x_raw = df["c1"].to_numpy().astype(float)
fs = float(1e6 / np.median(np.diff(ts_raw)))
R_idx = detect_qrs(x_raw, fs)
pt_pk = ts_raw[R_idx]
print(f"Pan-Tompkins: {len(pt_pk)} R-peaks, full session")

pt_tmid, pt_hr_raw, pt_rr = beat_hr(pt_pk)
rr_ok = (pt_rr >= 0.3) & (pt_rr <= 1.5)
pt_tmid_ok, pt_hr_ok = pt_tmid[rr_ok], pt_hr_raw[rr_ok]
pt_hr_sm = smooth_plain(pt_tmid_ok, pt_hr_ok)

# ---- align SCG (shifted by LAG_US into ECG clock) vs Pan-Tompkins ECG ----
xa, yb, ta = align(scg_tmid_final, scg_hr_final, pt_tmid_ok, pt_hr_sm, lag_us=LAG_US)
s_all = stats(xa, yb)
print(f"\n=== SCG pipeline vs Pan-Tompkins ECG, FULL overlap session ===")
print(f"r={s_all['r']:.4f}  n={s_all['n']}  MAE={s_all['mae']:.2f}bpm  bias={s_all['bias']:.2f}bpm")

# split by region: previously-validated (has hand click) vs new/remaining
in_during = (ta>=RESTRICT_PREV[0])&(ta<=RESTRICT_PREV[1])
in_before = ta < RESTRICT_PREV[0]
in_after  = ta > RESTRICT_PREV[1]
s_during = stats(xa[in_during], yb[in_during])
s_before = stats(xa[in_before], yb[in_before])
s_after  = stats(xa[in_after], yb[in_after])
print(f"\n  BEFORE (17:04-17:26, no hand-click ground truth): r={s_before['r']:.4f}  n={s_before['n']}  MAE={s_before['mae']:.2f}  bias={s_before['bias']:.2f}")
print(f"  DURING (17:26-18:00, previously hand-click validated): r={s_during['r']:.4f}  n={s_during['n']}  MAE={s_during['mae']:.2f}  bias={s_during['bias']:.2f}")
print(f"  AFTER  (18:00-18:29, no hand-click ground truth):  r={s_after['r']:.4f}  n={s_after['n']}  MAE={s_after['mae']:.2f}  bias={s_after['bias']:.2f}")

with open("/tmp/extended_final_compare_result.pkl","wb") as f:
    pickle.dump(dict(xa=xa, yb=yb, ta=ta, s_all=s_all, s_before=s_before, s_during=s_during, s_after=s_after,
                      scg_tmid_final=scg_tmid_final, scg_hr_final=scg_hr_final,
                      pt_tmid_ok=pt_tmid_ok, pt_hr_sm=pt_hr_sm, RESTRICT_PREV=RESTRICT_PREV, LAG_US=LAG_US), f)
print("saved /tmp/extended_final_compare_result.pkl")

# ---- figure ----
region_color = np.where(in_before, "#d62728", np.where(in_during, "#2ca02c", "#1f77b4"))
fig, axs = plt.subplots(1,2, figsize=(15,5.8))
ax = axs[0]
for lab, mask, c in [("before (no GT)", in_before, "#d62728"), ("during (hand-click validated)", in_during, "#2ca02c"), ("after (no GT)", in_after, "#1f77b4")]:
    ax.scatter(yb[mask], xa[mask], s=8, alpha=0.4, color=c, label=f"{lab}  n={mask.sum()}")
lo_v, hi_v = min(xa.min(),yb.min()), max(xa.max(),yb.max())
ax.plot([lo_v,hi_v],[lo_v,hi_v], color="black", lw=1, ls="--")
ax.set_xlabel("Pan-Tompkins ECG HR (bpm)"); ax.set_ylabel("SCG pipeline HR (bpm)")
ax.set_title(f"Full session: r={s_all['r']:.3f}  n={s_all['n']}  MAE={s_all['mae']:.2f}bpm  bias={s_all['bias']:+.2f}bpm", fontsize=10)
ax.legend(fontsize=8, loc="upper left")

ax2 = axs[1]
t0_plot = ts_raw.min()
def to_min(t): return (t.astype(float)-t0_plot)/1e6/60.0
# SCG (shifted into ECG frame), broken at internal gaps for clean plotting
scg_t_shift = scg_tmid_final + LAG_US
order = np.argsort(scg_t_shift)
st, sh = scg_t_shift[order], scg_hr_final[order]
gap_here = np.diff(st) > 10_000_000
st_plot = st.astype(float).copy(); sh_plot = sh.astype(float).copy()
st_plot = np.insert(st_plot, np.where(gap_here)[0]+1, np.nan)
sh_plot = np.insert(sh_plot, np.where(gap_here)[0]+1, np.nan)
ax2.plot(to_min(st_plot), sh_plot, color="tab:purple", lw=0.8, label="SCG pipeline (shifted to ECG clock)", alpha=0.85)
ax2.plot(to_min(pt_tmid_ok.astype(float)), pt_hr_sm, color="tab:blue", lw=0.7, label="Pan-Tompkins ECG", alpha=0.6)
ax2.axvspan(to_min(np.array([RESTRICT_PREV[0]]).astype(float))[0], to_min(np.array([RESTRICT_PREV[1]]).astype(float))[0],
            color="green", alpha=0.08, label="hand-click validated window")
ax2.set_xlabel("time (min, from ecg_polar recording start)"); ax2.set_ylabel("HR (bpm)")
ax2.legend(fontsize=8)
ax2.set_title("HR over time -- full overlap session", fontsize=10)

plt.tight_layout()
plt.savefig(f"{HERE}/figs/chelten_extended_full_session_compare.png", dpi=140)
print("saved chelten_extended_full_session_compare.png")
