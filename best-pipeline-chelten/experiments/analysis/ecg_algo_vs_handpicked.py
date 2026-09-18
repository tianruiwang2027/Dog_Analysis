import sqlite3, os
import numpy as np
import polars as pl
from scipy.ndimage import maximum_filter1d, minimum_filter1d
import pickle
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Final check: how well does the ORIGINAL automated ECG algorithm (CORAL run
# on the Polar ECG channel -- the same channel the hand clicks were made against) agree
# with the ECG hand clicks that have been used as ground truth throughout this whole
# investigation? Same channel, same clock -- no LAG_US needed here. ----------

HERE = "/tmp/dog-test-ecg/dog-test-ecg-code/analysis"
SQI_THR = 0.10
ATTRACTOR_BAND = (95, 125)
BAND_THR = 0.50

def rolling_ptp(x, w):
    return maximum_filter1d(x, w, mode="nearest") - minimum_filter1d(x, w, mode="nearest")

def frozen_runs_mask(bpm, min_hops=8):
    same = np.concatenate([[False], bpm[1:] == bpm[:-1]])
    run_id = np.cumsum(~same)
    run_len = np.zeros(len(bpm))
    for rid in np.unique(run_id):
        m = run_id == rid
        run_len[m] = m.sum()
    return run_len >= min_hops

def load_coral_full(path, maskpath=None):
    k = pl.read_csv(path)
    ts = k["ts"].to_numpy().astype("datetime64[us]").astype("int64")
    bpm = k["bpm"].to_numpy().astype(float); sqi = k["sqi"].to_numpy().astype(float)
    algo_valid = sqi >= SQI_THR
    inband = (bpm >= ATTRACTOR_BAND[0]) & (bpm <= ATTRACTOR_BAND[1])
    algo_valid &= ~(inband & (sqi < BAND_THR))
    hopdt = np.median(np.diff(ts)) / 1e6 if len(ts) > 2 else 0.25
    w = max(5, int(round(90.0 / max(hopdt, 1e-3))) | 1)
    algo_valid &= rolling_ptp(bpm, w) > 2.0
    if maskpath and os.path.exists(maskpath):
        m = pl.read_parquet(maskpath)
        for a, b in zip(m["mask_start"].to_numpy().astype("datetime64[us]").astype("int64"),
                         m["mask_end"].to_numpy().astype("datetime64[us]").astype("int64")):
            algo_valid &= ~((ts >= a) & (ts <= b))
    algo_valid &= ~frozen_runs_mask(bpm, min_hops=8)
    return ts, bpm, sqi, algo_valid

# ---- the automated ECG algorithm (CORAL, run on Polar ECG, maskfix = leadoff-corrected) ----
ats, abpm, asqi, avalid = load_coral_full(
    f"{HERE}/coral_ecg/Chelten_ecg_polar_maskfix/out.csv",
    f"{HERE}/coral_ecg/Chelten_ecg_polar_maskfix/mask.parquet")
print(f"CORAL-ECG (Polar): {len(ats)} samples, {avalid.sum()} pass quality gating ({avalid.mean()*100:.1f}%)")

# ---- the ECG hand clicks used as ground truth throughout this whole investigation ----
def load_peaks(path):
    con = sqlite3.connect(path); cur = con.cursor()
    cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = ''")
    return np.sort(np.array([r[0] for r in cur.fetchall()], dtype="int64"))
ecg_pk = load_peaks("/tmp/annotation_chelten_ecg2.db")
print(f"ECG hand clicks: {len(ecg_pk)}  span: {(ecg_pk.max()-ecg_pk.min())/1e6/60:.1f} min")

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

hand_tmid, hand_hr, _ = beat_hr(ecg_pk)
hand_hr_sm = smooth_plain(hand_tmid, hand_hr)

# ---- align: for each valid CORAL-ECG sample, average nearby hand-click smoothed HR (±2s) ----
WIN_US = 2_000_000
a_valid_idx = np.where(avalid)[0]
xa_algo, yb_hand, ta = [], [], []
for i in a_valid_idx:
    t = ats[i]
    sel = (hand_tmid>=t-WIN_US)&(hand_tmid<=t+WIN_US)
    if sel.sum()>=1:
        xa_algo.append(abpm[i]); yb_hand.append(hand_hr_sm[sel].mean()); ta.append(t)
xa_algo = np.array(xa_algo); yb_hand = np.array(yb_hand); ta = np.array(ta, dtype="int64")
s = stats(xa_algo, yb_hand)
print(f"\n=== CORAL-ECG (algorithm) vs ECG hand clicks ===")
print(f"r={s['r']:.4f}  n={s['n']}  MAE={s['mae']:.2f}bpm  bias={s['bias']:.2f}bpm")

with open("/tmp/ecg_algo_vs_hand_result.pkl","wb") as f:
    pickle.dump(dict(xa=xa_algo, yb=yb_hand, ta=ta, s=s), f)

# ---- figure: r-scatter + HR over time, same style as everything else this session ----
order = np.argsort(ta)
ta_s, xa_s, yb_s = ta[order], xa_algo[order], yb_hand[order]
BREAK_US = 10_000_000
gap_here = np.diff(ta_s) > BREAK_US
xa_plot = xa_s.copy().astype(float); yb_plot = yb_s.copy().astype(float)
xa_plot = np.insert(xa_plot, np.where(gap_here)[0]+1, np.nan)
yb_plot = np.insert(yb_plot, np.where(gap_here)[0]+1, np.nan)
ta_ext = np.insert(ta_s.astype(float), np.where(gap_here)[0]+1, np.nan)

fig, axs = plt.subplots(1,2, figsize=(14,5.5))
ax = axs[0]
lo_v, hi_v = min(xa_s.min(),yb_s.min()), max(xa_s.max(),yb_s.max())
ax.scatter(yb_s, xa_s, s=6, alpha=0.35, color="tab:orange")
ax.plot([lo_v,hi_v],[lo_v,hi_v], color="black", lw=1, ls="--")
ax.set_xlabel("ECG hand-clicked HR (bpm)"); ax.set_ylabel("ECG algorithm HR, CORAL/Polar (bpm)")
ax.set_title(f"r={s['r']:.3f}  n={s['n']}  MAE={s['mae']:.2f}bpm  bias={s['bias']:.2f}bpm", fontsize=11)

ax2 = axs[1]
t0_all = np.nanmin(ta_ext)
t_hr = (ta_ext - t0_all)/1e6/60.0
ax2.plot(t_hr, yb_plot, color="black", lw=0.9, label="ECG hand-clicked (reference)", alpha=0.8)
ax2.plot(t_hr, xa_plot, color="tab:orange", lw=0.9, label="ECG algorithm (CORAL/Polar)", alpha=0.85)
ax2.set_xlabel("time (min)"); ax2.set_ylabel("HR (bpm)")
ax2.legend(fontsize=9)
ax2.set_title("HR over time", fontsize=11)

plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_ecg_algo_vs_handpicked.png", dpi=140)
print("saved chelten_ecg_algo_vs_handpicked.png")
