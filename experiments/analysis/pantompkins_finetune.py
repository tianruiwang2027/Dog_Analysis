import glob, sqlite3, time, itertools
import numpy as np
import polars as pl
from scipy import signal as sg
from scipy.ndimage import median_filter, uniform_filter1d
import pickle

# ---------- Broader hyperparameter search for the Pan-Tompkins-style detector, to find
# the max achievable r against the ECG hand clicks. Same detector family/logic as before
# (bandpass -> derivative -> square -> moving-window-integrate -> adaptive floor+SNR
# threshold), now sweeping the bandpass corners, refractory period, MWI window and SNR
# threshold together. Same gap-aware hand-click reference and eval methodology as the
# adopted run, just vectorized (searchsorted) so the grid is fast. ----------

HERE = "/tmp/dog-test-ecg/dog-test-ecg-code"

p = glob.glob(f"{HERE}/hive/username=ecg_polar/device=Chelten/stream=0/date=*/data_0.parquet")[0]
df = pl.read_parquet(p, columns=["ts", "c1"])
ts_raw = df["ts"].to_numpy().astype("int64")
x_raw = df["c1"].to_numpy().astype(float)
fs = float(1e6 / np.median(np.diff(ts_raw)))
print(f"loaded {len(ts_raw)} samples @ fs={fs:.2f}Hz")

# ---- ECG hand clicks + gap-aware reference (identical to the adopted script) ----
def load_peaks_and_good(path):
    con = sqlite3.connect(path); cur = con.cursor()
    cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = ''")
    pk = np.array([r[0] for r in cur.fetchall()], dtype="int64")
    cur.execute("SELECT DISTINCT timestamp, label FROM events WHERE label != ''")
    rows = cur.fetchall()
    ev_ts = np.array([r[0] for r in rows], dtype="int64"); ev_lab=[r[1].lower() for r in rows]
    return pk, ev_ts, ev_lab
def good_intervals(ev_ts, ev_lab, span_end):
    ivs=[]; state="bad"; cur_start=None
    for tt,l in zip(ev_ts, ev_lab):
        if "good" in l and "start" in l:
            if state!="good": cur_start=tt; state="good"
        elif "bad" in l and "start" in l:
            if state=="good": ivs.append((cur_start,tt)); state="bad"
        elif "bad" in l and "end" in l:
            if state!="good": cur_start=tt; state="good"
        elif "good" in l and "end" in l:
            if state=="good": ivs.append((cur_start,tt)); state="bad"
        elif "stop" in l:
            if state=="good": ivs.append((cur_start,tt)); state="bad"
    if state=="good": ivs.append((cur_start, span_end))
    return ivs
def good_mask(ts, ivs):
    g = np.zeros(len(ts), bool)
    for a,b in ivs: g |= (ts>=a)&(ts<b)
    return g
def beat_hr(peaks):
    rr = np.diff(peaks)/1e6; hr = 60.0/rr
    tmid = peaks[:-1] + np.diff(peaks)//2
    return tmid, hr, rr

ecg_pk, ecg_ev_ts, ecg_ev_lab = load_peaks_and_good("/tmp/annotation_chelten_ecg2.db")
span_end = ecg_pk.max()
ecg_ivs = good_intervals(ecg_ev_ts, ecg_ev_lab, span_end)
ecg_tmid, ecg_hr_raw, _ = beat_hr(ecg_pk)

GAP_THRESH_S = 1.5; WIN_S = 3.0
win_us = int(WIN_S*1e6); gap_us = int(GAP_THRESH_S*1e6)
gaps_len = np.diff(ecg_pk); has_gap = gaps_len >= gap_us
gap_windows = list(zip(ecg_pk[:-1][has_gap], ecg_pk[1:][has_gap]))
def window_touches_a_gap(t):
    lo, hi = t-win_us, t+win_us
    for gs, ge in gap_windows:
        if ge >= lo and gs <= hi: return True
    return False
ecg_hr_sm = np.full(len(ecg_tmid), np.nan)
for i, t in enumerate(ecg_tmid):
    if window_touches_a_gap(t): continue
    sel = (ecg_tmid>=t-win_us)&(ecg_tmid<=t+win_us)
    ecg_hr_sm[i] = ecg_hr_raw[sel].mean()
keep_ref = ~np.isnan(ecg_hr_sm)
ecg_tmid_k, ecg_hr_sm_k = ecg_tmid[keep_ref], ecg_hr_sm[keep_ref]
ecg_v_full = good_mask(ecg_tmid_k, ecg_ivs)
# pre-sorted (tmid always increasing) -- keep sorted arrays for searchsorted
order_ref = np.argsort(ecg_tmid_k)
ref_t = ecg_tmid_k[order_ref]; ref_hr = ecg_hr_sm_k[order_ref]; ref_v = ecg_v_full[order_ref]
ref_t_valid = ref_t[ref_v]; ref_hr_valid = ref_hr[ref_v]

def stats(a,b):
    dd=a-b
    return dict(n=int(len(a)), bias=float(dd.mean()), mae=float(np.abs(dd).mean()),
                r=float(np.corrcoef(a,b)[0,1]) if len(a)>2 else float("nan"))

def smooth_plain_fast(tmid, hr, win_s=3.0):
    # vectorized via searchsorted + cumulative sums (tmid assumed sorted ascending)
    win_us2 = int(win_s*1e6)
    lo_idx = np.searchsorted(tmid, tmid-win_us2, side="left")
    hi_idx = np.searchsorted(tmid, tmid+win_us2, side="right")
    csum = np.concatenate([[0.0], np.cumsum(hr)])
    ssum = csum[hi_idx]-csum[lo_idx]
    cnt = (hi_idx-lo_idx).astype(float)
    return ssum/cnt

def align_fast(tsA, hrA, refT, refHR, win_us=2_000_000):
    # refT assumed sorted ascending; average refHR within +-win_us of each tsA
    lo_idx = np.searchsorted(refT, tsA-win_us, side="left")
    hi_idx = np.searchsorted(refT, tsA+win_us, side="right")
    has = hi_idx > lo_idx
    csum = np.concatenate([[0.0], np.cumsum(refHR)])
    ssum = (csum[hi_idx]-csum[lo_idx])[has]
    cnt = (hi_idx-lo_idx)[has].astype(float)
    return hrA[has], ssum/cnt

def evaluate(FC_LO, FC_HI, REFRACT_S, MWI_S, FLOOR_S, SNR_THR):
    b = sg.butter(2, [FC_LO/(fs/2), FC_HI/(fs/2)], btype="band")
    xf = sg.filtfilt(*b, x_raw - np.mean(x_raw))
    deriv = np.convolve(xf, np.array([1,2,0,-2,-1])*(fs/8.0), mode="same")
    sqd = deriv**2
    mwi = uniform_filter1d(sqd, max(1, int(MWI_S*fs)))
    floor = median_filter(mwi, max(3,int(FLOOR_S*fs))|1, mode="nearest") + 1e-12
    pk, _ = sg.find_peaks(mwi, distance=max(1,int(fs*REFRACT_S)), height=None)
    snr = mwi[pk] / floor[pk]
    pk = pk[snr > SNR_THR]
    win = int(0.06*fs)
    R = []
    for p_ in pk:
        a, bnd = max(0, p_-win), min(len(xf), p_+win)
        R.append(a + int(np.argmax(np.abs(xf[a:bnd]))))
    R = np.unique(np.array(R, dtype=int))
    pt_pk = ts_raw[R]
    if len(pt_pk) < 10:
        return None
    pt_tmid, pt_hr_raw, pt_rr = beat_hr(pt_pk)
    rr_ok = (pt_rr >= 0.3) & (pt_rr <= 1.5)
    pt_tmid_ok, pt_hr_ok = pt_tmid[rr_ok], pt_hr_raw[rr_ok]
    if len(pt_tmid_ok) < 10:
        return None
    pt_hr_sm = smooth_plain_fast(pt_tmid_ok, pt_hr_ok)
    xa, yb = align_fast(pt_tmid_ok, pt_hr_sm, ref_t_valid, ref_hr_valid)
    if len(xa) < 10:
        return None
    return stats(xa, yb), len(pt_pk)

# ---- coarse grid over bandpass corners, refractory, MWI window, SNR threshold ----
t0 = time.time()
results = []
FC_LOS = [5, 8, 10, 12, 15]
FC_HIS = [20, 25, 30, 35, 40]
REFRACTS = [0.22, 0.26, 0.30, 0.34]
MWIS = [0.03, 0.05, 0.08]
SNRS = [30, 40, 50, 60, 70, 80, 100]

n_done = 0
for FC_LO, FC_HI in itertools.product(FC_LOS, FC_HIS):
    if FC_HI - FC_LO < 10: continue
    for REFRACT_S in REFRACTS:
        for MWI_S in MWIS:
            # compute the expensive filter/mwi/floor ONCE per (FC_LO,FC_HI,REFRACT,MWI) combo
            # by looping SNR cheaply inside evaluate() would recompute filtering each time --
            # accept the recompute here for simplicity/robustness given runtime budget below
            for SNR_THR in SNRS:
                out = evaluate(FC_LO, FC_HI, REFRACT_S, MWI_S, 2.0, SNR_THR)
                n_done += 1
                if out is None: continue
                s, npk = out
                results.append(dict(FC_LO=FC_LO, FC_HI=FC_HI, REFRACT_S=REFRACT_S, MWI_S=MWI_S,
                                     SNR_THR=SNR_THR, n_peaks=npk, **s))
    print(f"  ...FC=({FC_LO},{FC_HI}) done, {n_done} combos so far, {time.time()-t0:.1f}s elapsed")

print(f"\ntotal combos evaluated: {n_done}  ({time.time()-t0:.1f}s)")
results.sort(key=lambda d: -d["r"])
print("\nTOP 15 by r:")
for d in results[:15]:
    print(f"FC=({d['FC_LO']:2d},{d['FC_HI']:2d}) REFRACT={d['REFRACT_S']:.2f} MWI={d['MWI_S']:.2f} SNR={d['SNR_THR']:5.0f}  "
          f"r={d['r']:.4f} n={d['n']:5d} MAE={d['mae']:.2f} bias={d['bias']:+.2f} n_peaks={d['n_peaks']}")

with open("/tmp/pantompkins_finetune_grid.pkl","wb") as f:
    pickle.dump(results, f)
print("saved /tmp/pantompkins_finetune_grid.pkl")
