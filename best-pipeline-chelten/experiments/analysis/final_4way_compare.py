import pickle, datetime
import numpy as np
import polars as pl

with open("/tmp/ncc_cutoff_sweep.pkl","rb") as f:
    SWEEP = pickle.load(f)
with open("/tmp/final_full_comparison_data.pkl","rb") as f:
    D = pickle.load(f)

t0, t1, LAG_US = D["t0"], D["t1"], D["LAG_US"]
scg_ivs = sorted(D["scg_ivs"])
ecg_tmid, ecg_hr_sm, ecg_v = D["ecg_tmid"], D["ecg_hr_sm"], D["ecg_v"]

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

def stats(a,b):
    dd=a-b
    return dict(n=int(len(a)), bias=float(dd.mean()), mae=float(np.abs(dd).mean()),
                r=float(np.corrcoef(a,b)[0,1]) if len(a)>2 else float("nan"))

def align(tsA, hrA, vA, tsB, hrB, vB, lag_us=0, win_us=2_000_000, restrict=(t0,t1)):
    bts, bhr = tsB[vB], hrB[vB]
    A = np.where(vA)[0]
    xa, yb, ta = [], [], []
    for i in A:
        traw = tsA[i]
        if restrict is not None and not (restrict[0] <= traw <= restrict[1]): continue
        t = traw + lag_us
        sel = (bts>=t-win_us)&(bts<=t+win_us)
        if sel.sum()>=1:
            xa.append(hrA[i]); yb.append(bhr[sel].mean()); ta.append(traw)
    return np.array(xa), np.array(yb), np.array(ta, dtype="int64")

def strict_good_mask_for_pairs(peaks, ivs):
    a = peaks[:-1]; b = peaks[1:]
    ok = np.zeros(len(a), bool)
    for ga, gb in ivs: ok |= (a>=ga) & (b<=gb)
    return ok

def scg_good_point(ts):
    g = np.zeros(len(ts), bool)
    for a,b in scg_ivs: g |= (ts>=a)&(ts<b)
    return g

# ---------- 1) BEST PIPELINE (locked-in cutoff = 0.40) ----------
primary = SWEEP[0.40]["primary"]
det_tmid, det_hr_raw = beat_hr(primary)
det_hr_sm = smooth(det_tmid, det_hr_raw)
det_v = strict_good_mask_for_pairs(primary, scg_ivs)
xa_det, yb_det, ta_det = align(det_tmid, det_hr_sm, det_v, ecg_tmid, ecg_hr_sm, ecg_v, lag_us=LAG_US)
s_det = stats(xa_det, yb_det)

# ---------- 2) HAND-CLICKED SCG (gold standard) ----------
scg_tmid, scg_hr_sm, scg_v = D["scg_tmid"], D["scg_hr_sm"], D["scg_v"]
xa_scg, yb_scg, ta_scg = align(scg_tmid, scg_hr_sm, scg_v, ecg_tmid, ecg_hr_sm, ecg_v, lag_us=LAG_US)
s_scg = stats(xa_scg, yb_scg)

# ---------- 3) CORAL ----------
cdf = pl.read_csv("coral_scg/Chelten/out.csv")
c_ts_str = cdf["ts"].to_list()
c_bpm = cdf["bpm"].to_numpy()
c_sqi = cdf["sqi"].to_numpy()
UTC = datetime.timezone.utc
c_ts = np.array([datetime.datetime.fromisoformat(s).replace(tzinfo=UTC).timestamp()*1e6 for s in c_ts_str], dtype="int64")

SQI_MIN = 0.10   # CORAL's own convention (coral_scg.py: ok = sqi >= 0.10)
c_valid_base = (c_ts>=t0)&(c_ts<=t1)&(c_sqi>=SQI_MIN)
c_v = c_valid_base & scg_good_point(c_ts)
print("n CORAL hops in-window:", c_valid_base.sum(), " after good-mask:", c_v.sum(), "/ total", len(c_ts))

xa_cor, yb_cor, ta_cor = align(c_ts, c_bpm, c_v, ecg_tmid, ecg_hr_sm, ecg_v, lag_us=LAG_US)
s_cor = stats(xa_cor, yb_cor)

print("\n=== BEST PIPELINE (envelope + two-lobe template, NCC>=0.40) vs hand ECG ===")
print(s_det)
print("\n=== hand-clicked SCG vs hand ECG (gold standard) ===")
print(s_scg)
print("\n=== CORAL (coral-st, SQI>=0.10) vs hand ECG ===")
print(s_cor)

with open("/tmp/final_4way_compare.pkl","wb") as f:
    pickle.dump(dict(
        det_tmid=det_tmid, det_hr_sm=det_hr_sm, det_v=det_v,
        xa_det=xa_det, yb_det=yb_det, ta_det=ta_det, s_det=s_det,
        scg_tmid=scg_tmid, scg_hr_sm=scg_hr_sm, scg_v=scg_v,
        xa_scg=xa_scg, yb_scg=yb_scg, ta_scg=ta_scg, s_scg=s_scg,
        c_ts=c_ts, c_bpm=c_bpm, c_sqi=c_sqi, c_v=c_v,
        xa_cor=xa_cor, yb_cor=yb_cor, ta_cor=ta_cor, s_cor=s_cor,
        ecg_tmid=ecg_tmid, ecg_hr_sm=ecg_hr_sm, ecg_v=ecg_v,
        scg_ivs=D["scg_ivs"], t0=t0, t1=t1, LAG_US=LAG_US), f)
print("\nsaved")
