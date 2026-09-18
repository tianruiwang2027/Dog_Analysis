import pickle, sqlite3, datetime
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

with open("/tmp/combined_detector.pkl","rb") as f:
    c = pickle.load(f)
with open("/tmp/singles_labels.pkl","rb") as f:
    lab = pickle.load(f)

t0 = int(lab["t0"]); t1 = int(lab["t1"])
LAG_US = int(7.0e6)
UTC = datetime.timezone.utc

def load_peaks_and_good(path):
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = '' ORDER BY timestamp")
    pk = np.array([r[0] for r in cur.fetchall()], dtype="int64")
    cur.execute("SELECT DISTINCT timestamp, label FROM events WHERE label != '' ORDER BY timestamp")
    rows = cur.fetchall()
    ev_ts = np.array([r[0] for r in rows], dtype="int64")
    ev_lab = [r[1].lower() for r in rows]
    return pk, ev_ts, ev_lab

def good_intervals(ev_ts, ev_lab, span_end):
    ivs=[]; state="bad"; cur_start=None
    for t,l in zip(ev_ts, ev_lab):
        if "good" in l and "start" in l:
            if state!="good": cur_start=t; state="good"
        elif "bad" in l and "start" in l:
            if state=="good": ivs.append((cur_start,t)); state="bad"
        elif "bad" in l and "end" in l:
            if state!="good": cur_start=t; state="good"
        elif "good" in l and "end" in l:
            if state=="good": ivs.append((cur_start,t)); state="bad"
        elif "stop" in l:
            if state=="good": ivs.append((cur_start,t)); state="bad"
    if state=="good": ivs.append((cur_start, span_end))
    return ivs

def good_fn(ev_ts, ev_lab, span_end):
    ivs = good_intervals(ev_ts, ev_lab, span_end)
    def good(ts):
        g = np.zeros(len(ts), bool)
        for a,b in ivs: g |= (ts>=a)&(ts<b)
        return g
    return good, ivs

ecg_pk_full, ecg_ev_ts, ecg_ev_lab = load_peaks_and_good("/tmp/annotation_chelten_ecg2.db")
scg_pk_full, scg_ev_ts, scg_ev_lab = load_peaks_and_good("/tmp/annotation_chelten_scg2.db")
span_end = max(ecg_pk_full.max(), scg_pk_full.max(), t1)
ecg_good, ecg_ivs = good_fn(ecg_ev_ts, ecg_ev_lab, span_end)
scg_good, scg_ivs = good_fn(scg_ev_ts, scg_ev_lab, span_end)

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
                within10=float(np.mean(np.abs(dd)<=10)),
                r=float(np.corrcoef(a,b)[0,1]) if len(a)>2 else float("nan"))

def align(tsA, hrA, vA, tsB, hrB, vB, lag_us=0, win_us=2_000_000, restrict=(t0,t1)):
    bts, bhr = tsB[vB], hrB[vB]
    A = np.where(vA)[0]
    xa, yb, ta = [], [], []
    for i in A:
        traw = tsA[i]
        if restrict is not None and not (restrict[0] <= traw <= restrict[1]):
            continue
        t = traw + lag_us
        sel = (bts>=t-win_us)&(bts<=t+win_us)
        if sel.sum()>=1:
            xa.append(hrA[i]); yb.append(bhr[sel].mean()); ta.append(traw)
    return np.array(xa), np.array(yb), np.array(ta, dtype="int64")

ecg_tmid, ecg_hr_raw = beat_hr(ecg_pk_full)
ecg_hr_sm = smooth(ecg_tmid, ecg_hr_raw)
ecg_v = ecg_good(ecg_tmid)

scg_tmid, scg_hr_raw = beat_hr(scg_pk_full)
scg_hr_sm = smooth(scg_tmid, scg_hr_raw)
scg_v = scg_good(scg_tmid)

det = np.sort(c["primary"])   # best pipeline: envelope singles (thr0.3, refract0.5) + wide two-lobe template filter (NCC>=0.5)
det_tmid, det_hr_raw = beat_hr(det)
det_hr_sm = smooth(det_tmid, det_hr_raw)
det_v = scg_good(det_tmid)

# --- comparisons (lag-corrected, good-masked, restricted to [t0,t1]) ---
xa_det, yb_det, ta_det = align(det_tmid, det_hr_sm, det_v, ecg_tmid, ecg_hr_sm, ecg_v, lag_us=LAG_US)
s_det = stats(xa_det, yb_det)
xa_scg, yb_scg, ta_scg = align(scg_tmid, scg_hr_sm, scg_v, ecg_tmid, ecg_hr_sm, ecg_v, lag_us=LAG_US)
s_scg = stats(xa_scg, yb_scg)

print("=== BEST PIPELINE (envelope singles + two-lobe template filter) vs hand ECG ===")
print(s_det)
print("\n=== hand-clicked SCG vs hand ECG (gold standard) ===")
print(s_scg)

with open("/tmp/final_full_comparison_data.pkl","wb") as f:
    pickle.dump(dict(det_tmid=det_tmid, det_hr_sm=det_hr_sm, det_v=det_v,
                      ecg_tmid=ecg_tmid, ecg_hr_sm=ecg_hr_sm, ecg_v=ecg_v,
                      scg_tmid=scg_tmid, scg_hr_sm=scg_hr_sm, scg_v=scg_v,
                      xa_det=xa_det, yb_det=yb_det, ta_det=ta_det, s_det=s_det,
                      xa_scg=xa_scg, yb_scg=yb_scg, ta_scg=ta_scg, s_scg=s_scg,
                      ecg_ivs=ecg_ivs, scg_ivs=scg_ivs, t0=t0, t1=t1, LAG_US=LAG_US), f)
print("\nsaved data")
