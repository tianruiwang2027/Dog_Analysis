import pickle, datetime
import numpy as np

def load_peaks_and_good(path):
    import sqlite3
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

def good_mask(ts, ivs):
    g = np.zeros(len(ts), bool)
    for a,b in ivs: g |= (ts>=a)&(ts<b)
    return g

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

def align(tsA, hrA, vA, tsB, hrB, vB, lag_us=0, win_us=2_000_000, restrict=None):
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

ecg_pk, ecg_ev_ts, ecg_ev_lab = load_peaks_and_good("/tmp/annotation_chelten_ecg2.db")
scg_pk, scg_ev_ts, scg_ev_lab = load_peaks_and_good("/tmp/annotation_chelten_scg2.db")

with open("/tmp/final_4way_compare.pkl","rb") as f:
    D = pickle.load(f)
LAG_US = D["LAG_US"]
UTC = datetime.timezone.utc

windows = {
    "FULL 17:26:30-18:00:00":      (D["t0"], D["t1"]),
    "17:30:00-18:00:00":           (int(datetime.datetime(2026,6,26,17,30,0,tzinfo=UTC).timestamp()*1e6), D["t1"]),
    "17:26:30-17:59:30":           (D["t0"], int(datetime.datetime(2026,6,26,17,59,30,tzinfo=UTC).timestamp()*1e6)),
    "17:30:00-17:59:30":           (int(datetime.datetime(2026,6,26,17,30,0,tzinfo=UTC).timestamp()*1e6), int(datetime.datetime(2026,6,26,17,59,30,tzinfo=UTC).timestamp()*1e6)),
}

span_end = max(ecg_pk.max(), scg_pk.max(), D["t1"])
ecg_ivs = good_intervals(ecg_ev_ts, ecg_ev_lab, span_end)
scg_ivs = good_intervals(scg_ev_ts, scg_ev_lab, span_end)

ecg_tmid, ecg_hr = beat_hr(ecg_pk); ecg_hr_sm = smooth(ecg_tmid, ecg_hr)
ecg_v = good_mask(ecg_tmid, ecg_ivs)
scg_tmid, scg_hr = beat_hr(scg_pk); scg_hr_sm = smooth(scg_tmid, scg_hr)
scg_v = good_mask(scg_tmid, scg_ivs)

results = {}
for name, (t0,t1) in windows.items():
    xa, yb, ta = align(scg_tmid, scg_hr_sm, scg_v, ecg_tmid, ecg_hr_sm, ecg_v, lag_us=LAG_US, restrict=(t0,t1))
    s = stats(xa, yb)
    print(f"{name:28s}  n={s['n']:5d}  r={s['r']:.4f}  MAE={s['mae']:.2f}  bias={s['bias']:+.2f}")
    results[name] = dict(xa=xa,yb=yb,ta=ta,s=s,t0=t0,t1=t1)

with open("/tmp/compare_172630_175930.pkl","wb") as f:
    pickle.dump(results, f)
