import pickle, sqlite3
import numpy as np

with open("/tmp/combined_detector.pkl","rb") as f:
    c = pickle.load(f)
with open("/tmp/singles_labels.pkl","rb") as f:
    lab = pickle.load(f)

t0 = int(lab["t0"]); t1 = int(lab["t1"])

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
ecg_good_fn, _ = good_fn(ecg_ev_ts, ecg_ev_lab, span_end)
scg_good_fn, _ = good_fn(scg_ev_ts, scg_ev_lab, span_end)

def beat_hr(peaks):
    rr = np.diff(peaks)/1e6
    hr = 60.0/rr
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

# CORRECT methodology: compute RR/HR from the FULL, un-pruned consecutive-click
# list for each channel first (preserving true adjacency), THEN apply each
# channel's own good-mask (independently) only when selecting which points are
# eligible to be compared -- exactly like the earlier-established scripts.
def align(tsA, hrA, vA_fn, tsB, hrB, vB_fn, lag_us=0, win_us=2_000_000, restrict=(t0,t1)):
    vA = vA_fn(tsA); vB = vB_fn(tsB)
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
    return np.array(xa), np.array(yb)

ecg_tmid, ecg_hr = beat_hr(ecg_pk_full)
scg_hand_tmid, scg_hand_hr = beat_hr(scg_pk_full)

print("=== sanity check: reproduce established hand-SCG vs hand-ECG benchmark ===")
for lag_s in [0, 7.0]:
    xa,yb = align(scg_hand_tmid, scg_hand_hr, scg_good_fn, ecg_tmid, smooth(ecg_tmid,ecg_hr), ecg_good_fn, lag_us=int(lag_s*1e6))
    s = stats(smooth(scg_hand_tmid, scg_hand_hr)[np.isin(scg_hand_tmid, scg_hand_tmid)][:0] if False else xa, yb)  # placeholder, fixed below
