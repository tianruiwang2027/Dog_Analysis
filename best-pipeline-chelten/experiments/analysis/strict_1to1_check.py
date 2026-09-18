import pickle, sqlite3, datetime
import numpy as np

with open("/tmp/jitter_gold_pairs.pkl","rb") as f:
    J = pickle.load(f)
pairs, t0, t1, LAG_US = J["pairs"], J["t0"], J["t1"], J["LAG_US"]
matched_scg_times = set(int(a) for a in pairs[:,0])

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

ecg_pk, ecg_ev_ts, ecg_ev_lab = load_peaks_and_good("/tmp/annotation_chelten_ecg2.db")
scg_pk, scg_ev_ts, scg_ev_lab = load_peaks_and_good("/tmp/annotation_chelten_scg2.db")
span_end = max(ecg_pk.max(), scg_pk.max(), t1)
ecg_ivs = good_intervals(ecg_ev_ts, ecg_ev_lab, span_end)
scg_ivs = good_intervals(scg_ev_ts, scg_ev_lab, span_end)
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

ecg_tmid, ecg_hr = beat_hr(ecg_pk); ecg_hr_sm = smooth(ecg_tmid, ecg_hr)
ecg_v = good_mask(ecg_tmid, ecg_ivs)

scg_tmid, scg_hr = beat_hr(scg_pk); scg_hr_sm = smooth(scg_tmid, scg_hr)
scg_v_official = good_mask(scg_tmid, scg_ivs)   # the standard 0.897 mask (good data only)

# STRICT: an RR interval only counts if BOTH its endpoint clicks are individually 1:1 matched
# to an ECG click within 300ms (i.e. no ambiguity at either end of the interval)
both_endpoints_matched = np.array([
    (int(scg_pk[i]) in matched_scg_times) and (int(scg_pk[i+1]) in matched_scg_times)
    for i in range(len(scg_pk)-1)
])
scg_v_strict = scg_v_official & both_endpoints_matched

xa0, yb0, ta0 = align(scg_tmid, scg_hr_sm, scg_v_official, ecg_tmid, ecg_hr_sm, ecg_v, lag_us=LAG_US)
s0 = stats(xa0, yb0)
xa1, yb1, ta1 = align(scg_tmid, scg_hr_sm, scg_v_strict, ecg_tmid, ecg_hr_sm, ecg_v, lag_us=LAG_US)
s1 = stats(xa1, yb1)

print("OFFICIAL (good-data only, window-count check from before):")
print("   ", s0)
print("\nSTRICT (both RR endpoints individually 1:1 matched to an ECG click, +-300ms):")
print("   ", s1)
print(f"\nr: official={s0['r']:.4f}   strict-1:1-matched={s1['r']:.4f}")
print(f"kept {s1['n']}/{s0['n']} points ({100*s1['n']/s0['n']:.1f}%) under the strict per-beat matching requirement")
