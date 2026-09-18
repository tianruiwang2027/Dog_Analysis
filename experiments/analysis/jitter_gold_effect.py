import pickle, sqlite3, datetime
import numpy as np

with open("/tmp/jitter_gold_pairs.pkl","rb") as f:
    J = pickle.load(f)
pairs, t0, t1, LAG_US = J["pairs"], J["t0"], J["t1"], J["LAG_US"]

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

# full ECG reference series (unrestricted list, standard)
ecg_tmid, ecg_hr = beat_hr(ecg_pk); ecg_hr_sm = smooth(ecg_tmid, ecg_hr)
ecg_v = good_mask(ecg_tmid, ecg_ivs)

# (A) REAL matched-only SCG series: just the 1665 matched SCG click times
real_scg = np.sort(pairs[:,0].astype("int64"))
tmid_r, hr_r = beat_hr(real_scg); hr_sm_r = smooth(tmid_r, hr_r)
v_r = np.ones(len(tmid_r), bool)   # already restricted to matched/good beats
xa_r, yb_r, ta_r = align(tmid_r, hr_sm_r, v_r, ecg_tmid, ecg_hr_sm, ecg_v, lag_us=LAG_US)
s_r = stats(xa_r, yb_r)

# (B) PHANTOM series: use the matched ECG click times (on SCG clock) instead of real SCG clicks
#     i.e., zero out the SCG-side timing jitter entirely for these same beats
phantom = np.sort(pairs[:,1].astype("int64"))
tmid_p, hr_p = beat_hr(phantom); hr_sm_p = smooth(tmid_p, hr_p)
v_p = np.ones(len(tmid_p), bool)
xa_p, yb_p, ta_p = align(tmid_p, hr_sm_p, v_p, ecg_tmid, ecg_hr_sm, ecg_v, lag_us=LAG_US)
s_p = stats(xa_p, yb_p)

print("(A) REAL matched-SCG-click series (n_clicks={}) vs hand ECG:".format(len(real_scg)))
print("   ", s_r)
print("(B) PHANTOM series (same beats, but using ECG's own click time instead of SCG's) vs hand ECG:")
print("   ", s_p)
print(f"\nr:  real={s_r['r']:.4f}   phantom(jitter-free)={s_p['r']:.4f}")
print(f"MAE: real={s_r['mae']:.2f}   phantom={s_p['mae']:.2f}")
