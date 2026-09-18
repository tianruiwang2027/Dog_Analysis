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

# FULL ECG reference (standard, official)
ecg_tmid, ecg_hr = beat_hr(ecg_pk); ecg_hr_sm = smooth(ecg_tmid, ecg_hr)
ecg_v_full = good_mask(ecg_tmid, ecg_ivs)

# FULL SCG click list, standard official good-mask (this should reproduce r=0.897)
scg_tmid, scg_hr = beat_hr(scg_pk); scg_hr_sm = smooth(scg_tmid, scg_hr)
scg_v_full = good_mask(scg_tmid, scg_ivs)
xa0, yb0, ta0 = align(scg_tmid, scg_hr_sm, scg_v_full, ecg_tmid, ecg_hr_sm, ecg_v_full, lag_us=LAG_US)
s0 = stats(xa0, yb0)
print("SANITY CHECK -- official full SCG-click series vs ECG (should be ~0.897):")
print("   ", s0)

# Build a JITTER-CORRECTED version of the FULL scg_pk list: replace matched clicks' timestamps
# with their matched ECG-derived time; leave UNMATCHED clicks (and clicks outside the matched-pair
# analysis window) exactly as they were. This preserves full adjacency -- no clicks removed.
scg_pk_corrected = scg_pk.copy()
# build a dict: original SCG time -> corrected time
correction = {int(a): int(b) for a,b in pairs}
for i in range(len(scg_pk_corrected)):
    t = int(scg_pk_corrected[i])
    if t in correction:
        scg_pk_corrected[i] = correction[t]
scg_pk_corrected = np.sort(scg_pk_corrected)

n_changed = int((scg_pk_corrected != np.sort(scg_pk)).sum())
print(f"\nn clicks whose time actually changed after correction: {n_changed} / {len(scg_pk)}")

tmid_c, hr_c = beat_hr(scg_pk_corrected); hr_sm_c = smooth(tmid_c, hr_c)
v_c = good_mask(tmid_c, scg_ivs)  # same good-mask definition as before (intervals unaffected by tiny time shifts)
xa_c, yb_c, ta_c = align(tmid_c, hr_sm_c, v_c, ecg_tmid, ecg_hr_sm, ecg_v_full, lag_us=LAG_US)
s_c = stats(xa_c, yb_c)
print("\nJITTER-CORRECTED (matched clicks snapped to ECG time, adjacency preserved) vs ECG:")
print("   ", s_c)

print(f"\nr:  original={s0['r']:.4f}   jitter-corrected={s_c['r']:.4f}   (n={s0['n']} vs {s_c['n']})")
print(f"MAE: original={s0['mae']:.2f}   jitter-corrected={s_c['mae']:.2f}")
