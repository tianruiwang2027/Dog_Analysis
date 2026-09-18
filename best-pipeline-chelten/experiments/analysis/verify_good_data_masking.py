import pickle, sqlite3
import numpy as np

with open("/tmp/combined_detector.pkl","rb") as f:
    c = pickle.load(f)
with open("/tmp/singles_labels.pkl","rb") as f:
    lab = pickle.load(f)

t0 = int(lab["t0"]); t1 = int(lab["t1"])
LAG_US = int(7.0e6)

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

ecg_pk_full, ecg_ev_ts, ecg_ev_lab = load_peaks_and_good("/tmp/annotation_chelten_ecg2.db")
scg_pk_full, scg_ev_ts, scg_ev_lab = load_peaks_and_good("/tmp/annotation_chelten_scg2.db")
span_end = max(ecg_pk_full.max(), scg_pk_full.max(), t1)
ecg_ivs = good_intervals(ecg_ev_ts, ecg_ev_lab, span_end)
scg_ivs = good_intervals(scg_ev_ts, scg_ev_lab, span_end)

def in_any(t, ivs):
    for a,b in ivs:
        if a<=t<b: return True
    return False

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

print(f"Total good intervals -- ECG: {len(ecg_ivs)}, SCG: {len(scg_ivs)}")
total_ecg_good_s = sum((b-a) for a,b in ecg_ivs)/1e6
total_scg_good_s = sum((b-a) for a,b in scg_ivs)/1e6
print(f"Total ECG-labeled-good duration: {total_ecg_good_s:.1f}s = {total_ecg_good_s/60:.1f} min")
print(f"Total SCG-labeled-good duration: {total_scg_good_s:.1f}s = {total_scg_good_s/60:.1f} min")

# STRICT check: for each RR-derived point actually used in the baseline comparison,
# verify BOTH the peak before and the peak after (the two endpoints that made this
# RR interval) are themselves inside a good interval -- not just the midpoint.
singles_good = np.sort(lab["singles_good"])
tmid, hr_raw = beat_hr(singles_good)
hr_sm = smooth(tmid, hr_raw)
v_scg = good_mask(tmid, scg_ivs)

n_used = 0; n_endpoints_bad = 0
for i in np.where(v_scg)[0]:
    if t0 <= tmid[i] <= t1:
        n_used += 1
        p_before, p_after = singles_good[i], singles_good[i+1]
        if not (in_any(p_before, scg_ivs) and in_any(p_after, scg_ivs)):
            n_endpoints_bad += 1

print(f"\nOf {n_used} SCG-detector points passing the midpoint-in-good-interval check,")
print(f"{n_endpoints_bad} have at least one RR endpoint that is NOT itself in a good interval")
print(f"(i.e. midpoint looked 'good' only because it's just past a bad->good boundary)")
