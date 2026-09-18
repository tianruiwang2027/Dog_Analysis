import pickle, sqlite3
import numpy as np

# Consolidates everything we've established into ONE reusable "valid SCG region" definition,
# to be used as the standard gold-standard restriction for all future pipeline/mask comparisons:
#   1. hand-labeled good on SCG AND good on ECG (the original quality annotation)
#   2. NOT within +-3.0s of any of the 24 known click-count mismatches (missed/extra/double clicks)
#   3. NOT within +-3.0s of any SCG hand-labeled bad-interval edge (smoothing contamination bleed)
# radius = 3.0s because that's the smooth() window used everywhere in this analysis (win_s=3.0),
# i.e. the radius within which one bad instantaneous-HR sample can contaminate a neighboring
# "good" smoothed-HR sample.

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

def intersect_intervals(ivs_a, ivs_b):
    out = []
    for a0,a1 in ivs_a:
        for b0,b1 in ivs_b:
            lo,hi = max(a0,b0), min(a1,b1)
            if hi>lo: out.append((lo,hi))
    return sorted(out)

def subtract_point_radius(ivs, points, radius_us):
    """remove [p-radius, p+radius] from every interval, for each point"""
    for p in points:
        lo, hi = p-radius_us, p+radius_us
        new = []
        for a,b in ivs:
            if hi<=a or lo>=b:
                new.append((a,b)); continue
            if lo>a: new.append((a,lo))
            if hi<b: new.append((hi,b))
        ivs = new
    return sorted(ivs)

ecg_pk, ecg_ev_ts, ecg_ev_lab = load_peaks_and_good("/tmp/annotation_chelten_ecg2.db")
scg_pk, scg_ev_ts, scg_ev_lab = load_peaks_and_good("/tmp/annotation_chelten_scg2.db")
span_end = max(ecg_pk.max(), scg_pk.max())
ecg_ivs = good_intervals(ecg_ev_ts, ecg_ev_lab, span_end)
scg_ivs = good_intervals(scg_ev_ts, scg_ev_lab, span_end)

with open("/tmp/unmatched_clicks.pkl","rb") as f:
    U = pickle.load(f)
mismatch_times = np.concatenate([U["unmatched_scg"], U["unmatched_ecg"]])

with open("/tmp/remove_badinterval_result.pkl","rb") as f:
    B = pickle.load(f)
bad_edges = B["bad_edges"]

RADIUS_US = int(3.0e6)

both_good = intersect_intervals(scg_ivs, ecg_ivs)
valid = subtract_point_radius(both_good, mismatch_times, RADIUS_US)
valid = subtract_point_radius(valid, bad_edges, RADIUS_US)

total_span = span_end - min(scg_pk.min(), ecg_pk.min())
covered = sum(b-a for a,b in valid)
print(f"n valid intervals: {len(valid)}")
print(f"total valid coverage: {covered/1e6:.0f}s")
print(f"(both-good-only coverage was: {sum(b-a for a,b in both_good)/1e6:.0f}s)")

def valid_mask(ts, ivs=valid):
    ts = np.asarray(ts)
    g = np.zeros(len(ts), bool)
    for a,b in ivs: g |= (ts>=a)&(ts<b)
    return g

with open("/tmp/valid_regions.pkl","wb") as f:
    pickle.dump(dict(valid_ivs=valid, both_good_ivs=both_good, mismatch_times=mismatch_times,
                      bad_edges=bad_edges, RADIUS_US=RADIUS_US), f)
print("saved /tmp/valid_regions.pkl  -- use valid_ivs as the standard gold-standard restriction going forward")
