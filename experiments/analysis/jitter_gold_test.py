import pickle, sqlite3, datetime
import numpy as np

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

with open("/tmp/final_4way_compare.pkl","rb") as f:
    D = pickle.load(f)
t0, t1, LAG_US = D["t0"], D["t1"], D["LAG_US"]
span_end = max(ecg_pk.max(), scg_pk.max(), t1)
ecg_ivs = good_intervals(ecg_ev_ts, ecg_ev_lab, span_end)
scg_ivs = good_intervals(scg_ev_ts, scg_ev_lab, span_end)

def good_mask(ts, ivs):
    g = np.zeros(len(ts), bool)
    for a,b in ivs: g |= (ts>=a)&(ts<b)
    return g

ecg_on_scg = ecg_pk - LAG_US   # ECG clicks expressed on SCG clock

# restrict to good-on-both, in-window
scg_w = scg_pk[(scg_pk>=t0)&(scg_pk<=t1)]
scg_w = scg_w[good_mask(scg_w, scg_ivs) & good_mask(scg_w, ecg_ivs)]
ecg_w = ecg_on_scg[(ecg_on_scg>=t0)&(ecg_on_scg<=t1)]
ecg_w = ecg_w[good_mask(ecg_w, scg_ivs) & good_mask(ecg_w, ecg_ivs)]

print(f"n SCG (good both) = {len(scg_w)}   n ECG (good both) = {len(ecg_w)}")

# greedy nearest-neighbor 1:1 matching within +-150ms (S1 should follow R-wave by ~100-250ms typically,
# but LAG_US was fit to maximize correlation of smoothed HR, not to the EMD -- so search a symmetric window
# first to see where the offset naturally centers)
TOL = int(300e3)
used_ecg = np.zeros(len(ecg_w), bool)
pairs = []
j0 = 0
for t in scg_w:
    # advance j0 to first ecg_w candidate within reach
    while j0 < len(ecg_w) and ecg_w[j0] < t - TOL:
        j0 += 1
    best_j, best_d = None, TOL+1
    j = j0
    while j < len(ecg_w) and ecg_w[j] <= t + TOL:
        if not used_ecg[j]:
            d = abs(ecg_w[j]-t)
            if d < best_d:
                best_d = d; best_j = j
        j += 1
    if best_j is not None:
        used_ecg[best_j] = True
        pairs.append((t, ecg_w[best_j]))

pairs = np.array(pairs)
offsets_ms = (pairs[:,0]-pairs[:,1])/1e3
print(f"n matched pairs (1:1, within +-{TOL/1e3:.0f}ms): {len(pairs)}")
print(f"offset (SCG click - ECG click, on SCG clock): mean={offsets_ms.mean():+.1f}ms  median={np.median(offsets_ms):+.1f}ms  std={offsets_ms.std():.1f}ms")
print(f"percentiles: {np.percentile(offsets_ms,[5,25,50,75,95])}")

with open("/tmp/jitter_gold_pairs.pkl","wb") as f:
    pickle.dump(dict(pairs=pairs, offsets_ms=offsets_ms, t0=t0, t1=t1, LAG_US=LAG_US), f)
