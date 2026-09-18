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

def good_mask(ts, ivs):
    g = np.zeros(len(ts), bool)
    for a,b in ivs: g |= (ts>=a)&(ts<b)
    return g

ecg_pk, ecg_ev_ts, ecg_ev_lab = load_peaks_and_good("/tmp/annotation_chelten_ecg2.db")
scg_pk, scg_ev_ts, scg_ev_lab = load_peaks_and_good("/tmp/annotation_chelten_scg2.db")

with open("/tmp/final_4way_compare.pkl","rb") as f:
    D = pickle.load(f)
t0, t1, LAG_US = D["t0"], D["t1"], D["LAG_US"]
span_end = max(ecg_pk.max(), scg_pk.max(), t1)
ecg_ivs = good_intervals(ecg_ev_ts, ecg_ev_lab, span_end)
scg_ivs = good_intervals(scg_ev_ts, scg_ev_lab, span_end)

ecg_on_scg = ecg_pk - LAG_US

scg_w = scg_pk[(scg_pk>=t0)&(scg_pk<=t1)]
scg_w = scg_w[good_mask(scg_w, scg_ivs) & good_mask(scg_w, ecg_ivs)]
ecg_w = ecg_on_scg[(ecg_on_scg>=t0)&(ecg_on_scg<=t1)]
ecg_w = ecg_w[good_mask(ecg_w, scg_ivs) & good_mask(ecg_w, ecg_ivs)]

print(f"n SCG (good both) = {len(scg_w)}   n ECG (good both) = {len(ecg_w)}")

TOL = int(300e3)
used_ecg = np.zeros(len(ecg_w), bool)
used_scg = np.zeros(len(scg_w), bool)
pairs = []
j0 = 0
for si, t in enumerate(scg_w):
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
        used_scg[si] = True
        pairs.append((t, ecg_w[best_j]))

unmatched_scg = scg_w[~used_scg]
unmatched_ecg = ecg_w[~used_ecg]
print(f"n matched pairs: {len(pairs)}")
print(f"n unmatched SCG clicks: {len(unmatched_scg)}")
print(f"n unmatched ECG clicks: {len(unmatched_ecg)}")

UTC = datetime.timezone.utc
print("\nUnmatched SCG clicks:")
for t in unmatched_scg:
    print(" ", datetime.datetime.fromtimestamp(t/1e6, tz=UTC).strftime("%H:%M:%S.%f")[:-3])
print("\nUnmatched ECG clicks (on SCG clock):")
for t in unmatched_ecg:
    print(" ", datetime.datetime.fromtimestamp(t/1e6, tz=UTC).strftime("%H:%M:%S.%f")[:-3])

with open("/tmp/unmatched_clicks.pkl","wb") as f:
    pickle.dump(dict(unmatched_scg=unmatched_scg, unmatched_ecg=unmatched_ecg, pairs=np.array(pairs),
                      t0=t0, t1=t1, LAG_US=LAG_US), f)
print("\nsaved /tmp/unmatched_clicks.pkl")
