import pickle, sqlite3, datetime
import numpy as np

with open("/tmp/jitter_gold_pairs.pkl","rb") as f:
    J = pickle.load(f)
pairs, t0, t1, LAG_US = J["pairs"], J["t0"], J["t1"], J["LAG_US"]
matched_scg_times = set(int(a) for a in pairs[:,0])
matched_ecg_times = set(int(b) for b in pairs[:,1])

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
ecg_on_scg = ecg_pk - LAG_US

scg_w = scg_pk[(scg_pk>=t0)&(scg_pk<=t1)]
scg_w = scg_w[good_mask(scg_w, scg_ivs) & good_mask(scg_w, ecg_ivs)]
ecg_w = ecg_on_scg[(ecg_on_scg>=t0)&(ecg_on_scg<=t1)]
ecg_w = ecg_w[good_mask(ecg_w, scg_ivs) & good_mask(ecg_w, ecg_ivs)]

# the TRUE individual anomalies: SCG clicks with no ECG partner, and ECG clicks with no SCG partner
unmatched_scg = np.array(sorted(t for t in scg_w if int(t) not in matched_scg_times))
unmatched_ecg = np.array(sorted(t for t in ecg_w if int(t) not in matched_ecg_times))
print(f"TRUE individual anomalies: {len(unmatched_scg)} unmatched SCG clicks + {len(unmatched_ecg)} unmatched ECG clicks = {len(unmatched_scg)+len(unmatched_ecg)} total")

anomaly_locs = np.sort(np.concatenate([unmatched_scg, unmatched_ecg]))
print("\nlocations (on SCG clock):")
UTC = datetime.timezone.utc
for t in anomaly_locs:
    print("  ", datetime.datetime.fromtimestamp(t/1e6, tz=UTC).strftime("%H:%M:%S.%f")[:-3])

with open("/tmp/final_4way_compare.pkl","rb") as f:
    D = pickle.load(f)
ta_scg = D["ta_scg"]
WIN = int(2.5e6)
flagged = np.zeros(len(ta_scg), bool)
for loc in anomaly_locs:
    flagged |= (np.abs(ta_scg-loc) <= WIN)
print(f"\ncomparison points within 2.5s of >=1 true anomaly: {flagged.sum()} / {len(ta_scg)}  ({100*flagged.mean():.1f}%)")
print("(compare to the 12.2% figure from before)")

avg_rr_s = np.median(np.diff(scg_w))/1e6
print(f"\nmedian RR ~ {avg_rr_s*1000:.0f}ms -> a +-2.5s window spans about {2*2.5/avg_rr_s:.0f} comparison points on each side of a single anomaly")
