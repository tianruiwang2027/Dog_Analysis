import pickle, sqlite3
import numpy as np
from scipy.signal import find_peaks

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)

ts_sd = d["ts_sd"]; sharp_s = d["sharp_s"]; fsd_s = d["fsd_s"]
ts_s = d["ts_s"]; x_s = d["x_s"]; xf_s = d["xf_s"]; fs_s = d["fs_s"]
t0 = int(d["t0"]); t1 = int(d["t1"])

THR = 0.3
REFRACT_S = 0.50
pk, _ = find_peaks(sharp_s, height=THR, distance=max(1,int(REFRACT_S*fsd_s)))
singles = ts_sd[pk]
print("n singles (raw, incl padding):", len(singles))

# restrict to the actual comparison window [t0, t1]
singles = singles[(singles>=t0)&(singles<=t1)]
print("n singles in-window:", len(singles))

# hand clicks + good-data mask, SCG side
con = sqlite3.connect("/tmp/annotation_chelten_scg2.db")
cur = con.cursor()
cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = '' ORDER BY timestamp")
hand = np.array([r[0] for r in cur.fetchall()], dtype="int64")
cur.execute("SELECT DISTINCT timestamp, label FROM events WHERE label != '' ORDER BY timestamp")
rows = cur.fetchall()
ev_ts = np.array([r[0] for r in rows], dtype="int64")
ev_lab = [r[1].lower() for r in rows]

def good_intervals(ev_ts, ev_lab, span_end):
    ivs = []; state="bad"; cur_start=None
    for t,lab in zip(ev_ts, ev_lab):
        if "good" in lab and "start" in lab:
            if state!="good": cur_start=t; state="good"
        elif "bad" in lab and "start" in lab:
            if state=="good": ivs.append((cur_start,t)); state="bad"
        elif "bad" in lab and "end" in lab:
            if state!="good": cur_start=t; state="good"
        elif "good" in lab and "end" in lab:
            if state=="good": ivs.append((cur_start,t)); state="bad"
        elif "stop" in lab:
            if state=="good": ivs.append((cur_start,t)); state="bad"
    if state=="good": ivs.append((cur_start, span_end))
    return ivs

span_end = max(hand.max(), t1)
ivs = good_intervals(ev_ts, ev_lab, span_end)

def good_mask(ts):
    g = np.zeros(len(ts), bool)
    for a,b in ivs:
        g |= (ts>=a)&(ts<b)
    return g

hand_w = hand[(hand>=t0)&(hand<=t1)]
hand_good = hand_w[good_mask(hand_w)]
singles_good = singles[good_mask(singles)]
print("n hand clicks in-window & good:", len(hand_good))
print("n singles in-window & good:", len(singles_good))

TOL_US = 100_000
# match singles -> hand clicks (nearest within tol), one-to-one greedy by time order
used_hand = np.zeros(len(hand_good), bool)
tp_times = []; fp_times = []
j = 0
for s in singles_good:
    # find nearest hand click
    idx = np.searchsorted(hand_good, s)
    best = None; bestd = TOL_US+1
    for cand_idx in (idx-1, idx):
        if 0 <= cand_idx < len(hand_good) and not used_hand[cand_idx]:
            dd = abs(hand_good[cand_idx]-s)
            if dd < bestd:
                bestd = dd; best = cand_idx
    if best is not None and bestd <= TOL_US:
        used_hand[best] = True
        tp_times.append(s)
    else:
        fp_times.append(s)

tp_times = np.array(tp_times, dtype="int64")
fp_times = np.array(fp_times, dtype="int64")
missed = hand_good[~used_hand]
print(f"TP={len(tp_times)}  FP={len(fp_times)}  missed={len(missed)}  (hand n={len(hand_good)})")
print(f"match rate = {len(tp_times)/len(hand_good)*100:.1f}%   FP rate = {len(fp_times)/len(singles_good)*100:.1f}%")

with open("/tmp/singles_labels.pkl","wb") as f:
    pickle.dump(dict(tp_times=tp_times, fp_times=fp_times, missed=missed,
                      singles_good=singles_good, hand_good=hand_good,
                      t0=t0, t1=t1), f)
