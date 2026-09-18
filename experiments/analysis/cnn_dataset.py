import pickle, sqlite3
import numpy as np
from scipy.signal import find_peaks

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
ts_sd = d["ts_sd"]; env_sd = d["env_sd"]; sharp_s = d["sharp_s"]; fsd_s = d["fsd_s"]
t0 = int(d["t0"]); t1 = int(d["t1"])

# same primary-detector candidate generation used throughout (production settings)
PRIMARY_THR = 0.3; PRIMARY_REFRACT_S = 0.50
pk_idx, _ = find_peaks(sharp_s, height=PRIMARY_THR, distance=max(1,int(PRIMARY_REFRACT_S*fsd_s)))
cand_t = ts_sd[pk_idx]
cand_t = cand_t[(cand_t>=t0)&(cand_t<=t1)]
print(f"n candidates (whole session, incl bad-interval regions): {len(cand_t)}")

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

scg_pk, ev_ts, ev_lab = load_peaks_and_good("/tmp/annotation_chelten_scg2.db")
span_end = max(scg_pk.max(), t1)
scg_ivs = good_intervals(ev_ts, ev_lab, span_end)
cand_good_iv = good_mask(cand_t, scg_ivs)

# match every candidate to nearest hand click (greedy, 1:1), across the WHOLE session
# (not just good intervals) -- candidates in hand-labeled-bad stretches will essentially
# never find a match, which is exactly what we want: they become clean hard negatives.
TOL_US = 150_000
order = np.argsort(cand_t)
cand_t_sorted = cand_t[order]
used_hand = np.zeros(len(scg_pk), bool)
label = np.zeros(len(cand_t_sorted), dtype=int)
for i, s in enumerate(cand_t_sorted):
    idx = np.searchsorted(scg_pk, s)
    best = None; bestd = TOL_US+1
    for cidx in (idx-1, idx):
        if 0 <= cidx < len(scg_pk) and not used_hand[cidx]:
            dd = abs(scg_pk[cidx]-s)
            if dd < bestd:
                bestd = dd; best = cidx
    if best is not None and bestd <= TOL_US:
        used_hand[best] = True
        label[i] = 1

n_pos = label.sum(); n_neg = len(label)-n_pos
print(f"matched (positive/TP): {n_pos}   unmatched (negative/FP): {n_neg}")
neg_in_bad = (~cand_good_iv[order])[label==0].sum()
neg_in_good = (cand_good_iv[order])[label==0].sum()
print(f"  negatives from hand-labeled-BAD stretches: {neg_in_bad}")
print(f"  negatives from hand-labeled-GOOD stretches (ambiguous/near-miss): {neg_in_good}")

# extract +-400ms envelope snippets, same window as the NCC template
HALF_S = 0.40
HALF_N = int(HALF_S*fsd_s)

def snippet(t):
    i = np.searchsorted(ts_sd, t)
    if i-HALF_N < 0 or i+HALF_N+1 > len(env_sd):
        return None
    return env_sd[i-HALF_N:i+HALF_N+1].copy()

snips = []
keep_t = []
keep_label = []
for t, lab in zip(cand_t_sorted, label):
    s = snippet(t)
    if s is None: continue
    snips.append(s); keep_t.append(t); keep_label.append(lab)

snips = np.array(snips, dtype=np.float32)
keep_t = np.array(keep_t, dtype="int64")
keep_label = np.array(keep_label, dtype=np.float32)
print(f"final dataset: {len(keep_t)} candidates, snippet length {snips.shape[1]} samples "
      f"({HALF_S*1000:.0f}ms half-window @ {fsd_s:.0f}Hz)")
print(f"  positives={int(keep_label.sum())}  negatives={int((1-keep_label).sum())}")

with open("/tmp/cnn_dataset.pkl","wb") as f:
    pickle.dump(dict(t=keep_t, label=keep_label, snips=snips, HALF_N=HALF_N, fsd_s=fsd_s,
                      t0=t0, t1=t1, TOL_US=TOL_US), f)
print("saved /tmp/cnn_dataset.pkl")
