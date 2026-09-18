import pickle, sqlite3
import numpy as np

with open("/tmp/cnn_mask_apply.pkl","rb") as f:
    A = pickle.load(f)
cand_t = A["cand_t"]; cand_cnn = A["cand_cnn"]

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
ts_s = d["ts_s"]

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

scg_pk, ev_ts, ev_lab = load_peaks_and_good("/tmp/annotation_chelten_scg2.db")
span_end = max(scg_pk.max(), ts_s.max())
scg_ivs_hand = good_intervals(ev_ts, ev_lab, span_end)
def label_for(tc):
    return any(a<=tc<b for a,b in scg_ivs_hand)

WIN_S = 3.0; STEP_S = 1.0
win_us = int(WIN_S*1e6); step_us = int(STEP_S*1e6)
t0_all, t1_all = ts_s[0], ts_s[-1]
centers = np.arange(t0_all+win_us//2, t1_all-win_us//2, step_us)
pad = int(1.0e6)
hand_bad = np.array([0 if label_for(tc) else 1 for tc in centers])

def window_quality(cutoff, gap_max_us, min_n):
    m = cand_cnn>=cutoff
    mt_all = np.sort(cand_t[m])
    good_flags = np.zeros(len(centers), bool)
    for i,tc in enumerate(centers):
        lo, hi = tc-win_us//2, tc+win_us//2
        sel = (mt_all>=lo-pad)&(mt_all<hi+pad)
        mt = mt_all[sel]
        inwin = (mt>=lo)&(mt<hi)
        n_in = inwin.sum()
        if n_in < min_n or len(mt)<2:
            continue
        gaps = np.diff(mt)
        gap_lo=mt[:-1]; gap_hi=mt[1:]
        overlap = (gap_hi>lo)&(gap_lo<hi)
        rel = gaps[overlap]
        if len(rel)==0: continue
        good_flags[i] = rel.max() <= gap_max_us
    return good_flags

grid = []
for cutoff in [0.3,0.4,0.5,0.6,0.7,0.8]:
    for gap_max_s in [1.0,1.5,2.0,2.5,3.0]:
        for min_n in [1,2,3]:
            good = window_quality(cutoff, int(gap_max_s*1e6), min_n)
            y = (~good).astype(int)
            tp=((y==1)&(hand_bad==1)).sum(); tn=((y==0)&(hand_bad==0)).sum()
            fp=((y==1)&(hand_bad==0)).sum(); fn=((y==0)&(hand_bad==1)).sum()
            acc=(tp+tn)/len(y); sens=tp/(tp+fn) if (tp+fn)>0 else np.nan
            spec=tn/(tn+fp) if (tn+fp)>0 else np.nan
            J = sens+spec-1
            grid.append(dict(cutoff=cutoff, gap_max_s=gap_max_s, min_n=min_n, acc=acc, sens=sens, spec=spec, J=J))

grid.sort(key=lambda r: -r["J"])
print("Top 8 by Youden's J (window-level, vs hand labels):")
for r in grid[:8]:
    print(f"  cutoff={r['cutoff']:.1f} gap={r['gap_max_s']:.1f}s min_n={r['min_n']}  "
          f"acc={r['acc']*100:.1f}% sens={r['sens']*100:.1f}% spec={r['spec']*100:.1f}% J={r['J']:.3f}")

with open("/tmp/cnn_grid.pkl","wb") as f:
    pickle.dump(grid, f)
print("saved /tmp/cnn_grid.pkl")
