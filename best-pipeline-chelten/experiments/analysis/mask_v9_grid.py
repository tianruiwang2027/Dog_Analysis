import pickle, sqlite3
import numpy as np

with open("/tmp/mask_v9_rows.pkl","rb") as f:
    D0 = pickle.load(f)
cand_t = D0["cand_t"]; cand_ncc = D0["cand_ncc"]; win_us = D0["win_us"]; step_us = D0["step_us"]

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

scg_pk, scg_ev_ts, scg_ev_lab = load_peaks_and_good("/tmp/annotation_chelten_scg2.db")
span_end = max(scg_pk.max(), ts_s.max())
scg_ivs_hand = good_intervals(scg_ev_ts, scg_ev_lab, span_end)

t0_all, t1_all = ts_s[0], ts_s[-1]
centers = np.arange(t0_all+win_us//2, t1_all-win_us//2, step_us)
hand_bad = np.array([not any(a<=tc<b for a,b in scg_ivs_hand) for tc in centers])
print(f"n windows={len(centers)}  hand_bad frac={hand_bad.mean()*100:.1f}%")

pad = int(1.0e6)

def evaluate(cutoff, gap_max_us, min_n):
    m_all = cand_ncc>=cutoff
    mt_all = np.sort(cand_t[m_all])
    pred_bad = np.zeros(len(centers), bool)
    for i,tc in enumerate(centers):
        lo, hi = tc-win_us//2, tc+win_us//2
        sel = (mt_all>=lo-pad)&(mt_all<hi+pad)
        mt = mt_all[sel]
        inwin = (mt>=lo)&(mt<hi)
        n_in = inwin.sum()
        if n_in < min_n or len(mt)<2:
            pred_bad[i] = True; continue
        gaps = np.diff(mt)
        gap_lo=mt[:-1]; gap_hi=mt[1:]
        overlap = (gap_hi>lo)&(gap_lo<hi)
        rel = gaps[overlap]
        if len(rel)==0:
            pred_bad[i]=True; continue
        pred_bad[i] = rel.max() > gap_max_us
    tp=((pred_bad)&(hand_bad)).sum(); tn=((~pred_bad)&(~hand_bad)).sum()
    fp=((pred_bad)&(~hand_bad)).sum(); fn=((~pred_bad)&(hand_bad)).sum()
    acc=(tp+tn)/len(centers); sens=tp/(tp+fn) if (tp+fn)>0 else np.nan; spec=tn/(tn+fp) if (tn+fp)>0 else np.nan
    return acc, sens, spec, pred_bad.mean()

best=None
results=[]
for cutoff in [0.2,0.3,0.4,0.5,0.6,0.7]:
    for gap_max_s in [1.0,1.2,1.5,2.0,2.5,3.0]:
        for min_n in [1,2,3]:
            acc,sens,spec,flagfrac = evaluate(cutoff, int(gap_max_s*1e6), min_n)
            j = sens+spec-1 if not (np.isnan(sens) or np.isnan(spec)) else -99
            results.append((cutoff,gap_max_s,min_n,acc,sens,spec,flagfrac,j))
            if best is None or j>best[-1]:
                best=(cutoff,gap_max_s,min_n,acc,sens,spec,flagfrac,j)

print("Top 10 by Youden J:")
for r in sorted(results, key=lambda x:-x[-1])[:10]:
    print(f"  cutoff={r[0]:.1f} gap_max={r[1]:.1f}s min_n={r[2]}  acc={r[3]*100:5.1f}% sens={r[4]*100:5.1f}% spec={r[5]*100:5.1f}%  flag%={r[6]*100:5.1f}%")

print("\nTop 10 by accuracy:")
for r in sorted(results, key=lambda x:-x[3])[:10]:
    print(f"  cutoff={r[0]:.1f} gap_max={r[1]:.1f}s min_n={r[2]}  acc={r[3]*100:5.1f}% sens={r[4]*100:5.1f}% spec={r[5]*100:5.1f}%  flag%={r[6]*100:5.1f}%")

with open("/tmp/mask_v9_grid.pkl","wb") as f:
    pickle.dump(dict(results=results, best=best, centers=centers, hand_bad=hand_bad), f)
print("\nsaved /tmp/mask_v9_grid.pkl")
