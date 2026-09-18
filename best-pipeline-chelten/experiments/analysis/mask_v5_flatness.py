import pickle, sqlite3, datetime
import numpy as np

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
ts_s = d["ts_s"]; xf_s = d["xf_s"]; fs_s = d["fs_s"]
# xf_s is already bandpassed 10-100Hz -> respiration (<1Hz) and baseline drift are already removed.

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
scg_ivs = good_intervals(scg_ev_ts, scg_ev_lab, span_end)

def label_for(tc):
    return "good" if any(a<=tc<b for a,b in scg_ivs) else "bad"

WIN_S = 3.0
STEP_S = 1.0
win_us = int(WIN_S*1e6); step_us = int(STEP_S*1e6)
t0_all, t1_all = ts_s[0], ts_s[-1]
centers = np.arange(t0_all+win_us//2, t1_all-win_us//2, step_us)

# --- the ONE feature: how "flat" is the respiration-removed (bandpassed) trace in this window ---
# use RMS -- a flat/quiet trace has low RMS, a motion-contaminated trace has high RMS
rows = []
for tc in centers:
    lo, hi = tc-win_us//2, tc+win_us//2
    i0 = np.searchsorted(ts_s, lo); i1 = np.searchsorted(ts_s, hi)
    if i1-i0 < 100: continue
    xf_w = xf_s[i0:i1]
    rms = float(np.sqrt(np.mean(xf_w**2)))
    rows.append((tc, rms, label_for(tc)))

tc_arr = np.array([r[0] for r in rows], dtype="int64")
rms_arr = np.array([r[1] for r in rows])
lab_arr = np.array([r[2] for r in rows])

good_rms = rms_arr[lab_arr=="good"]; bad_rms = rms_arr[lab_arr=="bad"]
print(f"n good={len(good_rms)}  n bad={len(bad_rms)}")
print("GOOD rms percentiles [5,25,50,75,90,95,99]:", np.percentile(good_rms,[5,25,50,75,90,95,99]))
print("BAD  rms percentiles [5,25,50,75,90,95,99]:", np.percentile(bad_rms,[5,25,50,75,90,95,99]))

print()
print("=== single-threshold sweep: flag BAD if rms >= thr ===")
print(f"{'thr':>8} {'acc':>7} {'sens(bad)':>10} {'spec(good)':>11} {'flagged%':>9}")
y = (lab_arr=="bad").astype(int)
best = None
for thr in np.percentile(rms_arr, np.arange(5,96,2.5)):
    pred = (rms_arr>=thr).astype(int)
    tp=((pred==1)&(y==1)).sum(); tn=((pred==0)&(y==0)).sum()
    fp=((pred==1)&(y==0)).sum(); fn=((pred==0)&(y==1)).sum()
    acc=(tp+tn)/len(y); sens=tp/(tp+fn); spec=tn/(tn+fp)
    j = sens+spec-1  # Youden's J
    if best is None or j>best[0]:
        best = (j, thr, acc, sens, spec)
for thr in [10,15,20,25,30,40,50,60,80,100]:
    pred = (rms_arr>=thr).astype(int)
    tp=((pred==1)&(y==1)).sum(); tn=((pred==0)&(y==0)).sum()
    fp=((pred==1)&(y==0)).sum(); fn=((pred==0)&(y==1)).sum()
    acc=(tp+tn)/len(y); sens=tp/(tp+fn); spec=tn/(tn+fp); flagged=pred.mean()*100
    print(f"{thr:8.1f} {acc*100:6.1f}% {sens*100:9.1f}% {spec*100:10.1f}% {flagged:8.1f}%")

print(f"\nbest single threshold by Youden's J: thr={best[1]:.1f}  acc={best[2]*100:.1f}%  sens={best[3]*100:.1f}%  spec={best[4]*100:.1f}%")

with open("/tmp/mask_v5_flatness.pkl","wb") as f:
    pickle.dump(dict(tc=tc_arr, rms=rms_arr, label=lab_arr, scg_ivs=scg_ivs,
                      win_us=win_us, step_us=step_us, best_thr=best[1]), f)
print("saved /tmp/mask_v5_flatness.pkl")
