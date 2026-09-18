import pickle, sqlite3, datetime
import numpy as np
from scipy.signal import find_peaks

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
ts_sd = d["ts_sd"]; env_sd = d["env_sd"]; sharp_s = d["sharp_s"]; fsd_s = d["fsd_s"]
ts_s = d["ts_s"]; xf_s = d["xf_s"]
with open("/tmp/env_template_wide.pkl","rb") as f:
    tpl = pickle.load(f)
template = tpl["template"]; HALF_N = tpl["HALF_N"]
template_energy = np.sqrt((template**2).sum())

def ncc_score(t):
    i = np.searchsorted(ts_sd, t)
    if i-HALF_N < 0 or i+HALF_N+1 > len(env_sd):
        return np.nan
    s = env_sd[i-HALF_N:i+HALF_N+1]
    s = s - s.mean()
    denom = np.sqrt((s**2).sum()) * template_energy
    if denom < 1e-12: return np.nan
    return float((s*template).sum() / denom)

# candidates = SAME primary-detector settings as the production pipeline (thr=0.3 on sharpened
# envelope, 0.5s refractory) -- this is the detector already validated to work well for HR estimation.
PRIMARY_THR = 0.3; PRIMARY_REFRACT_S = 0.50
pk_idx, _ = find_peaks(sharp_s, height=PRIMARY_THR, distance=max(1,int(PRIMARY_REFRACT_S*fsd_s)))
cand_t = ts_sd[pk_idx]
cand_ncc = np.array([ncc_score(t) for t in cand_t])
valid = ~np.isnan(cand_ncc)
cand_t = cand_t[valid]; cand_ncc = cand_ncc[valid]
print(f"n primary candidates: {len(cand_t)}")

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
def label_for(tc):
    return any(a<=tc<b for a,b in scg_ivs_hand)

WIN_S = 3.0; STEP_S = 1.0
win_us = int(WIN_S*1e6); step_us = int(STEP_S*1e6)
t0_all, t1_all = ts_s[0], ts_s[-1]
centers = np.arange(t0_all+win_us//2, t1_all-win_us//2, step_us)

def window_quality(tc, cutoff, gap_max_us, min_n=2):
    lo, hi = tc-win_us//2, tc+win_us//2
    # matched beats = primary candidates passing the shape cutoff, with a little context padding
    # so we can see the gap leading INTO and OUT OF the window too
    pad = int(1.0e6)
    m = (cand_t>=lo-pad)&(cand_t<hi+pad)&(cand_ncc>=cutoff)
    mt = np.sort(cand_t[m])
    inwin = (mt>=lo)&(mt<hi)
    n_in = inwin.sum()
    if n_in < min_n:
        return False, n_in, np.nan
    if len(mt) < 2:
        return False, n_in, np.nan
    gaps = np.diff(mt)
    # only consider gaps that actually overlap the window
    gap_lo = mt[:-1]; gap_hi = mt[1:]
    overlap = (gap_hi>lo)&(gap_lo<hi)
    rel_gaps = gaps[overlap]
    if len(rel_gaps)==0:
        return False, n_in, np.nan
    maxgap = rel_gaps.max()
    ok = maxgap <= gap_max_us
    return ok, n_in, maxgap

CUTOFF = 0.40
GAP_MAX_US = int(1.3e6)
rows = []
for tc in centers:
    ok, n_in, maxgap = window_quality(tc, CUTOFF, GAP_MAX_US)
    rows.append(dict(tc=tc, pred_good=ok, n_in=n_in, maxgap=maxgap, label_good=label_for(tc)))

y = np.array([0 if r["pred_good"] else 1 for r in rows])   # 1 = predicted bad
hand_bad = np.array([0 if r["label_good"] else 1 for r in rows])
tp=((y==1)&(hand_bad==1)).sum(); tn=((y==0)&(hand_bad==0)).sum()
fp=((y==1)&(hand_bad==0)).sum(); fn=((y==0)&(hand_bad==1)).sum()
acc=(tp+tn)/len(y); sens=tp/(tp+fn); spec=tn/(tn+fp)
print(f"CUTOFF={CUTOFF} GAP_MAX={GAP_MAX_US/1e6}s:  acc={acc*100:.1f}%  sens={sens*100:.1f}%  spec={spec*100:.1f}%")

with open("/tmp/mask_v9_rows.pkl","wb") as f:
    pickle.dump(dict(rows=rows, cand_t=cand_t, cand_ncc=cand_ncc, win_us=win_us, step_us=step_us,
                      CUTOFF=CUTOFF, GAP_MAX_US=GAP_MAX_US), f)
print("saved /tmp/mask_v9_rows.pkl")
