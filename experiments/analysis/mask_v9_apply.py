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

def good_mask(ts, ivs):
    g = np.zeros(len(ts), bool)
    for a,b in ivs: g |= (ts>=a)&(ts<b)
    return g

def beat_hr(peaks):
    rr = np.diff(peaks)/1e6; hr = 60.0/rr
    tmid = peaks[:-1] + np.diff(peaks)//2
    return tmid, hr

def smooth(tmid, hr, win_s=3.0):
    win_us2 = int(win_s*1e6)
    out = np.empty(len(tmid))
    for i,t in enumerate(tmid):
        sel = (tmid>=t-win_us2)&(tmid<=t+win_us2)
        out[i] = hr[sel].mean()
    return out

def stats(a,b):
    dd=a-b
    return dict(n=int(len(a)), bias=float(dd.mean()), mae=float(np.abs(dd).mean()),
                r=float(np.corrcoef(a,b)[0,1]) if len(a)>2 else float("nan"))

def align(tsA, hrA, vA, tsB, hrB, vB, lag_us=0, win_us=2_000_000, restrict=None):
    bts, bhr = tsB[vB], hrB[vB]
    A = np.where(vA)[0]
    xa, yb = [], []
    for i in A:
        traw = tsA[i]
        if restrict is not None and not (restrict[0] <= traw <= restrict[1]): continue
        t = traw + lag_us
        sel = (bts>=t-win_us)&(bts<=t+win_us)
        if sel.sum()>=1:
            xa.append(hrA[i]); yb.append(bhr[sel].mean())
    return np.array(xa), np.array(yb)

scg_pk, scg_ev_ts, scg_ev_lab = load_peaks_and_good("/tmp/annotation_chelten_scg2.db")
ecg_pk, ecg_ev_ts, ecg_ev_lab = load_peaks_and_good("/tmp/annotation_chelten_ecg2.db")
span_end = max(scg_pk.max(), ecg_pk.max(), ts_s.max())
scg_ivs_hand = good_intervals(scg_ev_ts, scg_ev_lab, span_end)
ecg_ivs = good_intervals(ecg_ev_ts, ecg_ev_lab, span_end)

with open("/tmp/final_4way_compare.pkl","rb") as f:
    D = pickle.load(f)
LAG_US = D["LAG_US"]; RESTRICT = (D["t0"], D["t1"])

ecg_tmid, ecg_hr = beat_hr(ecg_pk); ecg_hr_sm = smooth(ecg_tmid, ecg_hr)
scg_tmid, scg_hr = beat_hr(scg_pk); scg_hr_sm = smooth(scg_tmid, scg_hr)
ecg_v_full = good_mask(ecg_tmid, ecg_ivs)
ecg_v_at_scg = good_mask(scg_tmid, ecg_ivs)
scg_v_hand = good_mask(scg_tmid, scg_ivs_hand) & ecg_v_at_scg

xa_h, yb_h = align(scg_tmid, scg_hr_sm, scg_v_hand, ecg_tmid, ecg_hr_sm, ecg_v_full, lag_us=LAG_US, restrict=RESTRICT)
s_h = stats(xa_h, yb_h)
print(f"HAND baseline:  n={s_h['n']}  r={s_h['r']:.4f}  MAE={s_h['mae']:.2f}")
print()

t0_all, t1_all = ts_s[0], ts_s[-1]
centers = np.arange(t0_all+win_us//2, t1_all-win_us//2, step_us)
pad = int(1.0e6)

def make_auto_ivs(cutoff, gap_max_us, min_n):
    m_all = cand_ncc>=cutoff
    mt_all = np.sort(cand_t[m_all])
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
    order = np.argsort(centers); tc_s = centers[order]; good_s = good_flags[order]
    ivs=[]; cur_start=None
    for i in range(len(tc_s)):
        if good_s[i]:
            if cur_start is None: cur_start = tc_s[i]-step_us//2
        else:
            if cur_start is not None: ivs.append((cur_start, tc_s[i]-step_us//2)); cur_start=None
    if cur_start is not None: ivs.append((cur_start, tc_s[-1]+step_us//2))
    return ivs

configs = [
    ("cutoff=0.7 gap=2.0s min_n=2 (best J)",   0.7, 2.0, 2),
    ("cutoff=0.7 gap=3.0s min_n=1 (best acc)", 0.7, 3.0, 1),
    ("cutoff=0.6 gap=1.5s min_n=3",            0.6, 1.5, 3),
    ("cutoff=0.7 gap=2.5s min_n=3",            0.7, 2.5, 3),
    ("cutoff=0.8 gap=2.0s min_n=2",            0.8, 2.0, 2),
    ("cutoff=0.8 gap=1.5s min_n=2",            0.8, 1.5, 2),
]

results = {}
for name, cutoff, gap_max_s, min_n in configs:
    auto_ivs = make_auto_ivs(cutoff, int(gap_max_s*1e6), min_n)
    scg_v_auto = good_mask(scg_tmid, auto_ivs) & ecg_v_at_scg
    xa_a, yb_a = align(scg_tmid, scg_hr_sm, scg_v_auto, ecg_tmid, ecg_hr_sm, ecg_v_full, lag_us=LAG_US, restrict=RESTRICT)
    s_a = stats(xa_a, yb_a)
    in_win = (scg_tmid>=RESTRICT[0])&(scg_tmid<=RESTRICT[1])
    hv = scg_v_hand[in_win]; av = scg_v_auto[in_win]
    agree = (hv==av).mean()
    print(f"{name:38s}  n={s_a['n']:5d}  r={s_a['r']:.4f}  MAE={s_a['mae']:.2f}  bias={s_a['bias']:+.2f}  beat_agree={agree*100:.1f}%")
    results[name] = dict(auto_ivs=auto_ivs, s_a=s_a, agree=agree, cutoff=cutoff, gap_max_s=gap_max_s, min_n=min_n)

with open("/tmp/mask_v9_apply.pkl","wb") as f:
    pickle.dump(dict(results=results, s_h=s_h, scg_tmid=scg_tmid, scg_hr_sm=scg_hr_sm,
                      scg_v_hand=scg_v_hand, LAG_US=LAG_US, RESTRICT=RESTRICT), f)
print("\nsaved /tmp/mask_v9_apply.pkl")
