import pickle, sqlite3, datetime
import numpy as np

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
ts_s = d["ts_s"]; xf_s = d["xf_s"]; fs_s = d["fs_s"]

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
    win_us = int(win_s*1e6)
    out = np.empty(len(tmid))
    for i,t in enumerate(tmid):
        sel = (tmid>=t-win_us)&(tmid<=t+win_us)
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

def label_for(tc, ivs):
    return any(a<=tc<b for a,b in ivs)  # True = good

STEP_US = int(0.5e6)
WIN_LIST_S = [0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
PCTS = np.array([5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95])

t0_all, t1_all = ts_s[0], ts_s[-1]
grid = []   # list of dicts: WIN_S, pct, thr, acc, sens, spec, r, mae, n

for WIN_S in WIN_LIST_S:
    win_us = int(WIN_S*1e6)
    centers = np.arange(t0_all+win_us//2, t1_all-win_us//2, STEP_US)
    tcs, rmss, labs = [], [], []
    for tc in centers:
        lo, hi = tc-win_us//2, tc+win_us//2
        i0 = np.searchsorted(ts_s, lo); i1 = np.searchsorted(ts_s, hi)
        if i1-i0 < 20: continue
        xf_w = xf_s[i0:i1]
        rmss.append(float(np.sqrt(np.mean(xf_w**2))))
        tcs.append(tc)
        labs.append(label_for(tc, scg_ivs_hand))
    tcs = np.array(tcs, dtype="int64"); rmss = np.array(rmss); labs = np.array(labs)
    y = (~labs).astype(int)
    order = np.argsort(tcs); tcs_o = tcs[order]

    thr_vals = np.percentile(rmss, PCTS)
    for pct, THR in zip(PCTS, thr_vals):
        pred = (rmss>=THR).astype(int)
        tp=((pred==1)&(y==1)).sum(); tn=((pred==0)&(y==0)).sum()
        fp=((pred==1)&(y==0)).sum(); fn=((pred==0)&(y==1)).sum()
        acc=(tp+tn)/len(y); sens=tp/(tp+fn) if (tp+fn)>0 else np.nan; spec=tn/(tn+fp) if (tn+fp)>0 else np.nan

        good_s = (rmss[order] < THR)
        auto_ivs = []; cur_start=None
        for i in range(len(tcs_o)):
            if good_s[i]:
                if cur_start is None: cur_start = tcs_o[i]-STEP_US//2
            else:
                if cur_start is not None: auto_ivs.append((cur_start, tcs_o[i]-STEP_US//2)); cur_start=None
        if cur_start is not None: auto_ivs.append((cur_start, tcs_o[-1]+STEP_US//2))

        scg_v_auto = good_mask(scg_tmid, auto_ivs) & ecg_v_at_scg
        xa_a, yb_a = align(scg_tmid, scg_hr_sm, scg_v_auto, ecg_tmid, ecg_hr_sm, ecg_v_full, lag_us=LAG_US, restrict=RESTRICT)
        s_a = stats(xa_a, yb_a)

        grid.append(dict(WIN_S=WIN_S, pct=pct, thr=THR, acc=acc, sens=sens, spec=spec,
                          r=s_a["r"], mae=s_a["mae"], n=s_a["n"]))
    print(f"WIN={WIN_S:5.1f}s done ({len(thr_vals)} thresholds)")

with open("/tmp/mask_v8_grid_search.pkl","wb") as f:
    pickle.dump(dict(grid=grid, s_h=s_h, WIN_LIST_S=WIN_LIST_S, PCTS=PCTS), f)

# report best combos
valid = [g for g in grid if not np.isnan(g["r"]) and g["n"]>=500]
best_r = max(valid, key=lambda g: g["r"])
best_acc = max(valid, key=lambda g: g["acc"])
print()
print(f"BEST by r:        WIN={best_r['WIN_S']}s  pct={best_r['pct']}  thr={best_r['thr']:.2f}  "
      f"r={best_r['r']:.4f}  MAE={best_r['mae']:.2f}  n={best_r['n']}  acc={best_r['acc']*100:.1f}%  sens={best_r['sens']*100:.1f}%  spec={best_r['spec']*100:.1f}%")
print(f"BEST by accuracy:  WIN={best_acc['WIN_S']}s  pct={best_acc['pct']}  thr={best_acc['thr']:.2f}  "
      f"acc={best_acc['acc']*100:.1f}%  r={best_acc['r']:.4f}  MAE={best_acc['mae']:.2f}  n={best_acc['n']}  sens={best_acc['sens']*100:.1f}%  spec={best_acc['spec']*100:.1f}%")

print("\nsaved /tmp/mask_v8_grid_search.pkl")
