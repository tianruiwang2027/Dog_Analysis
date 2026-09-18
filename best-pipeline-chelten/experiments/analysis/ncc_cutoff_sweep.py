import pickle, sqlite3, datetime
import numpy as np
from scipy.signal import find_peaks

with open("/tmp/singles_labels.pkl","rb") as f:
    lab = pickle.load(f)
with open("/tmp/env_template_wide.pkl","rb") as f:
    tpl = pickle.load(f)
with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
with open("/tmp/final_full_comparison_data.pkl","rb") as f:
    D = pickle.load(f)

ts_sd = d["ts_sd"]; env_sd = d["env_sd"]; sharp_s = d["sharp_s"]; fsd_s = d["fsd_s"]
template = tpl["template"]; HALF_N = tpl["HALF_N"]
template_energy = np.sqrt((template**2).sum())

def ncc_score(t):
    i = np.searchsorted(ts_sd, t)
    if i-HALF_N < 0 or i+HALF_N+1 > len(env_sd): return np.nan
    s = env_sd[i-HALF_N:i+HALF_N+1]; s = s - s.mean()
    denom = np.sqrt((s**2).sum()) * template_energy
    if denom < 1e-12: return np.nan
    return float((s*template).sum() / denom)

t0, t1, LAG_US = D["t0"], D["t1"], D["LAG_US"]
scg_ivs = sorted(D["scg_ivs"])
ecg_tmid, ecg_hr_sm, ecg_v = D["ecg_tmid"], D["ecg_hr_sm"], D["ecg_v"]

def both_good(ts):
    # scg_ivs here already reflects the SCG channel's own good/bad annotation
    g = np.zeros(len(ts), bool)
    for a,b in scg_ivs: g |= (ts>=a)&(ts<b)
    return g

# base candidate pool: same amplitude-threshold detector as before (thr=0.3, refract=0.5s)
PRIMARY_THR = 0.3; PRIMARY_REFRACT_S = 0.50
pk, _ = find_peaks(sharp_s, height=PRIMARY_THR, distance=max(1,int(PRIMARY_REFRACT_S*fsd_s)))
primary_all = ts_sd[pk]
primary_all = primary_all[(primary_all>=t0)&(primary_all<=t1)]
primary_all = primary_all[both_good(primary_all)]
primary_scores = np.array([ncc_score(t) for t in primary_all])
print("n base candidates (thr=0.3, both-good masked):", len(primary_all))

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

def align(tsA, hrA, vA, tsB, hrB, vB, lag_us=0, win_us=2_000_000, restrict=(t0,t1)):
    bts, bhr = tsB[vB], hrB[vB]
    A = np.where(vA)[0]
    xa, yb, ta = [], [], []
    for i in A:
        traw = tsA[i]
        if restrict is not None and not (restrict[0] <= traw <= restrict[1]): continue
        t = traw + lag_us
        sel = (bts>=t-win_us)&(bts<=t+win_us)
        if sel.sum()>=1:
            xa.append(hrA[i]); yb.append(bhr[sel].mean()); ta.append(traw)
    return np.array(xa), np.array(yb), np.array(ta, dtype="int64")

def strict_good_mask_for_pairs(peaks, ivs):
    a = peaks[:-1]; b = peaks[1:]
    ok = np.zeros(len(a), bool)
    for ga, gb in ivs: ok |= (a>=ga) & (b<=gb)
    return ok

# also load labeled TP/FP (test half) NCC scores for tradeoff context
order = np.argsort(lab["tp_times"])
tp_sorted = lab["tp_times"][order]
n_train = len(tp_sorted)//2
test_tp = tp_sorted[n_train:]
fp_times = lab["fp_times"]
tp_scores_labeled = np.array([ncc_score(t) for t in test_tp])
fp_scores_labeled = np.array([ncc_score(t) for t in fp_times])

print(f"\n{'cutoff':>7} {'n_kept':>8} {'r':>7} {'MAE':>6} {'bias':>7}   {'TP%kept':>8} {'FP%kept':>8}")
results = {}
for cutoff in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    primary = np.sort(primary_all[primary_scores>=cutoff])
    det_tmid, det_hr_raw = beat_hr(primary)
    det_hr_sm = smooth(det_tmid, det_hr_raw)
    det_v = strict_good_mask_for_pairs(primary, scg_ivs)
    xa, yb, ta = align(det_tmid, det_hr_sm, det_v, ecg_tmid, ecg_hr_sm, ecg_v, lag_us=LAG_US)
    s = stats(xa, yb)
    tp_pct = np.nanmean(tp_scores_labeled>=cutoff)*100
    fp_pct = np.nanmean(fp_scores_labeled>=cutoff)*100
    print(f"{cutoff:7.2f} {len(primary):8d} {s['r']:7.3f} {s['mae']:6.2f} {s['bias']:+7.2f}   {tp_pct:7.1f}% {fp_pct:7.1f}%")
    results[cutoff] = dict(primary=primary, xa=xa, yb=yb, ta=ta, s=s, tp_pct=tp_pct, fp_pct=fp_pct)

with open("/tmp/ncc_cutoff_sweep.pkl","wb") as f:
    pickle.dump(results, f)
print("\nsaved")
