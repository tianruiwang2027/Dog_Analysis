import pickle, sqlite3, datetime
import numpy as np

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

with open("/tmp/final_4way_compare.pkl","rb") as f:
    D = pickle.load(f)
t0, t1 = D["t0"], D["t1"]
span_end = max(ecg_pk.max(), scg_pk.max(), t1)
ecg_ivs = good_intervals(ecg_ev_ts, ecg_ev_lab, span_end)
scg_ivs = good_intervals(scg_ev_ts, scg_ev_lab, span_end)

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

ecg_tmid, ecg_hr = beat_hr(ecg_pk); ecg_hr_sm = smooth(ecg_tmid, ecg_hr)
scg_tmid, scg_hr = beat_hr(scg_pk); scg_hr_sm = smooth(scg_tmid, scg_hr)

# interpolate onto regular grid, only where BOTH have good data
GRID_S = 0.25
grid = np.arange(t0, t1, int(GRID_S*1e6))
ecg_on_grid = np.interp(grid, ecg_tmid, ecg_hr_sm, left=np.nan, right=np.nan)
scg_on_grid = np.interp(grid, scg_tmid, scg_hr_sm, left=np.nan, right=np.nan)
g_ecg = good_mask(grid, ecg_ivs); g_scg = good_mask(grid, scg_ivs)
both = g_ecg & g_scg & ~np.isnan(ecg_on_grid) & ~np.isnan(scg_on_grid)

def xcorr_best_lag(lo_idx, hi_idx, lag_range_s=(5.5,8.5), step_s=0.05):
    sub = both[lo_idx:hi_idx]
    if sub.sum() < 40: return None
    e = ecg_on_grid[lo_idx:hi_idx]
    s = scg_on_grid[lo_idx:hi_idx]
    best = None
    for lag_s in np.arange(lag_range_s[0], lag_range_s[1], step_s):
        shift = int(round(lag_s/GRID_S))
        # shift SCG forward by 'shift' grid steps to align with ECG: scg[i] ~ ecg[i-shift]... 
        # we want t_ecg_effective = t_scg + lag -> so compare e[i+shift] vs s[i]
        n = len(s)
        idx_s = np.arange(0, n-shift)
        idx_e = idx_s + shift
        if idx_e.max() >= n: continue
        m = sub[idx_s] & sub[idx_e]
        if m.sum() < 30: continue
        r = np.corrcoef(s[idx_s][m], e[idx_e][m])[0,1]
        if best is None or r > best[0]:
            best = (r, lag_s, m.sum())
    return best

UTC = datetime.timezone.utc
def fmt(us): return datetime.datetime.fromtimestamp(us/1e6, tz=UTC).strftime("%H:%M:%S")

N = len(grid)
NSEG = 8
seglen = N // NSEG
print("sliding-window optimal lag (interpolated cross-correlation, both-good only):")
for i in range(NSEG):
    lo, hi = i*seglen, min((i+1)*seglen, N)
    best = xcorr_best_lag(lo, hi)
    if best:
        print(f"  {fmt(grid[lo])}-{fmt(grid[hi-1])}  best_lag={best[1]:.2f}s  r={best[0]:.3f}  n={best[2]}")
    else:
        print(f"  {fmt(grid[lo])}-{fmt(grid[hi-1])}  insufficient good overlap")

best_full = xcorr_best_lag(0, N)
print(f"\nfull-span best lag: {best_full}")
