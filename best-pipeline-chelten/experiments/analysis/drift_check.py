import pickle, sqlite3, datetime
import numpy as np

def load_peaks(path):
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = '' ORDER BY timestamp")
    return np.array([r[0] for r in cur.fetchall()], dtype="int64")

ecg_pk = load_peaks("/tmp/annotation_chelten_ecg2.db")
scg_pk = load_peaks("/tmp/annotation_chelten_scg2.db")

with open("/tmp/final_4way_compare.pkl","rb") as f:
    D = pickle.load(f)
t0, t1 = D["t0"], D["t1"]

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

ecg_tmid, ecg_hr = beat_hr(ecg_pk)
ecg_hr_sm = smooth(ecg_tmid, ecg_hr)
scg_tmid, scg_hr = beat_hr(scg_pk)
scg_hr_sm = smooth(scg_tmid, scg_hr)

def best_lag_r(scg_tmid, scg_hr_sm, ecg_tmid, ecg_hr_sm, lo, hi, lags_us):
    m_scg = (scg_tmid>=lo)&(scg_tmid<=hi)
    ts_s = scg_tmid[m_scg]; hr_s = scg_hr_sm[m_scg]
    best = None
    for lag in lags_us:
        t = ts_s + lag
        xa, yb = [], []
        for i in range(len(t)):
            sel = (ecg_tmid>=t[i]-1_000_000)&(ecg_tmid<=t[i]+1_000_000)
            if sel.sum()>=1:
                xa.append(hr_s[i]); yb.append(ecg_hr_sm[sel].mean())
        if len(xa)>10:
            r = np.corrcoef(xa,yb)[0,1]
            if best is None or r>best[0]:
                best = (r, lag, len(xa))
    return best

lags = np.arange(6.0e6, 8.0e6, 0.05e6)
# split into thirds
span = t1-t0
thirds = [(t0, t0+span//3), (t0+span//3, t0+2*span//3), (t0+2*span//3, t1)]
for lo,hi in thirds:
    best = best_lag_r(scg_tmid, scg_hr_sm, ecg_tmid, ecg_hr_sm, lo, hi, lags)
    UTC=datetime.timezone.utc
    print(f"{datetime.datetime.fromtimestamp(lo/1e6,tz=UTC).strftime('%H:%M:%S')}-{datetime.datetime.fromtimestamp(hi/1e6,tz=UTC).strftime('%H:%M:%S')}  best_lag={best[1]/1e6:.3f}s  r={best[0]:.4f}  n={best[2]}")

# also finer 6 segments
sixths = [(t0+i*span//6, t0+(i+1)*span//6) for i in range(6)]
print()
for lo,hi in sixths:
    best = best_lag_r(scg_tmid, scg_hr_sm, ecg_tmid, ecg_hr_sm, lo, hi, lags)
    UTC=datetime.timezone.utc
    print(f"{datetime.datetime.fromtimestamp(lo/1e6,tz=UTC).strftime('%H:%M:%S')}-{datetime.datetime.fromtimestamp(hi/1e6,tz=UTC).strftime('%H:%M:%S')}  best_lag={best[1]/1e6:.3f}s  r={best[0]:.4f}  n={best[2]}")
