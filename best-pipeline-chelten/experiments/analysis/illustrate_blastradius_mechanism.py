import pickle, datetime
import numpy as np

def load_peaks(path):
    import sqlite3
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = '' ORDER BY timestamp")
    return np.array([r[0] for r in cur.fetchall()], dtype="int64")

scg_pk = load_peaks("/tmp/annotation_chelten_scg2.db")

with open("/tmp/jitter_gold_pairs.pkl","rb") as f:
    J = pickle.load(f)
LAG_US = J["LAG_US"]

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

tmid, hr = beat_hr(scg_pk)
hr_sm = smooth(tmid, hr)

UTC = datetime.timezone.utc
def fmt(us): return datetime.datetime.fromtimestamp(us/1e6, tz=UTC).strftime("%H:%M:%S.%f")[:-3]

# the true anomaly near 17:26:57 (unmatched SCG click)
lo = int(datetime.datetime(2026,6,26,17,26,50,tzinfo=UTC).timestamp()*1e6)
hi = int(datetime.datetime(2026,6,26,17,27,4,tzinfo=UTC).timestamp()*1e6)
sel = (tmid>=lo)&(tmid<=hi)
idxs = np.where(sel)[0]
print("clicks/RR/instHR/smoothedHR around the anomaly at 17:26:57:")
for i in idxs:
    rr_ms = (scg_pk[i+1]-scg_pk[i])/1e3
    print(f"  tmid={fmt(tmid[i])}  RR={rr_ms:.0f}ms  instHR={hr[i]:.1f}  smoothedHR={hr_sm[i]:.1f}")

with open("/tmp/blastradius_illustration_data.pkl","wb") as f:
    pickle.dump(dict(tmid=tmid, hr=hr, hr_sm=hr_sm, scg_pk=scg_pk, lo=lo, hi=hi), f)
