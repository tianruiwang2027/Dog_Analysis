import pickle, datetime
import numpy as np

with open("/tmp/combined_detector.pkl","rb") as f:
    c = pickle.load(f)

UTC = datetime.timezone.utc
def fmt(us):
    return datetime.datetime.fromtimestamp(us/1e6, tz=UTC).strftime("%H:%M:%S.%f")[:-3]

det = np.sort(c["primary"])

def beat_hr(peaks):
    rr = np.diff(peaks)/1e6; hr = 60.0/rr
    tmid = peaks[:-1] + np.diff(peaks)//2
    return tmid, hr

tmid, hr = beat_hr(det)

lo = int(datetime.datetime(2026,6,26,17,27,10,tzinfo=UTC).timestamp()*1e6)
hi = int(datetime.datetime(2026,6,26,17,27,40,tzinfo=UTC).timestamp()*1e6)

sel = (tmid>=lo)&(tmid<=hi)
print("peaks/RR in window 17:27:10-17:27:40:")
idxs = np.where(sel)[0]
for i in idxs:
    print(f"  tmid={fmt(tmid[i])}  RR={ (det[i+1]-det[i])/1e3:.0f}ms  instHR={hr[i]:.1f}")

print("\nnow let's check what smoothed HR looks like at 17:27:20.026, 17:27:21.254, 17:27:22.983")
def smooth_at(t, win_s=3.0):
    win_us=int(win_s*1e6)
    sel = (tmid>=t-win_us)&(tmid<=t+win_us)
    contributing = np.where(sel)[0]
    print(f"  t={fmt(t)}  window=[{fmt(t-win_us)},{fmt(t+win_us)}]  n_contrib={sel.sum()}  mean={hr[sel].mean():.2f}")
    for i in contributing:
        print(f"      contrib tmid={fmt(tmid[i])} instHR={hr[i]:.1f}")

for tt in ["17:27:20.026","17:27:21.254","17:27:22.983"]:
    h,m,s = tt.split(":")
    sec, ms = s.split(".")
    t = int(datetime.datetime(2026,6,26,17,27,int(sec),int(ms)*1000,tzinfo=UTC).timestamp()*1e6)
    smooth_at(t)
    print()
