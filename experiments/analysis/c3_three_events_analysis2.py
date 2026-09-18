import pickle
import numpy as np
import datetime
from scipy.signal import find_peaks

with open("/tmp/axes_zoom_data.pkl", "rb") as f:
    d = pickle.load(f)

ts_w = d["ts_w"]
xf = d["xf"]
fs = d["fs"]

def dt_to_us(dt):
    return int(dt.replace(tzinfo=datetime.timezone.utc).timestamp() * 1e6)

t0 = dt_to_us(datetime.datetime(2026,6,26,17,56,3,0))
t1 = dt_to_us(datetime.datetime(2026,6,26,17,56,6,300000))

m = (ts_w >= t0) & (ts_w <= t1)
ts_m = ts_w[m]
x = xf["c3"][m]

t_rel = (ts_m - ts_m[0]) / 1e6

# envelope via abs + smoothing
from scipy.ndimage import uniform_filter1d
env = uniform_filter1d(np.abs(x), size=int(0.02*fs))  # 20ms smoothing

# find burst peaks in envelope, well separated
peaks, props = find_peaks(env, height=15, distance=int(0.15*fs))
print("n envelope peaks:", len(peaks))
for p in peaks:
    print(f"t={t_rel[p]:.3f}s  env={env[p]:.1f}")
