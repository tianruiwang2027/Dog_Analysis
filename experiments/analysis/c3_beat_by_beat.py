import pickle, datetime
import numpy as np
from scipy.signal import find_peaks

with open('/tmp/c3_doublet_check_results.pkl','rb') as f:
    d = pickle.load(f)
r = d['results']['c3']
ts_w = d['ts_w']; fs = d['fs']
xf = r['xf']  # bandpassed c3, native 2000Hz, aligned to ts_w

def to_us(dt):
    return int(dt.replace(tzinfo=datetime.timezone.utc).timestamp()*1e6)

ecg_clicks = ["17:56:03.633723","17:56:04.917833","17:56:06.106954","17:56:06.885760","17:56:07.596516"]
ecg_times = [to_us(datetime.datetime(2026,6,26,*map(int,[t.split(':')[0],t.split(':')[1],t.split(':')[2].split('.')[0]]),int(t.split('.')[1])*1000)) if False else None for t in ecg_clicks]

# simpler: hardcode as datetime directly
ecg_dts = [
    datetime.datetime(2026,6,26,17,56,3,633723),
    datetime.datetime(2026,6,26,17,56,4,917833),
    datetime.datetime(2026,6,26,17,56,6,106954),
    datetime.datetime(2026,6,26,17,56,6,885760),
    datetime.datetime(2026,6,26,17,56,7,596516),
]
ecg_us = [to_us(x) for x in ecg_dts]

BURST_HALF_MS = 45.0
FREQ_LO, FREQ_HI = 10, 100

def dominant_freq(t_center_us):
    lo = np.searchsorted(ts_w, t_center_us - int(BURST_HALF_MS*1000))
    hi = np.searchsorted(ts_w, t_center_us + int(BURST_HALF_MS*1000))
    if hi - lo < 8: return np.nan
    seg = xf[lo:hi] * np.hanning(hi-lo)
    n = 1 << int(np.ceil(np.log2(4*len(seg))))
    F = np.abs(np.fft.rfft(seg, n))
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    band = (freqs>=FREQ_LO)&(freqs<=FREQ_HI)
    if not band.any() or F[band].max()<=0: return np.nan
    return freqs[band][np.argmax(F[band])]

for ecg_t, ecg_dt in zip(ecg_us, ecg_dts):
    lo = np.searchsorted(ts_w, ecg_t + int(0.03e6))
    hi = np.searchsorted(ts_w, ecg_t + int(0.55e6))
    seg = np.abs(xf[lo:hi])
    pk, props = find_peaks(seg, distance=int(0.03*fs), height=15)
    print(f"\n--- ECG R at {ecg_dt.time()} ---")
    if len(pk)==0:
        print("  no bursts above 15 in window R+30..R+550ms")
        continue
    # report all local peaks found, sorted by height desc, then take top few in time order
    times_us = ts_w[lo:hi][pk]
    heights = seg[pk]
    order = np.argsort(times_us)
    times_us = times_us[order]; heights=heights[order]
    for t,h in zip(times_us, heights):
        delay_ms = (t-ecg_t)/1000
        f = dominant_freq(t)
        print(f"  burst at R+{delay_ms:6.1f}ms  |amp|={h:6.1f}  f_dom={f:.1f}Hz")
