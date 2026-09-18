import pickle, sqlite3, datetime
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/singles_labels.pkl","rb") as f:
    lab = pickle.load(f)
with open("/tmp/failure_breakdown.pkl","rb") as f:
    fb = pickle.load(f)

LAG_US = int(7.0e6)

def load_peaks(path):
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = '' ORDER BY timestamp")
    return np.array([r[0] for r in cur.fetchall()], dtype="int64")

ecg_pk = load_peaks("/tmp/annotation_chelten_ecg2.db")

def beat_hr(peaks):
    rr = np.diff(peaks)/1e6
    hr = 60.0/rr
    tmid = peaks[:-1] + np.diff(peaks)//2
    return tmid, hr

def smooth(tmid, hr, win_s=3.0):
    win_us = int(win_s*1e6)
    out = np.empty(len(tmid))
    for i,t in enumerate(tmid):
        sel = (tmid>=t-win_us)&(tmid<=t+win_us)
        out[i] = hr[sel].mean()
    return out

baseline = np.sort(lab["singles_good"])
filtered = fb["kept"]

fp_example = fb["removed_fp_times"][0]

WIN = int(20e6)
t_lo, t_hi = fp_example - WIN, fp_example + WIN

def local_hr(peaks, lo, hi, pad=int(8e6)):
    m = (peaks >= lo-pad) & (peaks <= hi+pad)
    p = peaks[m]
    if len(p) < 2: return np.array([]), np.array([])
    tmid, hr = beat_hr(p)
    hr_sm = smooth(tmid, hr)
    m2 = (tmid>=lo)&(tmid<=hi)
    return tmid[m2], hr_sm[m2]

t_base, hr_base = local_hr(baseline, t_lo, t_hi)
t_filt, hr_filt = local_hr(filtered, t_lo, t_hi)
t_ecg, hr_ecg = local_hr(ecg_pk, t_lo-LAG_US, t_hi-LAG_US)
t_ecg_shifted = t_ecg + LAG_US  # plot on SCG time axis

fig, ax = plt.subplots(figsize=(12,5.5))
ax.plot((t_ecg_shifted-fp_example)/1e6, hr_ecg, "o-", color="black", ms=4, label="hand-clicked ECG (lag-corrected)")
ax.plot((t_base-fp_example)/1e6, hr_base, "o-", color="tab:blue", ms=4, alpha=0.8, label="baseline singles detector")
ax.plot((t_filt-fp_example)/1e6, hr_filt, "s--", color="tab:green", ms=5, alpha=0.8, label="+ wide two-lobe template filter")
ax.axvline(0, color="red", ls=":", lw=1.2, label="removed false positive")
ax.set_xlabel("seconds from the removed false-positive candidate (17:28:52.32)")
ax.set_ylabel("3s-smoothed HR (bpm) -- same smoothing used in the r/MAE comparison")
ax.set_title("Local effect of removing one false-positive candidate on the HR curve")
ax.legend()
plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_fp_hr_effect.png", dpi=130)
print("saved")
print("baseline points:", list(zip(t_base, hr_base)))
print("filtered points:", list(zip(t_filt, hr_filt)))
