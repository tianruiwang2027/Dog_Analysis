import pickle, sqlite3
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/combined_detector.pkl","rb") as f:
    c = pickle.load(f)
with open("/tmp/singles_labels.pkl","rb") as f:
    lab = pickle.load(f)

t0 = int(lab["t0"]); t1 = int(lab["t1"])
LAG_US = int(7.0e6)

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

def good_fn(ev_ts, ev_lab, span_end):
    ivs = good_intervals(ev_ts, ev_lab, span_end)
    def good(ts):
        g = np.zeros(len(ts), bool)
        for a,b in ivs: g |= (ts>=a)&(ts<b)
        return g
    return good

ecg_pk_full, ecg_ev_ts, ecg_ev_lab = load_peaks_and_good("/tmp/annotation_chelten_ecg2.db")
scg_pk_full, scg_ev_ts, scg_ev_lab = load_peaks_and_good("/tmp/annotation_chelten_scg2.db")
span_end = max(ecg_pk_full.max(), scg_pk_full.max(), t1)
ecg_good = good_fn(ecg_ev_ts, ecg_ev_lab, span_end)
scg_good = good_fn(scg_ev_ts, scg_ev_lab, span_end)

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

ecg_tmid, ecg_hr_raw = beat_hr(ecg_pk_full)
ecg_hr_sm = smooth(ecg_tmid, ecg_hr_raw)
ecg_v = ecg_good(ecg_tmid)

def per_point_errors(det_peaks, lag_us=LAG_US, win_us=2_000_000, restrict=(t0,t1)):
    tmid, hr_raw = beat_hr(det_peaks)
    hr_sm = smooth(tmid, hr_raw)
    v = scg_good(tmid)
    bts, bhr = ecg_tmid[ecg_v], ecg_hr_sm[ecg_v]
    errs = []; times=[]
    for i in np.where(v)[0]:
        traw = tmid[i]
        if not (restrict[0]<=traw<=restrict[1]): continue
        t = traw + lag_us
        sel = (bts>=t-win_us)&(bts<=t+win_us)
        if sel.sum()>=1:
            errs.append(hr_sm[i] - bhr[sel].mean())
            times.append(traw)
    return np.array(times), np.array(errs)

t_base, err_base = per_point_errors(np.sort(lab["singles_good"]))
t_filt, err_filt = per_point_errors(c["primary"])

print(f"baseline: n={len(err_base)}  MAE={np.abs(err_base).mean():.2f}  frac|err|>15bpm: {np.mean(np.abs(err_base)>15)*100:.1f}%")
print(f"filtered: n={len(err_filt)}  MAE={np.abs(err_filt).mean():.2f}  frac|err|>15bpm: {np.mean(np.abs(err_filt)>15)*100:.1f}%")

fig, ax = plt.subplots(figsize=(9,5.5))
bins = np.linspace(-40,40,81)
ax.hist(err_base, bins=bins, alpha=0.55, color="tab:blue", label=f"baseline (MAE={np.abs(err_base).mean():.2f})", density=True)
ax.hist(err_filt, bins=bins, alpha=0.55, color="tab:green", label=f"+ template filter (MAE={np.abs(err_filt).mean():.2f})", density=True)
ax.set_xlabel("HR error vs hand-clicked ECG (bpm), lag-corrected, 3s-smoothed")
ax.set_ylabel("density")
ax.set_title("Full-recording error distribution: baseline vs template-filtered detector")
ax.legend()
plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_error_distribution.png", dpi=130)
print("saved")
