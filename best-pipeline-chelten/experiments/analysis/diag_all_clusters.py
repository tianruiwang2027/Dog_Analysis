import pickle, datetime, sqlite3
import numpy as np

with open("/tmp/combined_detector.pkl","rb") as f:
    c = pickle.load(f)
with open("/tmp/final_full_comparison_data.pkl","rb") as f:
    D = pickle.load(f)

UTC = datetime.timezone.utc
def fmt(us):
    return datetime.datetime.fromtimestamp(us/1e6, tz=UTC).strftime("%H:%M:%S.%f")[:-3]

det = np.sort(c["primary"])
def beat_hr(peaks):
    rr = np.diff(peaks)/1e6; hr = 60.0/rr
    tmid = peaks[:-1] + np.diff(peaks)//2
    return tmid, hr
tmid, hr = beat_hr(det)

ta_det = D["ta_det"]; xa_det = D["xa_det"]; yb_det = D["yb_det"]
err_det = xa_det - yb_det
idx = np.where(err_det < -18)[0]

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

scg_ivs = D["scg_ivs"]
def in_good(t):
    return any(a<=t<b for a,b in scg_ivs)

win_us = int(3e6)
for i in idx:
    t = ta_det[i]
    sel = np.where((tmid>=t-win_us)&(tmid<=t+win_us))[0]
    print(f"=== t={fmt(t)}  det_hr(smoothed)={xa_det[i]:.1f}  ecg_hr={yb_det[i]:.1f}  err={err_det[i]:.1f} ===")
    for j in sel:
        rr_ms = (det[j+1]-det[j])/1e3
        flag = "  <<< LONG RR (missed beat?)" if rr_ms>1200 else ""
        badflag = "" if in_good(tmid[j]) else "  [mask=BAD, excluded from own comparison]"
        print(f"    contrib tmid={fmt(tmid[j])} RR={rr_ms:.0f}ms instHR={hr[j]:.1f}{flag}{badflag}")
    print()
