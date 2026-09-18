import pickle, datetime
import numpy as np

with open("/tmp/combined_detector.pkl","rb") as f:
    c = pickle.load(f)
with open("/tmp/final_full_comparison_data.pkl","rb") as f:
    D = pickle.load(f)

det = np.sort(c["primary"])
scg_ivs = sorted(D["scg_ivs"])
t0, t1, LAG_US = D["t0"], D["t1"], D["LAG_US"]

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
        if restrict is not None and not (restrict[0] <= traw <= restrict[1]):
            continue
        t = traw + lag_us
        sel = (bts>=t-win_us)&(bts<=t+win_us)
        if sel.sum()>=1:
            xa.append(hrA[i]); yb.append(bhr[sel].mean()); ta.append(traw)
    return np.array(xa), np.array(yb), np.array(ta, dtype="int64")

# STRICT mask: an RR interval [a,b] only counts as "good" if it is fully
# contained within a SINGLE continuous good-labeled stretch (no bad sub-patch
# can be straddled), instead of only checking whether the interval's midpoint
# lands in a good stretch.
def strict_good_mask_for_pairs(peaks, ivs):
    a = peaks[:-1]; b = peaks[1:]
    ok = np.zeros(len(a), bool)
    for ga, gb in ivs:
        ok |= (a>=ga) & (b<=gb)
    return ok

det_tmid, det_hr_raw = beat_hr(det)
det_hr_sm = smooth(det_tmid, det_hr_raw)

det_v_old = D["det_v"]                                  # midpoint-only check (original)
det_v_strict = strict_good_mask_for_pairs(det, scg_ivs)  # full-span check (fixed)

print("n det RR intervals total:", len(det_tmid))
print("n good under OLD (midpoint-only) mask:   ", det_v_old.sum())
print("n good under STRICT (full-span) mask:    ", det_v_strict.sum())
print("n flipped good->bad by strict check:      ", int((det_v_old & ~det_v_strict).sum()))
print("n flipped bad->good by strict check:      ", int((~det_v_old & det_v_strict).sum()))

ecg_tmid, ecg_hr_sm, ecg_v = D["ecg_tmid"], D["ecg_hr_sm"], D["ecg_v"]

xa_old, yb_old, ta_old = align(det_tmid, det_hr_sm, det_v_old, ecg_tmid, ecg_hr_sm, ecg_v, lag_us=LAG_US)
xa_new, yb_new, ta_new = align(det_tmid, det_hr_sm, det_v_strict, ecg_tmid, ecg_hr_sm, ecg_v, lag_us=LAG_US)

s_old = stats(xa_old, yb_old)
s_new = stats(xa_new, yb_new)
print("\nOLD (midpoint-only mask):   ", s_old)
print("NEW (strict full-span mask):", s_new)

# how many of the originally-flagged worst 16 points are gone now?
err_old = xa_old - yb_old
n_bad_old = int((err_old < -18).sum())
err_new = xa_new - yb_new
n_bad_new = int((err_new < -18).sum())
print(f"\npoints with err<-18bpm: old={n_bad_old}  new={n_bad_new}")

with open("/tmp/final_compare_strict_mask.pkl","wb") as f:
    pickle.dump(dict(xa_old=xa_old, yb_old=yb_old, ta_old=ta_old, s_old=s_old,
                      xa_new=xa_new, yb_new=yb_new, ta_new=ta_new, s_new=s_new,
                      det_tmid=det_tmid, det_hr_sm=det_hr_sm,
                      det_v_old=det_v_old, det_v_strict=det_v_strict), f)
print("saved")
