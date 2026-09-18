import pickle, sqlite3
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Gap-aware smoothing: if there's a genuine detection gap (no confirmed SCG
# beat for >= GAP_THRESH_S) anywhere inside a point's 3-second smoothing window, don't
# report an HR value for that point at all -- rather than averaging in whatever beats
# happen to exist on one side of the gap (the "borrowing from the later segment" effect
# that was making post-gap recovery periods look like real momentary readings). ----------

with open("/tmp/s2_chain_final.pkl","rb") as f:
    F = pickle.load(f)
merged = F["merged_arr"]

with open("/tmp/cnn_mask_apply.pkl","rb") as f:
    CA = pickle.load(f)
LAG_US = CA["LAG_US"]; RESTRICT = CA["RESTRICT"]

def load_peaks_and_good(path):
    con = sqlite3.connect(path); cur = con.cursor()
    cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = ''")
    pk = np.array([r[0] for r in cur.fetchall()], dtype="int64")
    cur.execute("SELECT DISTINCT timestamp, label FROM events WHERE label != ''")
    rows = cur.fetchall()
    ev_ts = np.array([r[0] for r in rows], dtype="int64"); ev_lab=[r[1].lower() for r in rows]
    return pk, ev_ts, ev_lab
def good_intervals(ev_ts, ev_lab, span_end):
    ivs=[]; state="bad"; cur_start=None
    for tt,l in zip(ev_ts, ev_lab):
        if "good" in l and "start" in l:
            if state!="good": cur_start=tt; state="good"
        elif "bad" in l and "start" in l:
            if state=="good": ivs.append((cur_start,tt)); state="bad"
        elif "bad" in l and "end" in l:
            if state!="good": cur_start=tt; state="good"
        elif "good" in l and "end" in l:
            if state=="good": ivs.append((cur_start,tt)); state="bad"
        elif "stop" in l:
            if state=="good": ivs.append((cur_start,tt)); state="bad"
    if state=="good": ivs.append((cur_start, span_end))
    return ivs
def good_mask(ts, ivs):
    g = np.zeros(len(ts), bool)
    for a,b in ivs: g |= (ts>=a)&(ts<b)
    return g
def beat_hr(peaks):
    rr = np.diff(peaks)/1e6; hr = 60.0/rr
    tmid = peaks[:-1] + np.diff(peaks)//2
    return tmid, hr, rr
def stats(a,b):
    dd=a-b
    return dict(n=int(len(a)), bias=float(dd.mean()), mae=float(np.abs(dd).mean()),
                r=float(np.corrcoef(a,b)[0,1]) if len(a)>2 else float("nan"))
def align(tsA, hrA, vA, tsB, hrB, vB, lag_us=0, win_us=2_000_000, restrict=None):
    bts, bhr = tsB[vB], hrB[vB]
    A_ = np.where(vA)[0]
    xa, yb, ta = [], [], []
    for i in A_:
        traw = tsA[i]
        if restrict is not None and not (restrict[0] <= traw <= restrict[1]): continue
        t = traw + lag_us
        sel = (bts>=t-win_us)&(bts<=t+win_us)
        if sel.sum()>=1:
            xa.append(hrA[i]); yb.append(bhr[sel].mean()); ta.append(traw)
    return np.array(xa), np.array(yb), np.array(ta, dtype="int64")

ecg_pk, ecg_ev_ts, ecg_ev_lab = load_peaks_and_good("/tmp/annotation_chelten_ecg2.db")
span_end = max(ecg_pk.max(), merged.max())
ecg_ivs = good_intervals(ecg_ev_ts, ecg_ev_lab, span_end)
ecg_tmid, ecg_hr, _ = beat_hr(ecg_pk)
ecg_hr_sm_orig = ecg_hr  # placeholder, real smoothing below
def smooth_plain(tmid, hr, win_s=3.0):
    win_us2 = int(win_s*1e6); out = np.empty(len(tmid))
    for i,t in enumerate(tmid):
        sel = (tmid>=t-win_us2)&(tmid<=t+win_us2); out[i]=hr[sel].mean()
    return out
ecg_hr_sm = smooth_plain(ecg_tmid, ecg_hr)
ecg_v_full = good_mask(ecg_tmid, ecg_ivs)

# ---- confirmed SCG beats, RR-sanity filtered exactly as before ----
tmid, hr_raw, rr = beat_hr(merged)
MAX_RR_S = 1.5
ok = rr <= MAX_RR_S
tmid_ok, hr_ok = tmid[ok], hr_raw[ok]

GAP_THRESH_S = 1.5   # same threshold that already marks an interval as "a detection failure, not a real beat gap"
WIN_S = 3.0
win_us = int(WIN_S*1e6)
gap_us = int(GAP_THRESH_S*1e6)

# all beat-to-beat gaps in the FULL (unfiltered) confirmed sequence, used only to detect
# "was there silence here" -- irrespective of whether that gap's own RR pair got excluded
all_gaps_start = merged[:-1]; all_gaps_end = merged[1:]; all_gaps_len = np.diff(merged)
has_gap = all_gaps_len >= gap_us
gap_windows = list(zip(all_gaps_start[has_gap], all_gaps_end[has_gap]))
print(f"n detection gaps (>= {GAP_THRESH_S}s of silence): {len(gap_windows)}")

def window_touches_a_gap(t):
    lo, hi = t-win_us, t+win_us
    for gs, ge in gap_windows:
        if ge >= lo and gs <= hi:   # gap overlaps the smoothing window
            return True
    return False

# OLD (plain) smoothing, for comparison
hr_sm_old = smooth_plain(tmid_ok, hr_ok, win_s=WIN_S)

# NEW: gap-aware smoothing -- NaN out any point whose window touches a detection gap
hr_sm_new = np.full(len(tmid_ok), np.nan)
n_suppressed = 0
for i, t in enumerate(tmid_ok):
    if window_touches_a_gap(t):
        n_suppressed += 1
        continue
    sel = (tmid_ok>=t-win_us)&(tmid_ok<=t+win_us)
    hr_sm_new[i] = hr_ok[sel].mean()
print(f"n points suppressed (window touches a detection gap): {n_suppressed} / {len(tmid_ok)}")

keep_new = ~np.isnan(hr_sm_new)
ecg_v_at_old = good_mask(tmid_ok, ecg_ivs)
ecg_v_at_new = good_mask(tmid_ok[keep_new], ecg_ivs)

xa_old, yb_old, ta_old = align(tmid_ok, hr_sm_old, ecg_v_at_old, ecg_tmid, ecg_hr_sm, ecg_v_full, lag_us=LAG_US, restrict=RESTRICT)
xa_new, yb_new, ta_new = align(tmid_ok[keep_new], hr_sm_new[keep_new], ecg_v_at_new, ecg_tmid, ecg_hr_sm, ecg_v_full, lag_us=LAG_US, restrict=RESTRICT)

s_old = stats(xa_old, yb_old)
s_new = stats(xa_new, yb_new)
print(f"\nBEFORE (plain smoothing, borrows across gaps):  r={s_old['r']:.4f}  n={s_old['n']}  MAE={s_old['mae']:.2f}")
print(f"AFTER  (gap-aware, suppressed near detection gaps): r={s_new['r']:.4f}  n={s_new['n']}  MAE={s_new['mae']:.2f}")

# check the circled high-HR cluster specifically
mask_old = yb_old > 90
mask_new = yb_new > 90
print(f"\ncircled cluster (ECG HR>90bpm): BEFORE n={mask_old.sum()}  AFTER n={mask_new.sum()}")

with open("/tmp/gap_aware_result.pkl","wb") as f:
    pickle.dump(dict(xa_old=xa_old, yb_old=yb_old, ta_old=ta_old, s_old=s_old,
                      xa_new=xa_new, yb_new=yb_new, ta_new=ta_new, s_new=s_new,
                      n_suppressed=n_suppressed, gap_windows=gap_windows), f)

# ---- figure: before/after scatter ----
fig, axs = plt.subplots(1,2, figsize=(12,5.5))
for ax, xa, yb, s, title in [(axs[0], xa_old, yb_old, s_old, "BEFORE: plain 3s smoothing\n(borrows across detection gaps)"),
                              (axs[1], xa_new, yb_new, s_new, "AFTER: gap-aware smoothing\n(suppressed near detection gaps)")]:
    lo_v, hi_v = min(xa.min(),yb.min()), max(xa.max(),yb.max())
    ax.scatter(yb, xa, s=6, alpha=0.35, color="tab:purple")
    ax.plot([lo_v,hi_v],[lo_v,hi_v], color="black", lw=1, ls="--")
    ax.set_xlabel("ECG HR (bpm)"); ax.set_ylabel("SCG HR (bpm)")
    ax.set_title(f"{title}\nr={s['r']:.3f}  n={s['n']}  MAE={s['mae']:.2f}bpm", fontsize=10)
plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_gap_aware_compare.png", dpi=130)
print("saved chelten_gap_aware_compare.png")
