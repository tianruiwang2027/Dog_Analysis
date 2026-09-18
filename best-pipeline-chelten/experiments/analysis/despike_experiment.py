import pickle, sqlite3, datetime as dt
import numpy as np
from scipy import signal as sg
from scipy.ndimage import gaussian_filter1d, uniform_filter1d, median_filter, percentile_filter
import torch, torch.nn as nn

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
ts_s = d["ts_s"]; xf_s = d["xf_s"]; fs_s = d["fs_s"]; t0 = int(d["t0"]); t1 = int(d["t1"])

print(f"raw signal: n={len(xf_s)}  fs={fs_s}Hz  global max|x|={np.max(np.abs(xf_s)):.1f}  "
      f"median|x|={np.median(np.abs(xf_s)):.2f}  99th pct|x|={np.percentile(np.abs(xf_s),99):.1f}")

# ---------- despike: Hampel-style outlier clipping BEFORE the Shannon-energy pipeline.
# Rolling median + rolling MAD over a SHORT window (0.5s) -- short enough that a spike is
# only a tiny fraction of any window it appears in, so it barely moves the local median/MAD,
# unlike the original pipeline's global max-normalization and 8s percentile window, both of
# which get dragged around by a single huge sample. Anything more than K local-MADs from the
# local median gets clipped back down to that boundary (not deleted -- just capped in amplitude
# so it can't dominate the log-energy / percentile stages downstream). ----------
def despike(x, fs, win_s=0.5, K=8.0):
    win = max(3, int(win_s*fs)) | 1
    med = median_filter(x, size=win, mode="nearest")
    mad = median_filter(np.abs(x-med), size=win, mode="nearest") + 1e-9
    lo = med - K*1.4826*mad
    hi = med + K*1.4826*mad
    return np.clip(x, lo, hi)

xf_s_despiked = despike(xf_s, fs_s)
n_clipped = (xf_s_despiked != xf_s).sum()
print(f"despike: clipped {n_clipped}/{len(xf_s)} samples ({n_clipped/len(xf_s)*100:.3f}%)")
print(f"despiked signal: global max|x|={np.max(np.abs(xf_s_despiked)):.1f}")

def shannon_envelope_decimated(xf, ts, fs, avg_win_s=0.02, gauss_sigma_s=0.01, FSD=200.0):
    xn = xf / (np.max(np.abs(xf)) + 1e-12)
    se = -(xn**2) * np.log(xn**2 + 1e-9)
    se_avg = uniform_filter1d(se, max(1, int(avg_win_s*fs)))
    step = max(1, int(round(fs/FSD)))
    se_d = se_avg[::step]; ts_d = ts[::step]; fsd = fs/step
    se_smooth = gaussian_filter1d(se_d, sigma=max(1, gauss_sigma_s*fsd))
    return ts_d, se_smooth, fsd

def sharpen_local(env, fsd, q995_win_s=8.0, q=99.5):
    win = max(3, int(q995_win_s*fsd)) | 1
    local_q995 = percentile_filter(env, q, size=win, mode="nearest")
    return np.exp(env / np.maximum(local_q995, 1e-9)) - 1.0, local_q995

ts_sd2, env_sd2, fsd_s2 = shannon_envelope_decimated(xf_s_despiked, ts_s, fs_s)
sharp_s2, q995_s2 = sharpen_local(env_sd2, fsd_s2)

PRIMARY_THR = 0.3; PRIMARY_REFRACT_S = 0.50
pk_idx2, _ = sg.find_peaks(sharp_s2, height=PRIMARY_THR, distance=max(1,int(PRIMARY_REFRACT_S*fsd_s2)))
cand_t2 = ts_sd2[pk_idx2]; cand_t2 = cand_t2[(cand_t2>=t0)&(cand_t2<=t1)]
print(f"n candidates AFTER despike: {len(cand_t2)}")

with open("/tmp/despike_pipeline_result.pkl","wb") as f:
    pickle.dump(dict(ts_sd2=ts_sd2, env_sd2=env_sd2, sharp_s2=sharp_s2, q995_s2=q995_s2, fsd_s2=fsd_s2,
                      cand_t2=cand_t2, xf_s_despiked=xf_s_despiked), f)
print("saved /tmp/despike_pipeline_result.pkl")

# ---- compare candidate counts in the 4 previously-diagnosed windows ----
with open("/tmp/full_session_scores_all_models.pkl","rb") as f:
    S = pickle.load(f)
cand_t_orig = S["cand_t"]

def load_peaks(path):
    con = sqlite3.connect(path); cur = con.cursor()
    cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = ''")
    return np.sort(np.array([r[0] for r in cur.fetchall()], dtype="int64"))
ecg_pk = load_peaks("/tmp/annotation_chelten_ecg2.db")
with open("/tmp/cnn_mask_apply.pkl","rb") as f:
    CA = pickle.load(f)
LAG_US = CA["LAG_US"]
ecg_shift = ecg_pk - LAG_US

base = dt.datetime.utcfromtimestamp(ts_s[0]/1e6).date()
def tsec(hh,mm,ss):
    return int(dt.datetime(base.year,base.month,base.day,hh,mm,ss, tzinfo=dt.timezone.utc).timestamp()*1e6)

windows = [
    ("17:31:09-19", tsec(17,31,8), tsec(17,31,20)),
    ("17:43:19-25", tsec(17,43,18), tsec(17,43,26)),
    ("17:45:11-20", tsec(17,45,10), tsec(17,45,20)),
    ("17:59:35-46", tsec(17,59,34), tsec(17,59,47)),
]
print("\n--- candidate count in the 4 motion/tachycardia windows, before vs after despike ---")
for label, w0, w1 in windows:
    n_ecg = ((ecg_shift>=w0)&(ecg_shift<w1)).sum()
    n_before = ((cand_t_orig>=w0)&(cand_t_orig<w1)).sum()
    n_after = ((cand_t2>=w0)&(cand_t2<w1)).sum()
    print(f"{label}: n_ecg={n_ecg:3d}   n_primary_candidates_BEFORE={n_before:3d}   n_primary_candidates_AFTER={n_after:3d}")
