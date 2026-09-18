import pickle, sqlite3, datetime as dt
import numpy as np
from scipy import signal as sg
from scipy.ndimage import median_filter, percentile_filter

# ---------- v2: the earlier raw-signal despike barely moved anything, because
# sharpen_local's local-percentile normalization is already close to scale-invariant
# to a uniform rescale of env (which is all despiking the raw signal really did here --
# the true dominant global outlier lives elsewhere in the session, not inside these
# windows). The actual suppression mechanism is more specific: the 8-second sliding
# window used to compute the LOCAL q995 threshold (sharpen_local) straddles the motion
# spike itself, so its own huge Shannon-energy value drags the local q995 UP for ~8s
# around it -- which is exactly the denominator that gates every real beat's sharp_s
# score in that stretch. Fix: compute q995 from a version of env with the motion-energy
# outliers clipped out FIRST, but keep the numerator (env) as-is, so genuine beat energy
# is never touched -- only the threshold that's supposed to represent "normal" local
# background gets protected from being contaminated by the spike itself. ----------

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
ts_sd = d["ts_sd"]; env_sd = d["env_sd"]; fsd_s = d["fsd_s"]; t0 = int(d["t0"]); t1 = int(d["t1"])
sharp_s_orig = d["sharp_s"]; q995_s_orig = d["q995_s"]

def sharpen_local_v2(env, fsd, q995_win_s=8.0, q=99.5, clip_win_s=0.5, clip_K=6.0):
    # despike the env signal itself (short window Hampel clip) before computing the
    # local q995 threshold from it
    win_c = max(3, int(clip_win_s*fsd)) | 1
    med = median_filter(env, size=win_c, mode="nearest")
    mad = median_filter(np.abs(env-med), size=win_c, mode="nearest") + 1e-12
    env_for_q995 = np.minimum(env, med + clip_K*1.4826*mad)
    win = max(3, int(q995_win_s*fsd)) | 1
    local_q995 = percentile_filter(env_for_q995, q, size=win, mode="nearest")
    return np.exp(env / np.maximum(local_q995, 1e-9)) - 1.0, local_q995

sharp_s2, q995_s2 = sharpen_local_v2(env_sd, fsd_s)

PRIMARY_THR = 0.3; PRIMARY_REFRACT_S = 0.50
pk_idx2, _ = sg.find_peaks(sharp_s2, height=PRIMARY_THR, distance=max(1,int(PRIMARY_REFRACT_S*fsd_s)))
cand_t2 = ts_sd[pk_idx2]; cand_t2 = cand_t2[(cand_t2>=t0)&(cand_t2<=t1)]
pk_idx1, _ = sg.find_peaks(sharp_s_orig, height=PRIMARY_THR, distance=max(1,int(PRIMARY_REFRACT_S*fsd_s)))
cand_t1 = ts_sd[pk_idx1]; cand_t1 = cand_t1[(cand_t1>=t0)&(cand_t1<=t1)]
print(f"n candidates BEFORE: {len(cand_t1)}   AFTER (v2 despiked q995): {len(cand_t2)}")

with open("/tmp/despike_v2_result.pkl","wb") as f:
    pickle.dump(dict(ts_sd=ts_sd, sharp_s2=sharp_s2, q995_s2=q995_s2, cand_t2=cand_t2, cand_t1=cand_t1), f)

def load_peaks(path):
    con = sqlite3.connect(path); cur = con.cursor()
    cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = ''")
    return np.sort(np.array([r[0] for r in cur.fetchall()], dtype="int64"))
ecg_pk = load_peaks("/tmp/annotation_chelten_ecg2.db")
with open("/tmp/cnn_mask_apply.pkl","rb") as f:
    CA = pickle.load(f)
LAG_US = CA["LAG_US"]
ecg_shift = ecg_pk - LAG_US

base = dt.datetime.utcfromtimestamp(ts_sd[0]/1e6).date()
def tsec(hh,mm,ss):
    return int(dt.datetime(base.year,base.month,base.day,hh,mm,ss, tzinfo=dt.timezone.utc).timestamp()*1e6)

windows = [
    ("17:31:09-19", tsec(17,31,8), tsec(17,31,20)),
    ("17:43:19-25", tsec(17,43,18), tsec(17,43,26)),
    ("17:45:11-20", tsec(17,45,10), tsec(17,45,20)),
    ("17:59:35-46", tsec(17,59,34), tsec(17,59,47)),
]
print("\n--- candidate count in the 4 motion/tachycardia windows ---")
for label, w0, w1 in windows:
    n_ecg = ((ecg_shift>=w0)&(ecg_shift<w1)).sum()
    n_before = ((cand_t1>=w0)&(cand_t1<w1)).sum()
    n_after = ((cand_t2>=w0)&(cand_t2<w1)).sum()
    print(f"{label}: n_ecg={n_ecg:3d}   n_candidates_BEFORE={n_before:3d}   n_candidates_AFTER_v2={n_after:3d}")

print(f"\nwhole-session candidate count: before={len(cand_t1)}  after={len(cand_t2)}  (delta={len(cand_t2)-len(cand_t1)})")
