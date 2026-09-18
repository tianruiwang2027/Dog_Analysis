import pickle, sqlite3, datetime
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
ts_s = d["ts_s"]; xf_s = d["xf_s"]; ts_sd = d["ts_sd"]; env_sd = d["env_sd"]

with open("/tmp/s2_chain_final.pkl","rb") as f:
    F = pickle.load(f)
merged = F["merged_arr"]

with open("/tmp/full_session_scores_all_models.pkl","rb") as f:
    S = pickle.load(f)
cand_t_all = S["cand_t"]  # every primary-detector candidate, whole session (pre-any-classifier)

with open("/tmp/cnn_mask_apply.pkl","rb") as f:
    CA = pickle.load(f)
LAG_US = CA["LAG_US"]

def load_peaks(path):
    con = sqlite3.connect(path); cur = con.cursor()
    cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = ''")
    return np.sort(np.array([r[0] for r in cur.fetchall()], dtype="int64"))
scg_pk = load_peaks("/tmp/annotation_chelten_scg2.db")
ecg_pk = load_peaks("/tmp/annotation_chelten_ecg2.db")
ecg_shift = ecg_pk - LAG_US

base = datetime.datetime.utcfromtimestamp(ecg_pk[0]/1e6).date()
def tsec(hh,mm,ss,ms=0):
    return datetime.datetime(base.year,base.month,base.day,hh,mm,ss,ms*1000, tzinfo=datetime.timezone.utc).timestamp()*1e6

windows = [
    ("17:31:09-19  (transient tachycardia burst)", tsec(17,31,8), tsec(17,31,20)),
    ("17:43:19-25  (transient tachycardia burst)",  tsec(17,43,18), tsec(17,43,26)),
    ("17:45:11-20  (HR decelerating from ~120bpm)", tsec(17,45,10), tsec(17,45,20)),
    ("17:59:35-46  (transient tachycardia burst)",  tsec(17,59,34), tsec(17,59,47)),
]

fig, axs = plt.subplots(4,1, figsize=(16, 16))
for row,(label_, w0, w1) in enumerate(windows):
    ax = axs[row]
    sel = (ts_sd>=w0)&(ts_sd<w1)
    ax.plot((ts_sd[sel]-w0)/1e6, env_sd[sel], color="black", lw=0.7, zorder=2)
    ymax_local = env_sd[sel].max() if sel.sum() else 1.0
    for c in cand_t_all[(cand_t_all>=w0)&(cand_t_all<w1)]:
        ax.axvline((c-w0)/1e6, color="lightgray", lw=1.0, alpha=0.9, ymin=0.0, ymax=0.18)
    for c in merged[(merged>=w0)&(merged<w1)]:
        ax.axvline((c-w0)/1e6, color="tab:purple", lw=1.5, alpha=0.85, ymin=0.2, ymax=0.5)
    for c in scg_pk[(scg_pk>=w0)&(scg_pk<w1)]:
        ax.axvline((c-w0)/1e6, color="green", lw=1.1, alpha=0.7, ymin=0.52, ymax=0.75)
    for c in ecg_shift[(ecg_shift>=w0)&(ecg_shift<w1)]:
        ax.axvline((c-w0)/1e6, color="tab:blue", lw=1.1, alpha=0.7, ymin=0.78, ymax=1.0, ls="--")
    ax.set_title(f"{label_}   (gray=all primary-detector candidates, purple=3-step-chain confirmed beats,\n"
                 f"green=SCG hand click, blue dashed=ECG hand click)", fontsize=9)
    ax.set_xlabel("s")

plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_tachy_outlier_diag.png", dpi=120)
print("saved")

print("\n--- numeric summary per window ---")
for label_, w0, w1 in windows:
    n_ecg = ((ecg_shift>=w0)&(ecg_shift<w1)).sum()
    n_scg_click = ((scg_pk>=w0)&(scg_pk<w1)).sum()
    n_cand = ((cand_t_all>=w0)&(cand_t_all<w1)).sum()
    n_conf = ((merged>=w0)&(merged<w1)).sum()
    dur_s = (w1-w0)/1e6
    ecg_in = ecg_shift[(ecg_shift>=w0)&(ecg_shift<w1)]
    min_rr = np.diff(ecg_in).min()/1000.0 if len(ecg_in)>1 else float("nan")
    print(f"{label_:45s} n_ecg={n_ecg:3d} (min RR={min_rr:.0f}ms -> {60000/min_rr:.0f}bpm peak)  "
          f"n_scg_click={n_scg_click:3d}  n_primary_candidates={n_cand:3d}  n_chain_confirmed={n_conf:3d}")
