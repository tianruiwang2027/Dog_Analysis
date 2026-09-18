import pickle, sqlite3, datetime
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
ts_s = d["ts_s"]; xf_s = d["xf_s"]

with open("/tmp/relabel_failure_points.pkl","rb") as f:
    F = pickle.load(f)
confirmed = F["confirmed"]

def load_peaks(path):
    con = sqlite3.connect(path); cur = con.cursor()
    cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = ''")
    return np.array([r[0] for r in cur.fetchall()], dtype="int64")
scg_pk = load_peaks("/tmp/annotation_chelten_scg2.db")
ecg_pk = load_peaks("/tmp/annotation_chelten_ecg2.db")
LAG_US = 7_000_000
ecg_shift = ecg_pk - LAG_US  # corrected sign

def tsec(hh,mm,ss,ms=0):
    return datetime.datetime(2026,6,26,hh,mm,ss,ms*1000, tzinfo=datetime.timezone.utc).timestamp()*1e6

windows = [
    ("17:43:27-45 (big underestimate, ~40bpm)", tsec(17,43,26), tsec(17,43,47)),
    ("17:54:05-39 (sustained underestimate)",    tsec(17,54,3),  tsec(17,54,41)),
    ("17:55:37-57 (sustained underestimate)",    tsec(17,55,35), tsec(17,55,59)),
    ("17:59:40-56 (end-of-session, mixed)",       tsec(17,59,38), tsec(17,59,58)),
]

fig, axs = plt.subplots(4,1, figsize=(16, 15))
for row,(label_, w0, w1) in enumerate(windows):
    ax = axs[row]
    sel = (ts_s>=w0)&(ts_s<w1)
    ax.plot((ts_s[sel]-w0)/1e6, xf_s[sel], color="black", lw=0.55, zorder=2)
    for c in confirmed[(confirmed>=w0)&(confirmed<w1)]:
        ax.axvline((c-w0)/1e6, color="tab:purple", lw=1.3, alpha=0.8, ymin=0.0, ymax=0.3)
    for c in scg_pk[(scg_pk>=w0)&(scg_pk<w1)]:
        ax.axvline((c-w0)/1e6, color="green", lw=1.1, alpha=0.7, ymin=0.35, ymax=0.65)
    for c in ecg_shift[(ecg_shift>=w0)&(ecg_shift<w1)]:
        ax.axvline((c-w0)/1e6, color="tab:blue", lw=1.1, alpha=0.7, ymin=0.7, ymax=1.0, ls="--")
    ax.set_title(f"{label_}   (purple=CNN-confirmed(cutoff=0.25), green=SCG hand click, blue dashed=ECG hand click [corrected align])", fontsize=9)
    ax.set_xlabel("s")

plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_relabel_failure_diag.png", dpi=120)
print("saved")

# quick numeric summary: how many ECG beats vs confirmed algo beats in each window
for label_, w0, w1 in windows:
    n_ecg = ((ecg_shift>=w0)&(ecg_shift<w1)).sum()
    n_scg = ((scg_pk>=w0)&(scg_pk<w1)).sum()
    n_conf = ((confirmed>=w0)&(confirmed<w1)).sum()
    dur_s = (w1-w0)/1e6
    print(f"{label_:45s} dur={dur_s:.0f}s  n_ecg={n_ecg:3d} (implied {n_ecg/dur_s*60:.0f}bpm)  n_scg_click={n_scg:3d}  n_confirmed={n_conf:3d} (implied {n_conf/dur_s*60:.0f}bpm)")
