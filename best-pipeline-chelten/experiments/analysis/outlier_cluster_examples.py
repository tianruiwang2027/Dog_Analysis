import pickle, sqlite3, datetime
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
ts_s = d["ts_s"]; xf_s = d["xf_s"]

with open("/tmp/cnn_recover_eval.pkl","rb") as f:
    R = pickle.load(f)
confirmed_new = R["confirmed_new"]

with open("/tmp/cnn_mask_apply.pkl","rb") as f:
    CA = pickle.load(f)
cand_t = CA["cand_t"]; cand_cnn = CA["cand_cnn"]

def load_peaks(path):
    con = sqlite3.connect(path); cur = con.cursor()
    cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = ''")
    return np.array([r[0] for r in cur.fetchall()], dtype="int64")
scg_pk = load_peaks("/tmp/annotation_chelten_scg2.db")
ecg_pk = load_peaks("/tmp/annotation_chelten_ecg2.db")

with open("/tmp/final_4way_compare.pkl","rb") as f:
    Dc = pickle.load(f)
LAG_US = Dc.get("LAG_US", 7_000_000)
ecg_shift = ecg_pk + LAG_US

with open("/tmp/valid_regions.pkl","rb") as f:
    VR = pickle.load(f)
valid_ivs = VR["valid_ivs"]
def in_ivs(tc, ivs): return any(a<=tc<b for a,b in ivs)

def tsec(hh,mm,ss,ms=0):
    return datetime.datetime(2026,6,26,hh,mm,ss,ms*1000, tzinfo=datetime.timezone.utc).timestamp()*1e6

windows = [
    ("17:35:47 sub-cluster", tsec(17,35,45), tsec(17,35,56)),
    ("17:36:15 sub-cluster", tsec(17,36,14), tsec(17,36,24)),
    ("17:37:39 sub-cluster", tsec(17,37,38), tsec(17,37,46)),
    ("17:39:36 sub-cluster", tsec(17,39,35), tsec(17,39,45)),
    ("17:40:43 sub-cluster", tsec(17,40,42), tsec(17,40,50)),
    ("17:55:22 sub-cluster", tsec(17,55,20), tsec(17,55,26)),
]

fig, axs = plt.subplots(6, 1, figsize=(15, 20))
for row,(label, w0, w1) in enumerate(windows):
    ax = axs[row]
    sel = (ts_s>=w0)&(ts_s<w1)
    ax.plot((ts_s[sel]-w0)/1e6, xf_s[sel], color="black", lw=0.6, zorder=2)

    # shade valid_ivs coverage
    for a,b in valid_ivs:
        if b<w0 or a>w1: continue
        ax.axvspan((max(a,w0)-w0)/1e6, (min(b,w1)-w0)/1e6, color="tab:green", alpha=0.08, zorder=0)

    # algorithm confirmed beats (purple), highlight the ones NOT matched to any SCG click within 150ms as the "extra" ones (orange)
    csel = (confirmed_new>=w0)&(confirmed_new<w1)
    for c in confirmed_new[csel]:
        d_scg = np.min(np.abs(scg_pk-c)) if len(scg_pk) else 1e12
        color = "tab:orange" if d_scg>150_000 else "tab:purple"
        j = np.argmin(np.abs(cand_t-c)); sc = cand_cnn[j]
        ax.axvline((c-w0)/1e6, color=color, lw=1.6, alpha=0.85, ymin=0.0, ymax=0.35, zorder=3)
        ax.text((c-w0)/1e6, ax.get_ylim()[0]*0.0, f"{sc:.2f}", fontsize=6.5, color=color, rotation=90, va="bottom", ha="center")

    # ECG hand clicks (blue dashed, lag-shifted)
    for c in ecg_shift[(ecg_shift>=w0)&(ecg_shift<w1)]:
        ax.axvline((c-w0)/1e6, color="tab:blue", lw=1.3, alpha=0.7, ymin=0.65, ymax=1.0, ls="--", zorder=3)

    # SCG hand clicks (green) -- shown since these windows fall in valid regions
    for c in scg_pk[(scg_pk>=w0)&(scg_pk<w1)]:
        ax.axvline((c-w0)/1e6, color="green", lw=1.3, alpha=0.7, ymin=0.35, ymax=0.65, zorder=3)

    ax.set_title(f"{label}   (purple=confirmed matches an SCG click, orange=confirmed but NO nearby SCG click -- the 'extra' beat, "
                 f"blue dashed=ECG click, green=SCG click, shaded=valid_ivs)", fontsize=8.5)
    ax.set_xlabel("s", fontsize=8)

plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_outlier_cluster_examples.png", dpi=120)
print("saved")
