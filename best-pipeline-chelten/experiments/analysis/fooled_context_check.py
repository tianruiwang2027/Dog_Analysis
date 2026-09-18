import pickle, sqlite3
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
ts_s = d["ts_s"]; xf_s = d["xf_s"]

with open("/tmp/cnn_dataset.pkl","rb") as f:
    D = pickle.load(f)
t = D["t"]
with open("/tmp/cnn_fooling_negs.pkl","rb") as f:
    FN = pickle.load(f)
fool = FN["fool"]; prob_all = FN["prob_all_orig"]
fool_idx = np.where(fool)[0]
order_f = np.argsort(-prob_all[fool_idx])
picks_f = fool_idx[order_f[:: max(1, len(order_f)//10)][:10]]

def load_peaks(path):
    con = sqlite3.connect(path); cur = con.cursor()
    cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = ''")
    return np.array([r[0] for r in cur.fetchall()], dtype="int64")
scg_pk = load_peaks("/tmp/annotation_chelten_scg2.db")
ecg_pk = load_peaks("/tmp/annotation_chelten_ecg2.db")
with open("/tmp/final_4way_compare.pkl","rb") as f:
    Dc = pickle.load(f)
LAG_US = Dc.get("LAG_US", 7_000_000)
ecg_pk_shifted = ecg_pk + LAG_US   # into SCG time frame

WIN_S = 3.0

fig, axs = plt.subplots(5, 2, figsize=(18, 15))
for k, idx in enumerate(picks_f):
    ax = axs.flat[k]
    tc = t[idx]; sc = prob_all[idx]
    sel = (ts_s>=tc-WIN_S*1e6)&(ts_s<=tc+WIN_S*1e6)
    ax.plot((ts_s[sel]-tc)/1e6, xf_s[sel], color="black", lw=0.5)
    for c in scg_pk[(scg_pk>=tc-WIN_S*1e6)&(scg_pk<=tc+WIN_S*1e6)]:
        ax.axvline((c-tc)/1e6, color="green", lw=1.4, alpha=0.8, ymin=0.55, ymax=1.0)
    for c in ecg_pk_shifted[(ecg_pk_shifted>=tc-WIN_S*1e6)&(ecg_pk_shifted<=tc+WIN_S*1e6)]:
        ax.axvline((c-tc)/1e6, color="tab:blue", lw=1.4, alpha=0.8, ymin=0.0, ymax=0.45, ls="--")
    ax.axvline(0, color="red", lw=1.6, alpha=0.85)
    ax.set_title(f"CNN score={sc:.2f}   (red=flagged candidate, green(top)=SCG hand clicks, blue dashed(bottom)=ECG hand clicks)", fontsize=8.5)
    ax.set_xlabel("s", fontsize=8)
    ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_fooled_context_check.png", dpi=125)
print("saved")
