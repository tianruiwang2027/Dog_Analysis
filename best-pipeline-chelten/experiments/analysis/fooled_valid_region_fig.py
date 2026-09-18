import pickle, sqlite3, datetime
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

with open("/tmp/valid_regions.pkl","rb") as f:
    VR = pickle.load(f)
valid_ivs = VR["valid_ivs"]

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

WIN_S = 3.0
fig, axs = plt.subplots(5, 2, figsize=(18, 15))
for k, idx in enumerate(picks_f):
    ax = axs.flat[k]
    tc = t[idx]; sc = prob_all[idx]
    d_ecg = np.min(np.abs(ecg_shift-tc))/1e3
    sel = (ts_s>=tc-WIN_S*1e6)&(ts_s<=tc+WIN_S*1e6)
    ax.plot((ts_s[sel]-tc)/1e6, xf_s[sel], color="black", lw=0.5, zorder=3)
    # shade any part of the visible window that is inside a "valid region"
    for a,b in valid_ivs:
        if b < tc-WIN_S*1e6 or a > tc+WIN_S*1e6: continue
        ax.axvspan((max(a,tc-WIN_S*1e6)-tc)/1e6, (min(b,tc+WIN_S*1e6)-tc)/1e6, color="tab:green", alpha=0.12, zorder=0)
    for c in scg_pk[(scg_pk>=tc-WIN_S*1e6)&(scg_pk<=tc+WIN_S*1e6)]:
        ax.axvline((c-tc)/1e6, color="green", lw=1.4, alpha=0.8, ymin=0.55, ymax=1.0)
    for c in ecg_shift[(ecg_shift>=tc-WIN_S*1e6)&(ecg_shift<=tc+WIN_S*1e6)]:
        ax.axvline((c-tc)/1e6, color="tab:blue", lw=1.4, alpha=0.8, ymin=0.0, ymax=0.45, ls="--")
    ax.axvline(0, color="red", lw=1.6, alpha=0.85, zorder=4)
    ax.set_title(f"score={sc:.2f}  nearest ECG click={d_ecg:.0f}ms  |  candidate is OUTSIDE valid_ivs (no green shading at t=0)", fontsize=8.5)
    ax.set_xlabel("s", fontsize=8)
    ax.tick_params(labelsize=7)

fig.suptitle("Red=candidate, green(top)=SCG hand clicks, blue dashed(bottom)=ECG hand clicks, green shading=valid_ivs\nAll 10 examples sit OUTSIDE valid_ivs -> per the region-aware logic, judge these against ECG only", fontsize=11, y=1.0)
plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_fooled_valid_region.png", dpi=125)
print("saved")
