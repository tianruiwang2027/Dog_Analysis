import pickle, sqlite3
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/cnn_dataset.pkl","rb") as f:
    D = pickle.load(f)
t = D["t"]; label = D["label"]
with open("/tmp/cnn_fooling_negs.pkl","rb") as f:
    FN = pickle.load(f)
fool = FN["fool"]

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

pos_t = t[label==1]
fool_t = t[fool]

d_pos = np.array([np.min(np.abs(ecg_shift-tc)) for tc in pos_t])/1e3
d_fool = np.array([np.min(np.abs(ecg_shift-tc)) for tc in fool_t])/1e3

rng = np.random.default_rng(0)
t0,t1 = t.min(), t.max()
d_null = np.array([np.min(np.abs(ecg_shift-rng.uniform(t0,t1))) for _ in range(3000)])/1e3

thresholds = [50,100,150,200,250,300]
fig, ax = plt.subplots(figsize=(9,6))
for arr, name, color in [(d_pos,"known TRUE beats\n(matched SCG click)","tab:green"),
                           (d_fool,"the 'fooled' candidates\n(no SCG click nearby)","tab:red"),
                           (d_null,"random time points\n(pure chance baseline)","0.5")]:
    frac = [ (arr<=thr).mean()*100 for thr in thresholds]
    ax.plot(thresholds, frac, "o-", color=color, label=name, lw=2)
ax.set_xlabel("distance threshold to nearest ECG click (ms)")
ax.set_ylabel("% of points within that distance")
ax.set_title("Is 'close to an ECG click' meaningful evidence?\nGenuine confirmed beats sit at the SAME chance-level proximity to ECG as random time points --\nso ECG-proximity alone can't confirm or rule out these candidates")
ax.legend(fontsize=9.5, loc="lower right")
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_ecg_proximity_null_check.png", dpi=135)
print("saved")
for name, arr in [("known true", d_pos), ("fool", d_fool), ("null", d_null)]:
    print(f"{name}: within250ms={ (arr<=250).mean()*100:.1f}%  within100ms={(arr<=100).mean()*100:.1f}%")
