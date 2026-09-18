import pickle, sqlite3
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_peaks(path):
    con = sqlite3.connect(path); cur = con.cursor()
    cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = ''")
    return np.array([r[0] for r in cur.fetchall()], dtype="int64")
scg_pk = load_peaks("/tmp/annotation_chelten_scg2.db")
ecg_pk = load_peaks("/tmp/annotation_chelten_ecg2.db")
with open("/tmp/valid_regions.pkl","rb") as f:
    VR = pickle.load(f)
valid_ivs = VR["valid_ivs"]
def in_ivs(tc, ivs): return any(a<=tc<b for a,b in ivs)
scg_valid = np.array([tc for tc in scg_pk if in_ivs(tc, valid_ivs)])

# CORRECTED sign, confirmed by brute-force search: ecg_shift = ecg_pk - 7.0s lands tightly on scg clicks
LAG_US = 7_000_000
ecg_shift = ecg_pk - LAG_US

WIN_US = 150_000
offs = []
for s in scg_valid:
    j = np.searchsorted(ecg_shift, s)
    cand=[]
    if j>0: cand.append(ecg_shift[j-1])
    if j<len(ecg_shift): cand.append(ecg_shift[j])
    if not cand: continue
    best = min(cand, key=lambda e: abs(e-s))
    d = best-s
    if abs(d)<=WIN_US:
        offs.append(d/1e3)
offs = np.array(offs)
print(f"n matched (within +-{WIN_US/1e3:.0f}ms, corrected sign) = {len(offs)} / {len(scg_valid)}  ({len(offs)/len(scg_valid)*100:.1f}%)")
print(f"offset = (ECG click - 7.0s) - (nearest SCG click):")
print(f"  mean={offs.mean():.1f}ms  median={np.median(offs):.1f}ms  std={offs.std():.1f}ms")
print(f"  interpretation: NEGATIVE means the ECG click (shifted) falls BEFORE the SCG click,")
print(f"  i.e. ECG leads SCG by |offset| ms. POSITIVE means ECG lags SCG.")

fig, ax = plt.subplots(figsize=(9,6))
ax.hist(offs, bins=60, range=(-150,150), color="teal", alpha=0.85)
ax.axvline(0, color="0.3", lw=1)
ax.axvspan(-100,-50, color="tab:orange", alpha=0.2, label="expected: ECG leads SCG by 50-100ms")
ax.axvline(np.median(offs), color="red", ls="--", lw=1.6, label=f"observed median={np.median(offs):+.0f}ms")
ax.set_xlabel("offset (ms):  negative = ECG leads SCG,  positive = ECG lags SCG")
ax.set_title(f"Fine ECG-SCG hand-click alignment, corrected sign\n(valid regions only, n={len(offs)}, global lag=7.000s)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_ecg_scg_alignment_fixed.png", dpi=140)
print("saved")
