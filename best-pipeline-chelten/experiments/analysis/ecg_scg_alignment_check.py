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

with open("/tmp/final_4way_compare.pkl","rb") as f:
    Dc = pickle.load(f)
LAG_US = Dc.get("LAG_US", 7_000_000)   # the coarse clock-offset lag used everywhere else
ecg_shift = ecg_pk + LAG_US            # into SCG time frame

with open("/tmp/valid_regions.pkl","rb") as f:
    VR = pickle.load(f)
valid_ivs = VR["valid_ivs"]
def in_ivs(tc, ivs): return any(a<=tc<b for a,b in ivs)

scg_valid = np.array([tc for tc in scg_pk if in_ivs(tc, valid_ivs)])
print(f"SCG hand clicks total={len(scg_pk)}  inside valid_ivs={len(scg_valid)}")

# for each valid SCG click, find nearest ECG click (lag-shifted), within a generous +-500ms search
WIN_US = 500_000
offsets = []
for s in scg_valid:
    j = np.searchsorted(ecg_shift, s)
    cand = []
    if j>0: cand.append(ecg_shift[j-1])
    if j<len(ecg_shift): cand.append(ecg_shift[j])
    if not cand: continue
    best = min(cand, key=lambda e: abs(e-s))
    d = best - s   # signed: ECG_shifted - SCG.  If ECG leads SCG (ECG happens first), then...
    if abs(d) <= WIN_US:
        offsets.append(d)
offsets = np.array(offsets)/1e3  # ms
print(f"matched pairs within +-{WIN_US/1e3:.0f}ms: {len(offsets)} / {len(scg_valid)}")
print(f"offset (ECG_shifted - SCG) stats: mean={offsets.mean():.1f}ms  median={np.median(offsets):.1f}ms  std={offsets.std():.1f}ms")
print(f"  positive offset means ECG(shifted) click comes AFTER the SCG click; negative means ECG(shifted) click comes BEFORE (leads) the SCG click")
for lo,hi in [(-30,30),(-50,-0),(-100,-50),(-150,-100),(0,50),(50,100),(100,150)]:
    frac = ((offsets>=lo)&(offsets<hi)).mean()*100
    print(f"  offset in [{lo:+4d},{hi:+4d})ms: {frac:5.1f}%")

fig, ax = plt.subplots(figsize=(9,6))
ax.hist(offsets, bins=100, range=(-500,500), color="teal", alpha=0.8)
ax.axvline(0, color="0.3", lw=1)
ax.axvspan(-100,-50, color="tab:orange", alpha=0.15, label="expected: ECG leads SCG by 50-100ms")
ax.set_xlabel("offset = (ECG hand click + 7.0s) - (nearest SCG hand click)   [ms]")
ax.set_ylabel("count")
ax.set_title(f"Fine-scale ECG-SCG hand-click alignment (valid regions only, n={len(offsets)})\nmean={offsets.mean():.1f}ms  median={np.median(offsets):.1f}ms  std={offsets.std():.1f}ms")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_ecg_scg_alignment.png", dpi=140)
print("saved")
