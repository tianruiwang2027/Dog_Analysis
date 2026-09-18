import pickle, sqlite3, datetime
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
ts_s = d["ts_s"]; xf_s = d["xf_s"]
with open("/tmp/cnn_dataset.pkl","rb") as f:
    D = pickle.load(f)
t = D["t"]; label = D["label"]; snips = D["snips"]; HALF_N = D["HALF_N"]; fsd_s = D["fsd_s"]

def load_peaks(path):
    con = sqlite3.connect(path); cur = con.cursor()
    cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = ''")
    return np.array([r[0] for r in cur.fetchall()], dtype="int64")
scg_pk = load_peaks("/tmp/annotation_chelten_scg2.db")

# pick a clean ~6s window with a mix of matched (positive) and unmatched (negative) candidates
t0w = int(datetime.datetime(2026,6,26,17,31,20, tzinfo=datetime.timezone.utc).timestamp()*1e6)
t1w = t0w + 8_000_000

fig = plt.figure(figsize=(16,9))
gs = fig.add_gridspec(2,4, height_ratios=[1.3,1])
axTop = fig.add_subplot(gs[0,:])
sel = (ts_s>=t0w)&(ts_s<t1w)
axTop.plot((ts_s[sel]-t0w)/1e6, xf_s[sel], color="black", lw=0.7)
for c in scg_pk[(scg_pk>=t0w)&(scg_pk<t1w)]:
    axTop.axvline((c-t0w)/1e6, color="0.75", lw=6, alpha=0.5, zorder=0)
csel = np.where((t>=t0w)&(t<t1w))[0]
picks_for_snip = []
for i in csel:
    color = "tab:green" if label[i]==1 else "tab:red"
    axTop.axvline((t[i]-t0w)/1e6, color=color, lw=1.8, alpha=0.9)
    picks_for_snip.append(i)
axTop.set_title("Candidate generation + labeling: gray band = hand SCG click, green candidate = matched (label=1, 'beat'), "
                 "red candidate = unmatched within 150ms (label=0, 'not a beat')", fontsize=10)
axTop.set_xlabel("s")

# show 2 example extracted snippets: one positive, one negative, from this window
pos_ex = next(i for i in picks_for_snip if label[i]==1)
neg_ex = next((i for i in picks_for_snip if label[i]==0), None)
if neg_ex is None:
    neg_ex = np.where(label==0)[0][0]

x_ms = (np.arange(2*HALF_N+1)-HALF_N)/fsd_s*1000
ax1 = fig.add_subplot(gs[1,0:2])
ax1.plot(x_ms, snips[pos_ex], color="tab:green", lw=1.3)
ax1.set_title("extracted snippet, label=1 (±400ms envelope around a matched candidate)", fontsize=9)
ax1.set_xlabel("ms")

ax2 = fig.add_subplot(gs[1,2:4])
ax2.plot(x_ms, snips[neg_ex], color="tab:red", lw=1.3)
ax2.set_title("extracted snippet, label=0 (±400ms envelope around an unmatched candidate)", fontsize=9)
ax2.set_xlabel("ms")

plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_training_set_explainer.png", dpi=135)
print("saved")
print(f"total candidates={len(t)}  positive={int(label.sum())}  negative={int((1-label).sum())}")
