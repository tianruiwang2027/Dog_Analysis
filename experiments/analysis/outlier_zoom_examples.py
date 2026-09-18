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

# find the "extra" (unmatched-to-SCG) confirmed beats within these clusters directly
def tsec(hh,mm,ss,ms=0):
    return datetime.datetime(2026,6,26,hh,mm,ss,ms*1000, tzinfo=datetime.timezone.utc).timestamp()*1e6

search_windows = [(tsec(17,35,45), tsec(17,35,56)), (tsec(17,36,14), tsec(17,36,24)),
                   (tsec(17,37,38), tsec(17,37,46)), (tsec(17,39,35), tsec(17,39,45)),
                   (tsec(17,40,42), tsec(17,40,50))]
extras = []
for w0,w1 in search_windows:
    csel = (confirmed_new>=w0)&(confirmed_new<w1)
    for c in confirmed_new[csel]:
        d_scg = np.min(np.abs(scg_pk-c)) if len(scg_pk) else 1e12
        if d_scg>150_000:
            extras.append(c)
extras = np.array(sorted(extras))
print(f"found {len(extras)} extra beats across these clusters; using first 4 for zoom")

picks = extras[:4]
fig, axs = plt.subplots(2,2, figsize=(15,9))
WIN_S = 1.2
for k, tc in enumerate(picks):
    ax = axs.flat[k]
    j = np.argmin(np.abs(cand_t-tc)); sc = cand_cnn[j]
    sel = (ts_s>=tc-WIN_S*1e6)&(ts_s<=tc+WIN_S*1e6)
    ax.plot((ts_s[sel]-tc)/1e6*1000, xf_s[sel], color="black", lw=0.9, zorder=2)
    # all confirmed beats in this window
    csel = (confirmed_new>=tc-WIN_S*1e6)&(confirmed_new<=tc+WIN_S*1e6)
    for c in confirmed_new[csel]:
        is_extra = abs(c-tc)<1000
        ax.axvline((c-tc)/1e6*1000, color="tab:orange" if is_extra else "tab:purple", lw=2 if is_extra else 1.4, alpha=0.9)
    for c in ecg_shift[(ecg_shift>=tc-WIN_S*1e6)&(ecg_shift<=tc+WIN_S*1e6)]:
        ax.axvline((c-tc)/1e6*1000, color="tab:blue", lw=1.6, ls="--", alpha=0.8)
    for c in scg_pk[(scg_pk>=tc-WIN_S*1e6)&(scg_pk<=tc+WIN_S*1e6)]:
        ax.axvline((c-tc)/1e6*1000, color="green", lw=1.6, alpha=0.8)
    tstr = datetime.datetime.utcfromtimestamp(tc/1e6).strftime("%H:%M:%S.%f")[:-3]
    ax.set_title(f"{tstr}  extra-beat CNN score={sc:.2f}\n(orange=the extra beat, purple=other confirmed beats, blue dashed=ECG, green=SCG hand click)", fontsize=9)
    ax.set_xlabel("ms")

plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_outlier_zoom_examples.png", dpi=135)
print("saved")
