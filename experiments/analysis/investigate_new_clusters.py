import pickle, datetime, sqlite3
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/final_4way_compare.pkl","rb") as f:
    D = pickle.load(f)
with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
LAG_US = D["LAG_US"]
ts_s = d["ts_s"]; xf_s = d["xf_s"]

def load_peaks(path):
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = '' ORDER BY timestamp")
    return np.array([r[0] for r in cur.fetchall()], dtype="int64")
ecg_pk = load_peaks("/tmp/annotation_chelten_ecg2.db")
scg_pk = load_peaks("/tmp/annotation_chelten_scg2.db")
ecg_on_scg = ecg_pk - LAG_US

UTC = datetime.timezone.utc
def fmt(us):
    return datetime.datetime.fromtimestamp(us/1e6, tz=UTC).strftime("%H:%M:%S.%f")[:-3]

def panel(ax, lo_dt, hi_dt, title):
    lo = int(lo_dt.timestamp()*1e6); hi = int(hi_dt.timestamp()*1e6)
    ilo = np.searchsorted(ts_s, lo); ihi = np.searchsorted(ts_s, hi)
    t_rel = (ts_s[ilo:ihi]-lo)/1e6
    ax.plot(t_rel, xf_s[ilo:ihi], color="tab:purple", lw=0.6)
    scg_ct = scg_pk[(scg_pk>=lo)&(scg_pk<=hi)]
    ecg_ct = ecg_on_scg[(ecg_on_scg>=lo)&(ecg_on_scg<=hi)]
    for t in scg_ct:
        ax.axvline((t-lo)/1e6, color="tab:green", lw=1.5, alpha=0.85)
    for t in ecg_ct:
        ax.axvline((t-lo)/1e6, color="black", lw=1.1, ls="--", alpha=0.75)
    ax.set_title(f"{title}\nn(SCG clicks)={len(scg_ct)}  n(ECG clicks)={len(ecg_ct)}", fontsize=9)
    ax.set_xlim(0,(hi-lo)/1e6)

clusters = [
    ("#1  min~5-6: 17:31:05-18 (err -8 to -14)", datetime.datetime(2026,6,26,17,31,3,tzinfo=UTC), datetime.datetime(2026,6,26,17,31,20,tzinfo=UTC)),
    ("#2  min~7: 17:33:33-40 (err +2 to +10)",    datetime.datetime(2026,6,26,17,33,31,tzinfo=UTC), datetime.datetime(2026,6,26,17,33,42,tzinfo=UTC)),
    ("#3  min~11: 17:37:47-53 (err -8 to -14)",   datetime.datetime(2026,6,26,17,37,44,tzinfo=UTC), datetime.datetime(2026,6,26,17,37,56,tzinfo=UTC)),
    ("#4  min~16.6: 17:43:12 single (err -26.8)", datetime.datetime(2026,6,26,17,43,9,tzinfo=UTC),  datetime.datetime(2026,6,26,17,43,16,tzinfo=UTC)),
    ("#5  min~22-24: 17:48:39-56 (err -8 to -21, worst part of big oval)", datetime.datetime(2026,6,26,17,48,36,tzinfo=UTC), datetime.datetime(2026,6,26,17,49,0,tzinfo=UTC)),
    ("#6  min~33: 17:59:44 single (err -13.6)",   datetime.datetime(2026,6,26,17,59,41,tzinfo=UTC), datetime.datetime(2026,6,26,17,59,48,tzinfo=UTC)),
]

fig, axs = plt.subplots(3,2, figsize=(16,11))
for ax,(title, lo_dt, hi_dt) in zip(axs.flat, clusters):
    panel(ax, lo_dt, hi_dt, title)
for ax in axs[-1]: ax.set_xlabel("seconds from window start")
fig.suptitle("The 6 circled residual clusters -- green=SCG click, black dashed=ECG click (shifted to SCG clock)", fontsize=12)
plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_new_clusters.png", dpi=130)
print("saved")
