import pickle, datetime, sqlite3
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/final_4way_compare.pkl","rb") as f:
    D = pickle.load(f)
with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)

LAG_US = D["LAG_US"]
ts_s = d["ts_s"]; x_s = d["x_s"]; xf_s = d["xf_s"]
ts_sd = d["ts_sd"]; sharp_s = d["sharp_s"]

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

examples = [
    ("EXTRA SCG click  (17:43:04-07, err=+20.1, SCG has 1 more click than ECG)",
     datetime.datetime(2026,6,26,17,43,2,500000,tzinfo=UTC), datetime.datetime(2026,6,26,17,43,8,500000,tzinfo=UTC)),
    ("MISSING SCG click  (17:26:55-56, err=-22.9, SCG has 3 fewer clicks than ECG)",
     datetime.datetime(2026,6,26,17,26,53,tzinfo=UTC), datetime.datetime(2026,6,26,17,26,59,tzinfo=UTC)),
]

fig, axs = plt.subplots(len(examples), 2, figsize=(15, 3.6*len(examples)))
if len(examples)==1: axs = axs.reshape(1,-1)

for row,(title, t_lo_dt, t_hi_dt) in enumerate(examples):
    lo = int(t_lo_dt.timestamp()*1e6); hi = int(t_hi_dt.timestamp()*1e6)

    ilo = np.searchsorted(ts_s, lo); ihi = np.searchsorted(ts_s, hi)
    t_s = (ts_s[ilo:ihi]-lo)/1e6

    ax = axs[row,0]
    ax.plot(t_s, xf_s[ilo:ihi], color="tab:purple", lw=0.6)
    for t in scg_pk[(scg_pk>=lo)&(scg_pk<=hi)]:
        ax.axvline((t-lo)/1e6, color="tab:green", lw=1.4, alpha=0.85)
    for t in ecg_on_scg[(ecg_on_scg>=lo)&(ecg_on_scg<=hi)]:
        ax.axvline((t-lo)/1e6, color="black", lw=1.0, ls="--", alpha=0.7)
    ax.set_title(f"{title}\nBANDPASSED SCG (green=SCG hand click, black dashed=ECG hand click, shifted to SCG clock)", fontsize=9)
    ax.set_xlim(0,(hi-lo)/1e6)

    ilo2 = np.searchsorted(ts_sd, lo); ihi2 = np.searchsorted(ts_sd, hi)
    t_s2 = (ts_sd[ilo2:ihi2]-lo)/1e6
    ax = axs[row,1]
    ax.plot(t_s2, sharp_s[ilo2:ihi2], color="tab:blue", lw=1.1)
    ax.axhline(0.3, color="gray", ls=":", lw=0.8)
    for t in scg_pk[(scg_pk>=lo)&(scg_pk<=hi)]:
        ax.axvline((t-lo)/1e6, color="tab:green", lw=1.4, alpha=0.85)
    for t in ecg_on_scg[(ecg_on_scg>=lo)&(ecg_on_scg<=hi)]:
        ax.axvline((t-lo)/1e6, color="black", lw=1.0, ls="--", alpha=0.7)
    ax.set_title("SHARPENED ENVELOPE, same window", fontsize=9)
    ax.set_xlim(0,(hi-lo)/1e6)

for ax in axs[-1]:
    ax.set_xlabel("seconds from window start")

fig.suptitle("Where hand-clicked SCG and hand-clicked ECG disagree on beat COUNT (not just timing)", fontsize=13)
plt.tight_layout()
out = "/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_scg_gold_mismatch.png"
plt.savefig(out, dpi=140)
print("->", out)
