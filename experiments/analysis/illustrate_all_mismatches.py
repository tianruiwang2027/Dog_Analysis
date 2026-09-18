import pickle, datetime
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/unmatched_clicks.pkl","rb") as f:
    U = pickle.load(f)
unmatched_scg = U["unmatched_scg"]; unmatched_ecg = U["unmatched_ecg"]; LAG_US = U["LAG_US"]

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
ts_s = d["ts_s"]; xf_s = d["xf_s"]

import sqlite3
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

# standalone unmatched SCG clicks (excluding the trailing cluster)
scg_standalone = [t for t in unmatched_scg if datetime.datetime.fromtimestamp(t/1e6,tz=UTC) < datetime.datetime(2026,6,26,17,50,0,tzinfo=UTC)]
scg_cluster = [t for t in unmatched_scg if t not in scg_standalone]
print(f"standalone unmatched SCG: {len(scg_standalone)}   trailing cluster: {len(scg_cluster)}")

def panel(ax, center_us, half_s, label):
    lo = center_us - int(half_s*1e6); hi = center_us + int(half_s*1e6)
    ilo = np.searchsorted(ts_s, lo); ihi = np.searchsorted(ts_s, hi)
    t_rel = (ts_s[ilo:ihi]-center_us)/1e6
    ax.plot(t_rel, xf_s[ilo:ihi], color="tab:purple", lw=0.7)
    for t in scg_pk[(scg_pk>=lo)&(scg_pk<=hi)]:
        ax.axvline((t-center_us)/1e6, color="tab:green", lw=1.6, alpha=0.85)
    for t in ecg_on_scg[(ecg_on_scg>=lo)&(ecg_on_scg<=hi)]:
        ax.axvline((t-center_us)/1e6, color="black", lw=1.1, ls="--", alpha=0.75)
    ax.axvline(0, color="red", lw=0.8, ls=":")
    ax.set_title(label, fontsize=9)
    ax.set_xlim(-half_s, half_s)

# === FIGURE 1: 4 standalone unmatched SCG + first 2 unmatched ECG ===
fig, axs = plt.subplots(2,3, figsize=(16,7.5))
items1 = [(t,"SCG") for t in scg_standalone] + [(t,"ECG") for t in unmatched_ecg[:2]]
for i,(t,kind) in enumerate(items1):
    ax = axs.flat[i]
    label = f"#{i+1}  UNMATCHED {kind} click  @ {fmt(t)}\n(green=SCG click, black dashed=ECG click on SCG clock)"
    panel(ax, t, 2.0, label)
axs.flat[-1].set_visible(False) if len(items1)<6 else None
for ax in axs[-1]: ax.set_xlabel("seconds from the unmatched click")
fig.suptitle("Unmatched clicks, part 1 of 2  (green tick with no nearby black dashed tick = unmatched SCG; vice versa = unmatched ECG)", fontsize=11)
plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_mismatches_part1.png", dpi=130)
print("saved part1")

# === FIGURE 2: remaining 5 unmatched ECG + grouped trailing SCG cluster ===
fig, axs = plt.subplots(2,3, figsize=(16,7.5))
items2 = [(t,"ECG") for t in unmatched_ecg[2:]]
for i,(t,kind) in enumerate(items2):
    ax = axs.flat[i]
    label = f"#{i+7}  UNMATCHED {kind} click  @ {fmt(t)}\n(green=SCG click, black dashed=ECG click on SCG clock)"
    panel(ax, t, 2.0, label)
# last panel: the trailing cluster, wide view
ax = axs.flat[len(items2)]
lo = min(scg_cluster)-1_000_000; hi = max(scg_cluster)+1_000_000
ilo = np.searchsorted(ts_s, lo); ihi = np.searchsorted(ts_s, hi)
center = (lo+hi)//2
t_rel = (ts_s[ilo:ihi]-center)/1e6
ax.plot(t_rel, xf_s[ilo:ihi], color="tab:purple", lw=0.6)
for t in scg_pk[(scg_pk>=lo)&(scg_pk<=hi)]:
    ax.axvline((t-center)/1e6, color="tab:green", lw=1.3, alpha=0.85)
for t in ecg_on_scg[(ecg_on_scg>=lo)&(ecg_on_scg<=hi)]:
    ax.axvline((t-center)/1e6, color="black", lw=1.1, ls="--", alpha=0.75)
ax.set_title(f"#12  TRAILING CLUSTER: {len(scg_cluster)} unmatched SCG clicks\n{fmt(min(scg_cluster))} - {fmt(max(scg_cluster))} (SCG keeps beating after ECG clicks stop)", fontsize=9)
ax.set_xlim(-(hi-center)/1e6, (hi-center)/1e6)
for i in range(len(items2)+1, 6): axs.flat[i].set_visible(False)
for ax in axs[-1]: ax.set_xlabel("seconds from window center")
fig.suptitle("Unmatched clicks, part 2 of 2", fontsize=11)
plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_mismatches_part2.png", dpi=130)
print("saved part2")
