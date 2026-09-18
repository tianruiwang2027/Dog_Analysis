import pickle, datetime
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

with open("/tmp/blastradius_illustration_data.pkl","rb") as f:
    B = pickle.load(f)
tmid, hr, hr_sm, scg_pk, lo, hi = B["tmid"], B["hr"], B["hr_sm"], B["scg_pk"], B["lo"], B["hi"]

UTC = datetime.timezone.utc
def dn(us):
    return mdates.date2num([datetime.datetime.fromtimestamp(t/1e6, tz=UTC) for t in np.atleast_1d(us)])

sel = (tmid>=lo)&(tmid<=hi)
idxs = np.where(sel)[0]

# the problem interval: the one with the huge RR gap (missed SCG beat -> instHR crashes)
rr_ms = np.diff(scg_pk)/1e3
bad_i = idxs[np.argmax(rr_ms[idxs])]
bad_t = tmid[bad_i]

WIN3S = int(3.0e6)   # smoothing half-window
WIN25S = int(2.5e6)  # the coarse exclusion half-window used before

fig, ax = plt.subplots(figsize=(13,6.5))

ax.plot(dn(tmid[idxs]), hr_sm[idxs], "o-", color="tab:green", ms=5, lw=1.3, label="3s-smoothed HR -- what actually gets compared to ECG")
ax.plot(dn(tmid[idxs]), hr[idxs], "x", color="tab:red", ms=9, mew=1.8, label="instantaneous HR (60/RR), one per beat")

# mark the bad interval
ax.axvline(dn(bad_t)[0], color="black", ls="-", lw=1.5)
ax.annotate(f"the ONE problem RR interval\n(missed SCG beat, RR={rr_ms[bad_i]:.0f}ms, instHR={hr[bad_i]:.1f})",
            (dn(bad_t)[0], hr[bad_i]), textcoords="offset points", xytext=(15,55), fontsize=9, color="black",
            arrowprops=dict(arrowstyle="->", lw=1))

# shade the +-3s smoothing contamination radius around the bad interval
ax.axvspan(dn(bad_t-WIN3S)[0], dn(bad_t+WIN3S)[0], color="tab:red", alpha=0.10,
           label="±3s smoothing radius -- every point in here has this bad value blended into its own average")

# mark which points get removed under each strategy
strict_removed = {bad_i}
coarse_removed = set(i for i in idxs if abs(tmid[i]-bad_t) <= WIN25S)

for i in idxs:
    y = hr_sm[i]
    if i in strict_removed:
        marker, color = "X", "black"
    elif i in coarse_removed:
        marker, color = "s", "tab:orange"
    else:
        marker, color = "o", "tab:green"
    # already plotted the green line above; just annotate removal category below axis
ax.scatter([dn(tmid[i])[0] for i in coarse_removed if i not in strict_removed],
           [hr_sm[i] for i in coarse_removed if i not in strict_removed],
           s=140, facecolors="none", edgecolors="tab:orange", linewidths=2.2, zorder=5,
           label="removed by the ±2.5s COARSE window check\n(still contaminated, even though not the bad interval itself)")
ax.scatter([dn(bad_t)[0]], [hr_sm[bad_i]], s=140, marker="X", color="black", zorder=6,
           label="removed by the STRICT single-interval check\n(only this exact point)")

ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=UTC))
ax.set_ylabel("HR (bpm)")
ax.set_title("Why a wide exclusion window recovers more r than removing just the one bad interval\n(one real SCG dropout near 17:26:57 -- true ECG rate stayed ~80-90bpm the whole time)")
ax.legend(loc="upper left", fontsize=8.5)
plt.tight_layout()
out = "/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_blastradius_mechanism.png"
plt.savefig(out, dpi=140)
print("->", out)

print("\npoints and their status:")
for i in idxs:
    tag = "STRICT-removed" if i in strict_removed else ("coarse-removed" if i in coarse_removed else "kept by both")
    print(f"  {tag:16s}  t={dn(tmid[i])}  instHR={hr[i]:.1f}  smoothedHR={hr_sm[i]:.1f}")
