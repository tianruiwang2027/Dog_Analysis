import pickle, datetime
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
ts_s = d["ts_s"]; xf_s = d["xf_s"]

THR = 20.0
UTC = datetime.timezone.utc
def T(y,mo,day,h,mi,s,us=500178):
    return datetime.datetime(y,mo,day,h,mi,s,us,tzinfo=UTC).timestamp()*1e6

# (time_center_us, rms, hand_label, category, note)
works = [
    (T(2026,6,26,17,28,19), 3.2,  "good", "TN", "clean baseline + 2 pulses -> correctly kept"),
    (T(2026,6,26,17,57,0),  5.4,  "good", "TN", "clean baseline + 2 pulses -> correctly kept"),
    (T(2026,6,26,17,44,2),  89.9, "bad",  "TP", "large motion artifact -> correctly flagged bad"),
]
fails = [
    (T(2026,6,26,17,26,58), 11.3, "bad", "FN", "brief burst buried in an otherwise flat 3s window\n-> averaged RMS stays low -> WRONGLY kept as good"),
    (T(2026,6,26,17,31,7),  7.7,  "bad", "FN", "same failure mode: short artifact, long flat window\n-> averaged RMS stays low -> WRONGLY kept as good"),
    (T(2026,6,26,17,47,11), 20.8, "good","FP", "brief strong-but-real heartbeat -> WRONGLY flagged bad"),
]

fig, axs = plt.subplots(2,3, figsize=(15,9), sharey=False)

def plot_one(ax, tc, rms, hand, cat, note):
    i0 = np.searchsorted(ts_s, tc-1.5e6); i1 = np.searchsorted(ts_s, tc+1.5e6)
    seg = xf_s[i0:i1]
    color = "tab:green" if cat in ("TN",) else ("tab:red" if cat in ("TP",) else "tab:orange")
    ax.plot((ts_s[i0:i1]-tc)/1e3, seg, color=color, lw=0.8)
    pred = "BAD" if rms>=THR else "GOOD"
    correct = "CORRECT" if ((pred=="BAD")==(hand=="bad")) else "WRONG"
    tstr = datetime.datetime.utcfromtimestamp(tc/1e6).strftime("%H:%M:%S")
    ax.set_title(f"{tstr}  RMS={rms:.1f}\nhand={hand.upper()}  mask={pred}  [{correct}]\n{note}", fontsize=9)
    ax.set_xlabel("ms")
    ax.axhline(0, color="gray", lw=0.5)

for ax,(tc,rms,hand,cat,note) in zip(axs[0], works):
    plot_one(ax, tc, rms, hand, cat, note)
for ax,(tc,rms,hand,cat,note) in zip(axs[1], fails):
    plot_one(ax, tc, rms, hand, cat, note)

axs[0,0].set_ylabel("bandpassed (10-100Hz) amplitude", fontsize=9)
axs[1,0].set_ylabel("bandpassed (10-100Hz) amplitude", fontsize=9)

plt.tight_layout(rect=[0,0,1,0.94])
plt.subplots_adjust(hspace=0.75)

fig.text(0.5, 0.975, "TOP ROW: mask WORKS  (agrees with hand label)", ha="center", fontsize=13, fontweight="bold", color="tab:green")
fig.text(0.5, 0.47, "BOTTOM ROW: mask FAILS  (disagrees with hand label)", ha="center", fontsize=13, fontweight="bold", color="tab:red")
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_flatness_mask_examples.png", dpi=130)
print("saved")
