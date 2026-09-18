import pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/mask_v7_window_sweep.pkl","rb") as f:
    D = pickle.load(f)
rows = D["rows"]; s_h = D["s_h"]

win = [r["WIN_S"] for r in rows]
acc = [r["acc"]*100 for r in rows]
sens = [r["sens"]*100 for r in rows]
spec = [r["spec"]*100 for r in rows]
beat_agree = [r["beat_agree"]*100 for r in rows]
rvals = [r["r"] for r in rows]
mae = [r["mae"] for r in rows]

fig, axs = plt.subplots(1,2, figsize=(13,5))

axs[0].plot(win, acc, "o-", color="black", label="accuracy")
axs[0].plot(win, sens, "o-", color="tab:red", label="sensitivity (bad caught)")
axs[0].plot(win, spec, "o-", color="tab:green", label="specificity (good kept)")
axs[0].plot(win, beat_agree, "o--", color="gray", label="beat-level agreement w/ hand")
axs[0].set_xlabel("window length (s)")
axs[0].set_ylabel("%")
axs[0].set_title("A. Classification performance vs window length\n(threshold re-calibrated at each window length)")
axs[0].legend(fontsize=8)
axs[0].set_ylim(45,100)

ax2 = axs[1]
ax2.plot(win, rvals, "o-", color="tab:blue", label="r (this mask)")
ax2.axhline(s_h["r"], color="black", ls="--", label=f"HAND baseline (r={s_h['r']:.3f})")
ax2.set_xlabel("window length (s)")
ax2.set_ylabel("r  (SCG smoothed-HR vs ECG smoothed-HR)", color="tab:blue")
ax2.set_ylim(0.5,1.0)
ax2.set_title("B. Effect on the gold-standard comparison vs window length\nr stays stuck ~0.69-0.73 at every window length")
ax2.legend(fontsize=8, loc="lower right")

plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_window_sweep.png", dpi=130)
print("saved")
