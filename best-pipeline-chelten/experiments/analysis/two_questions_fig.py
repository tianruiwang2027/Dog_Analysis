import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))

# panel 1: refractory sweep
ax = axs[0]
refracts = ["0.50s\n(120bpm cap)", "0.40s\n(150bpm cap)", "0.30s\n(200bpm cap)"]
rs = [0.8482, 0.8235, 0.8182]
ns = [1541, 1280, 1319]
colors = ["tab:green", "tab:orange", "tab:red"]
bars = ax.bar(refracts, rs, color=colors)
for b,r,n in zip(bars, rs, ns):
    ax.text(b.get_x()+b.get_width()/2, r+0.01, f"r={r:.3f}\nn={n}", ha="center", va="bottom", fontsize=9)
ax.set_ylim(0,1.0)
ax.set_ylabel("downstream r (SCG HR vs ECG HR)")
ax.set_title("Lowering the primary-detector refractory period\n(same well-validated main CNN scoring the new candidates)", fontsize=10)

# panel 2: despike-v2, candidate recovery in the 4 motion windows
ax2 = axs[1]
labels = ["17:31:09-19", "17:43:19-25", "17:45:11-20", "17:59:35-46"]
n_ecg = [19, 14, 17, 20]
n_before = [13, 8, 8, 9]
n_after = [14, 9, 8, 10]
x = np.arange(len(labels)); w = 0.27
ax2.bar(x-w, n_ecg, width=w, label="ECG beats (ground truth)", color="tab:blue")
ax2.bar(x, n_before, width=w, label="primary candidates (before)", color="tab:gray")
ax2.bar(x+w, n_after, width=w, label="primary candidates (despike-v2)", color="tab:purple")
ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8)
ax2.legend(fontsize=8)
ax2.set_title("Protecting the local threshold from the motion spike's own energy\n(recovers a few candidates, but downstream r barely moves: 0.8482->0.8486)", fontsize=10)
ax2.set_ylabel("count")

plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_despike_and_refractory.png", dpi=130)
print("saved")
