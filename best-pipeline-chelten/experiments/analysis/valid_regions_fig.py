import pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/recompute_with_valid_regions.pkl","rb") as f:
    R = pickle.load(f)

names = list(R.keys())
old_r = [R[n]["old"]["r"] for n in names]
new_r = [R[n]["new"]["r"] for n in names]
old_n = [R[n]["old"]["n"] for n in names]
new_n = [R[n]["new"]["n"] for n in names]

x = np.arange(len(names))
w = 0.35
fig, ax = plt.subplots(figsize=(9,5.5))
b1 = ax.bar(x-w/2, old_r, w, label="OLD restriction\n(hand-good-both)", color="tab:gray")
b2 = ax.bar(x+w/2, new_r, w, label="NEW restriction\n(valid regions)", color="tab:blue")
for i in range(len(names)):
    ax.text(x[i]-w/2, old_r[i]+0.012, f"{old_r[i]:.3f}\nn={old_n[i]}", ha="center", fontsize=8)
    ax.text(x[i]+w/2, new_r[i]+0.012, f"{new_r[i]:.3f}\nn={new_n[i]}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(names)
ax.set_ylabel("r  (vs ECG smoothed HR)")
ax.set_ylim(0,1.05)
ax.set_title("Standardizing all comparisons on the 'valid regions' gold standard\n(hand-good-both, minus known mismatches, minus bad-interval-edge contamination)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_valid_regions_update.png", dpi=130)
print("saved")
