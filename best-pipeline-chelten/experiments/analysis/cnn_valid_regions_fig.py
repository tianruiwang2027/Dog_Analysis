import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

names = ["HAND\n(gold std)", "NCC template\n(balanced)", "NCC template\n(stricter)", "CNN mask\n(best)"]
full_r = [0.8959, 0.8324, 0.8448, 0.8711]
full_n = [1645, 1452, 1304, 1549]
valid_r = [0.9465, 0.9467, 0.9459, 0.9463]
valid_n = [1446, 1314, 1169, 1398]
colors = ["tab:gray","tab:orange","tab:orange","tab:purple"]

fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))

x = np.arange(len(names))
b = axs[0].bar(x, full_r, color=colors, alpha=0.85)
for i,rect in enumerate(b):
    axs[0].text(rect.get_x()+rect.get_width()/2, rect.get_height()+0.015, f"r={full_r[i]:.3f}\nn={full_n[i]}", ha="center", fontsize=8.5)
axs[0].set_xticks(x); axs[0].set_xticklabels(names)
axs[0].set_ylim(0,1.05); axs[0].set_ylabel("r (vs ECG smoothed HR)")
axs[0].set_title("A. Evaluated on FULL hand-good-both data\n(includes the known click-mismatches / bad-edge contamination)")

b2 = axs[1].bar(x, valid_r, color=colors, alpha=0.85)
for i,rect in enumerate(b2):
    axs[1].text(rect.get_x()+rect.get_width()/2, rect.get_height()+0.015, f"r={valid_r[i]:.3f}\nn={valid_n[i]}", ha="center", fontsize=8.5)
axs[1].set_xticks(x); axs[1].set_xticklabels(names)
axs[1].set_ylim(0,1.05); axs[1].set_ylabel("r (vs ECG smoothed HR)")
axs[1].set_title("B. Evaluated on the 'valid regions' standard\n(already excludes those same problem points)")

fig.suptitle("Correction: the CNN-vs-NCC gap only shows up on the messier full dataset --\n"
              "once restricted to valid regions (as instructed), all methods converge to the ~0.946-0.947 noise floor",
              fontsize=11)
plt.tight_layout(rect=[0,0,1,0.90])
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_cnn_valid_regions_corrected.png", dpi=130)
print("saved")
