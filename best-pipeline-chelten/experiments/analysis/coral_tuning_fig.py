import pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/coral_tuned_compare.pkl","rb") as f:
    R = pickle.load(f)

order = ["orig","delta","epszeta","windows","beta","tuned"]
labels = ["ORIGINAL\n(d=10,e=.005,z=5)","delta only\n(10->100)","eps/zeta only\n(loosened)",
          "windows only\n(drop 0.5s)","beta only\n(30->60)","FULL TUNED\n(mentor's set)"]
colors = ["0.5","tab:blue","tab:purple","tab:cyan","tab:brown","tab:red"]

cov = [R[k]["coverage"] for k in order]
r_vals = [R[k]["s"]["r"] for k in order]
mae_vals = [R[k]["s"]["mae"] for k in order]
bias_vals = [R[k]["s"]["bias"] for k in order]

fig, axs = plt.subplots(1, 3, figsize=(16,5.2))

ax = axs[0]
bars = ax.bar(labels, cov, color=colors)
ax.set_ylabel("coverage: % of hops with SQI≥0.10")
ax.set_title("Coverage")
ax.tick_params(axis='x', labelsize=8, rotation=20)
for b,v in zip(bars,cov): ax.annotate(f"{v:.1f}%", (b.get_x()+b.get_width()/2, v), ha="center", va="bottom", fontsize=8)

ax = axs[1]
bars = ax.bar(labels, r_vals, color=colors)
ax.set_ylabel("r vs hand ECG")
ax.set_title("Accuracy (r)")
ax.axhline(0.897, color="black", ls=":", lw=1, label="hand-SCG gold standard (0.897)")
ax.axhline(0.822, color="green", ls=":", lw=1, label="our best pipeline (0.822)")
ax.legend(fontsize=7, loc="lower right")
ax.tick_params(axis='x', labelsize=8, rotation=20)
for b,v in zip(bars,r_vals): ax.annotate(f"{v:.3f}", (b.get_x()+b.get_width()/2, v), ha="center", va="bottom", fontsize=8)

ax = axs[2]
bars = ax.bar(labels, mae_vals, color=colors)
ax.set_ylabel("MAE vs hand ECG (bpm)")
ax.set_title("Error magnitude (MAE)")
ax.tick_params(axis='x', labelsize=8, rotation=20)
for b,v in zip(bars,mae_vals): ax.annotate(f"{v:.2f}", (b.get_x()+b.get_width()/2, v), ha="center", va="bottom", fontsize=8)

fig.suptitle("[Chelten] Effect of CORAL parameter tuning: coverage rises, but accuracy vs. ECG gets WORSE", fontsize=13)
plt.tight_layout()
out = "/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_coral_tuning_ablation.png"
plt.savefig(out, dpi=140)
print("->", out)
