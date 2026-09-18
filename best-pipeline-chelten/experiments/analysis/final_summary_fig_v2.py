import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

labels = ["Hand SCG\nvs Hand ECG\n(gold standard)", "Baseline\nsingles detector", "+ wide two-lobe\ntemplate filter", "+ template filter\n+ search-back"]
r_vals = [0.897, 0.778, 0.808, 0.708]
colors = ["gray", "tab:blue", "tab:green", "tab:red"]

fig, ax = plt.subplots(figsize=(8,5))
bars = ax.bar(labels, r_vals, color=colors, width=0.6)
for b,v in zip(bars, r_vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}", ha="center", fontsize=11)
ax.set_ylabel("Pearson r vs hand-clicked ECG heart rate")
ax.set_ylim(0,1.0)
ax.set_title("SCG detection accuracy vs hand-clicked ECG (clock-lag corrected, +7.0s)")
ax.axhline(r_vals[0], color="gray", ls="--", lw=0.8, alpha=0.6)
plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_final_summary_v2.png", dpi=130)
print("saved")
