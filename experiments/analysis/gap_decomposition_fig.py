import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

labels = ["Full template-\nfiltered detector", "Clean-only,\ndetector's own timing", "Clean-only, but with\nhand-click timing substituted", "Gold standard\n(hand SCG vs hand ECG)"]
r_vals = [0.808, 0.831, 0.837, 0.897]
colors = ["tab:green", "tab:cyan", "tab:blue", "gray"]

fig, ax = plt.subplots(figsize=(9.5,5))
bars = ax.bar(labels, r_vals, color=colors, width=0.6)
for b,v in zip(bars, r_vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}", ha="center", fontsize=10)
ax.set_ylabel("Pearson r vs hand-clicked ECG heart rate")
ax.set_ylim(0,1.0)
ax.set_title("Decomposing the remaining gap to the gold standard")
ax.annotate("", xy=(1,0.845), xytext=(0,0.82), arrowprops=dict(arrowstyle="->",color="black"))
ax.text(0.5, 0.86, "removing\ndetection errors", ha="center", fontsize=8)
ax.annotate("", xy=(2,0.85), xytext=(1,0.845), arrowprops=dict(arrowstyle="->",color="black"))
ax.text(1.5, 0.87, "swap in human\ntiming (~0 effect)", ha="center", fontsize=8)
ax.annotate("", xy=(3,0.885), xytext=(2,0.85), arrowprops=dict(arrowstyle="->",color="red"))
ax.text(2.5, 0.87, "still\nunexplained", ha="center", fontsize=8, color="red")
plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_gap_decomposition.png", dpi=130)
print("saved")
