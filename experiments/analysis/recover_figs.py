import pickle, datetime
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/cnn_recover_eval.pkl","rb") as f:
    R = pickle.load(f)
res_old = R["res_old"]; res_new = R["res_new"]
with open("/tmp/final_4way_compare.pkl","rb") as f:
    D = pickle.load(f)

# ---------- 1) updated comparison bar chart ----------
fig, ax = plt.subplots(figsize=(9,5.5))
names = ["CORAL\n(the filter)", "Per-beat confirm\n(no S2 recovery)", "Per-beat confirm\n+ S2 recovery"]
rs = [D["s_cor"]["r"], res_old["s"]["r"], res_new["s"]["r"]]
ns = [D["s_cor"]["n"], res_old["s"]["n"], res_new["s"]["n"]]
colors = ["tab:orange", "0.6", "tab:purple"]
bars = ax.bar(names, rs, color=colors, alpha=0.85, width=0.55)
for b, r, n in zip(bars, rs, ns):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f"r = {r:.3f}\nn = {n}", ha="center", fontsize=10)
ax.set_ylim(0, 1.05)
ax.set_ylabel("r (vs ECG hand-picked HR)")
ax.set_title("Recovering the missed S1 when the detector locks onto S2\n(fully automatic, no SCG hand labels)")
plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_s2_recovery_bar.png", dpi=140)
print("saved bar chart")

# ---------- 2) end-cluster zoom, before/after ----------
t0w = int(datetime.datetime(2026,6,26,17,53,11, tzinfo=datetime.timezone.utc).timestamp()*1e6)
t1w = int(datetime.datetime(2026,6,26,17,55,41, tzinfo=datetime.timezone.utc).timestamp()*1e6)

ecg_tmid = D["ecg_tmid"]; ecg_hr_sm = D["ecg_hr_sm"]; ecg_v = D["ecg_v"]
tref = t0w/1e6

fig2, axs = plt.subplots(2,1, figsize=(12,7.5), sharex=True, sharey=True)
for ax, res, label, color in [(axs[0], res_old, "before: no S2 recovery", "0.5"),
                                (axs[1], res_new, "after: S2 recovery", "tab:purple")]:
    esel = (ecg_tmid>=t0w-3e6)&(ecg_tmid<t1w+3e6)&ecg_v
    ax.plot((ecg_tmid[esel]-t0w)/1e6, ecg_hr_sm[esel], "-", lw=1.3, color="tab:blue", alpha=0.85, label="ECG hand-picked")
    sel = (res["ta"]>=t0w-3e6)&(res["ta"]<t1w+3e6)
    ax.plot((res["ta"][sel]-t0w)/1e6, res["xa"][sel], ".", ms=5, color=color, alpha=0.8, label=label)
    ax.set_ylabel("HR (bpm)")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(35,75)
axs[1].set_xlabel("time (s, relative to 17:53:11)")
fig2.suptitle("The circled end cluster, before and after S2 recovery")
plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_s2_recovery_zoom.png", dpi=140)
print("saved zoom")
