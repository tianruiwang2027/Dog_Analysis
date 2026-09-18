import pickle, datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open('/tmp/c3_doublet_check_results.pkl','rb') as f:
    d = pickle.load(f)
r = d['results']['c3']
ts_w = d['ts_w']; fs = d['fs']
xf = r['xf']; x = r['x']

def to_us(dt):
    return int(dt.replace(tzinfo=datetime.timezone.utc).timestamp()*1e6)

ecg_dts = [
    datetime.datetime(2026,6,26,17,56,3,633723),
    datetime.datetime(2026,6,26,17,56,4,917833),
    datetime.datetime(2026,6,26,17,56,6,106954),
    datetime.datetime(2026,6,26,17,56,6,885760),
    datetime.datetime(2026,6,26,17,56,7,596516),
]
ecg_us = [to_us(x_) for x_ in ecg_dts]

t0 = to_us(datetime.datetime(2026,6,26,17,56,3))
t1 = to_us(datetime.datetime(2026,6,26,17,56,8,2))
a = np.searchsorted(ts_w, t0); b = np.searchsorted(ts_w, t1)
t_rel = (ts_w[a:b]-t0)/1e6

fig, axs = plt.subplots(2,1, figsize=(16,7), sharex=True)
axs[0].plot(t_rel, x[a:b], lw=0.6, color="tab:green")
axs[0].set_title("Raw c3 (17:56:03 - 17:56:08)")
axs[0].set_ylim(-400,900)
axs[1].plot(t_rel, xf[a:b], lw=0.6, color="tab:green")
axs[1].set_title("Bandpassed 10-100Hz c3, with hand-clicked ECG R-times (red) and R+[30,550]ms search window (shaded)")

for e in ecg_us:
    if t0 <= e <= t1:
        tr = (e-t0)/1e6
        for ax in axs:
            ax.axvline(tr, color="red", lw=1.4)
        axs[1].axvspan(tr+0.03, tr+0.55, color="red", alpha=0.08)

axs[1].set_xlabel("seconds from 17:56:03.000")
plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_c3_ecg_aligned.png", dpi=130)
print("saved")
