#!/usr/bin/env python3
"""Illustrate the SINGLE-PEAK-PER-CYCLE detector (no S1/S2 pairing requirement):
just the strongest heart-sound peak each refractory period, like an R-peak
detector, instead of insisting on finding both S1 and S2."""
import pickle, sqlite3
import numpy as np
from scipy import signal as sg
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import datetime

UTC = datetime.timezone.utc
THR = 0.3
REFRACT_S = 0.50

def dn(us):
    return mdates.date2num([datetime.datetime.fromtimestamp(t/1e6, tz=UTC) for t in np.atleast_1d(us)])

with open("/tmp/shannon_restricted_results_thr0.3_gap0.17-0.25.pkl", "rb") as f:
    d = pickle.load(f)
ts_s, x_s, xf_s, fs_s = d["ts_s"], d["x_s"], d["xf_s"], d["fs_s"]
ts_sd, env_sd, sharp_s, q995_s, fsd_s = d["ts_sd"], d["env_sd"], d["sharp_s"], d["q995_s"], d["fsd_s"]
s1_doublet, s2_doublet = d["s1_scg"], d["s2_scg"]

# single-peak detector: strongest candidate per refractory window, no pairing
cand_single, _ = sg.find_peaks(sharp_s, height=THR, distance=max(1, int(REFRACT_S*fsd_s)))
singles = ts_sd[cand_single]

con = sqlite3.connect("/tmp/annotation_chelten_scg2.db")
cur = con.cursor()
cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = '' ORDER BY timestamp")
h_scg_pk = np.array([r[0] for r in cur.fetchall()], dtype="int64")

w0 = np.datetime64("2026-06-26T17:29:10").astype("datetime64[us]").astype(int)
w1 = np.datetime64("2026-06-26T17:29:40").astype("datetime64[us]").astype(int)

mR = (ts_s >= w0) & (ts_s <= w1)
mD = (ts_sd >= w0) & (ts_sd <= w1)
singles_w = singles[(singles >= w0) & (singles <= w1)]
s1d_w = s1_doublet[(s1_doublet >= w0) & (s1_doublet <= w1)]
s2d_w = s2_doublet[(s2_doublet >= w0) & (s2_doublet <= w1)]
hand_w = h_scg_pk[(h_scg_pk >= w0) & (h_scg_pk <= w1)]

def env_at(ts_query, ts_arr, env_arr):
    idx = np.searchsorted(ts_arr, ts_query); idx = np.clip(idx, 0, len(env_arr)-1)
    return env_arr[idx]

fig, axs = plt.subplots(4, 1, figsize=(16, 11), sharex=True,
                         gridspec_kw={"height_ratios":[1,1,1.4,0.45]})

axs[0].plot(dn(ts_s[mR]), x_s[mR], color="0.4", lw=0.6)
axs[0].set_ylabel("raw SCG"); axs[0].set_title("1) Raw SCG signal")
axs[0].grid(True, alpha=0.3)

axs[1].plot(dn(ts_s[mR]), xf_s[mR], color="#1f77b4", lw=0.7)
axs[1].set_ylabel("bandpassed"); axs[1].set_title("2) Bandpassed 10-100Hz")
axs[1].grid(True, alpha=0.3)

axs[2].plot(dn(ts_sd[mD]), sharp_s[mD], color="#2ca02c", lw=1.2, label="sharpened envelope")
axs[2].axhline(THR, color="0.4", ls="--", lw=1, label=f"threshold ({THR})")
axs[2].scatter(dn(singles_w), env_at(singles_w, ts_sd, sharp_s), s=110, color="#17becf",
               marker="D", edgecolor="k", label=f"single-peak detector (refract={REFRACT_S}s)", zorder=6)
axs[2].scatter(dn(s1d_w), env_at(s1d_w, ts_sd, sharp_s), s=70, color="#d62728",
               marker="o", edgecolor="k", label="doublet detector: S1", zorder=5, alpha=0.85)
axs[2].scatter(dn(s2d_w), env_at(s2d_w, ts_sd, sharp_s), s=70, color="#9467bd",
               marker="^", edgecolor="k", label="doublet detector: S2", zorder=5, alpha=0.85)
axs[2].set_ylabel("sharpened\n(local q995 + exp)")
axs[2].set_title("3) Single-peak-per-cycle detector (cyan diamond) vs the strict S1/S2 doublet detector (red/purple)")
axs[2].legend(loc="upper right", fontsize=8, ncol=2)
axs[2].grid(True, alpha=0.3)

axs[3].vlines(dn(hand_w), 0, 1, color="#145214", lw=1.5)
axs[3].vlines(dn(singles_w), 0.35, 1, color="#17becf", lw=1.8, linestyle="-")
axs[3].vlines(dn(s1d_w), 0, 0.65, color="#d62728", lw=1.5, linestyle=":")
axs[3].set_ylim(0, 1); axs[3].set_yticks([0.15,0.5,0.85]); axs[3].set_yticklabels(["doublet S1","",  "single"])
axs[3].set_title("4) hand-clicked (green, full height) vs single-peak detector (cyan, top) vs doublet-detector S1 (red dotted, bottom)", fontsize=10)
axs[3].xaxis.set_major_formatter(DateFormatter("%H:%M:%S", tz=UTC))
axs[3].set_xlim(dn(w0)[0], dn(w1)[0])

fig.suptitle(f"[Chelten] SCG single-peak-per-cycle detector (thr={THR}, refract={REFRACT_S}s) -- 17:29:10-17:29:40", fontsize=13)
fig.tight_layout()
out = "/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_shannon_scg_singles_illustration.png"
fig.savefig(out, dpi=150)
print("->", out)
print(f"in window: {len(singles_w)} single-detector beats, {len(s1d_w)} doublet-detector cycles, {len(hand_w)} hand clicks")
