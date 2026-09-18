#!/usr/bin/env python3
"""Illustrate the SCG Shannon-energy S1/S2 peak-finding pipeline stage by
stage, with detected peaks overlaid on the raw signal, for a representative
window."""
import pickle, sqlite3
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import datetime

UTC = datetime.timezone.utc

def dn(us):
    return mdates.date2num([datetime.datetime.fromtimestamp(t/1e6, tz=UTC) for t in np.atleast_1d(us)])

with open("/tmp/shannon_restricted_results.pkl","rb") as f:
    d = pickle.load(f)

ts_s, x_s, xf_s, fs_s = d["ts_s"], d["x_s"], d["xf_s"], d["fs_s"]
ts_sd, env_sd, sharp_s, q995_s, fsd_s = d["ts_sd"], d["env_sd"], d["sharp_s"], d["q995_s"], d["fsd_s"]
cand_scg, s1_scg, s2_scg = d["cand_scg"], d["s1_scg"], d["s2_scg"]

# hand-clicked SCG peaks, for reference
con = sqlite3.connect("/tmp/annotation_chelten_scg2.db")
cur = con.cursor()
cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = '' ORDER BY timestamp")
h_scg_pk = np.array([r[0] for r in cur.fetchall()], dtype="int64")

w0 = np.datetime64("2026-06-26T17:47:00").astype("datetime64[us]").astype(int)
w1 = np.datetime64("2026-06-26T17:47:16").astype("datetime64[us]").astype(int)

mR = (ts_s >= w0) & (ts_s <= w1)
mD = (ts_sd >= w0) & (ts_sd <= w1)
cand_w = cand_scg[(cand_scg >= w0) & (cand_scg <= w1)]
s1_w = s1_scg[(s1_scg >= w0) & (s1_scg <= w1)]
s2_w = s2_scg[(s2_scg >= w0) & (s2_scg <= w1)]
hand_w = h_scg_pk[(h_scg_pk >= w0) & (h_scg_pk <= w1)]

# rejected candidates = candidates in window that are NOT part of an accepted pair
accepted_set = set(s1_w.tolist()) | set(s2_w.tolist())
rejected_w = np.array([t for t in cand_w if t not in accepted_set])

def env_at(ts_query, ts_arr, env_arr):
    idx = np.searchsorted(ts_arr, ts_query)
    idx = np.clip(idx, 0, len(env_arr)-1)
    return env_arr[idx]

fig, axs = plt.subplots(5, 1, figsize=(16, 13), sharex=True,
                         gridspec_kw={"height_ratios":[1,1,1,1,0.35]})

axs[0].plot(dn(ts_s[mR]), x_s[mR], color="0.4", lw=0.6)
axs[0].set_ylabel("raw SCG"); axs[0].set_title("1) Raw SCG signal")
axs[0].grid(True, alpha=0.3)

axs[1].plot(dn(ts_s[mR]), xf_s[mR], color="#1f77b4", lw=0.7)
axs[1].set_ylabel("bandpassed"); axs[1].set_title("2) Bandpassed 10-100Hz")
axs[1].grid(True, alpha=0.3)

axs[2].plot(dn(ts_sd[mD]), env_sd[mD], color="#ff7f0e", lw=1.2)
axs[2].set_ylabel("Shannon energy\n(avg + Gaussian smooth)")
axs[2].set_title("3) Shannon-energy envelope, Gaussian-smoothed")
axs[2].grid(True, alpha=0.3)

axs[3].plot(dn(ts_sd[mD]), sharp_s[mD], color="#2ca02c", lw=1.2, label="sharpened envelope")
axs[3].axhline(0.6, color="0.5", ls="--", lw=1, label="detection threshold (0.6)")
if len(rejected_w):
    axs[3].scatter(dn(rejected_w), env_at(rejected_w, ts_sd, sharp_s), s=40, color="0.6",
                    marker="x", label="candidate peak (rejected)", zorder=4)
axs[3].scatter(dn(s1_w), env_at(s1_w, ts_sd, sharp_s), s=90, color="#d62728",
                marker="o", edgecolor="k", label="accepted S1", zorder=5)
axs[3].scatter(dn(s2_w), env_at(s2_w, ts_sd, sharp_s), s=90, color="#9467bd",
                marker="^", edgecolor="k", label="accepted S2", zorder=5)
for a, b in zip(s1_w, s2_w):
    axs[3].plot(dn([a, b]), [env_at(a, ts_sd, sharp_s), env_at(b, ts_sd, sharp_s)],
                color="k", lw=1, alpha=0.5, zorder=3)
axs[3].set_ylabel("sharpened\n(local q995 + exp)")
axs[3].set_title("4) Local-q995-sharpened envelope -- peak-picking + S1/S2 pairing (180±20ms, ≥150ms refractory)")
axs[3].legend(loc="upper right", fontsize=8, ncol=2)
axs[3].grid(True, alpha=0.3)

axs[4].vlines(dn(hand_w), 0, 1, color="#145214", lw=1.5)
axs[4].vlines(dn(s1_w), 0, 1, color="#d62728", lw=1.5, linestyle=":")
axs[4].set_ylim(0, 1); axs[4].set_yticks([])
axs[4].set_title("5) hand-clicked SCG beats (green) vs algorithm-accepted cycles (red dotted)", fontsize=10)
axs[4].xaxis.set_major_formatter(DateFormatter("%H:%M:%S", tz=UTC))
axs[4].set_xlim(dn(w0)[0], dn(w1)[0])

fig.suptitle("[Chelten] SCG Shannon-energy S1/S2 peak-finding pipeline, illustrated -- 17:47:00-17:47:16", fontsize=13)
fig.tight_layout()
out = "/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_shannon_scg_illustration.png"
fig.savefig(out, dpi=150)
print("->", out)
print(f"in window: {len(cand_w)} candidates, {len(s1_w)} accepted cycles, {len(rejected_w)} rejected, {len(hand_w)} hand clicks")
