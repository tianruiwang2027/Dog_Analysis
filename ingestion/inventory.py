#!/usr/bin/env python3
"""Complete-picture inventory of the unified hive: per (dog x modality) window,
fs, coverage, gaps and ADC-saturation, plus the SCG-vs-ECG temporal OVERLAP
(where a HR comparison is even possible). Emits analysis/registry.json and a
slide-style overview timeline figure (gap-aware bars; saturation shaded).

Raw stays raw: nothing is resampled. Coverage = real samples / fs / span;
segments are split on true gaps (Δts > 5x the nominal period)."""
import os, json, glob
import numpy as np, polars as pl
import datetime
from zoneinfo import ZoneInfo
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
HIVE = os.path.join(HERE, "hive")
OUT = os.path.join(HERE, "analysis"); os.makedirs(OUT, exist_ok=True)
FIGS = os.path.join(HERE, "figs"); os.makedirs(FIGS, exist_ok=True)
TZ = ZoneInfo("America/Chicago"); UTC = datetime.timezone.utc

SAT_INT16 = 32767
BIOPAC_RAIL = 4.999            # ECG100C ±5 mV amplifier rail

def lab(p):
    return {s.split("=", 1)[0]: s.split("=", 1)[1] for s in p.split("/") if "=" in s}

def dn(us):
    return mdates.date2num([datetime.datetime.fromtimestamp(t/1e6, tz=UTC).astimezone(TZ)
                            for t in np.atleast_1d(np.asarray(us, dtype="int64"))])

def local(us):
    return datetime.datetime.fromtimestamp(int(us)/1e6, tz=UTC).astimezone(TZ)

def segments(ts, fs, gap_factor=5):
    """continuous segments split on Δts > gap_factor * nominal period."""
    period = 1e6 / fs
    if len(ts) < 2:
        return [(int(ts[0]), int(ts[0]))] if len(ts) else []
    d = np.diff(ts)
    brk = np.where(d > gap_factor * period)[0]
    starts = np.concatenate([[0], brk + 1]); ends = np.concatenate([brk, [len(ts) - 1]])
    return [(int(ts[s]), int(ts[e])) for s, e in zip(starts, ends)]

def load_meta(part_dir):
    for fn in ("channels.json", "device.json"):
        p = os.path.join(part_dir, fn)
        if os.path.exists(p):
            return json.load(open(p))
    return {}

# ----- walk hive: ECG partitions (biopac, polar) and SCG stream45 -----
reg = []
for p in sorted(glob.glob(HIVE + "/username=*/device=*/stream=*/date=*/data_0.parquet")):
    L = lab(p); modality = L["username"]; dog = L["device"]; stream = L["stream"]; date = L["date"]
    # only inventory signal streams: all ECG (stream0), SCG accel (stream45)
    if modality == "scg_mwd" and stream != "45":
        continue
    df = pl.read_parquet(p)
    ts = df["ts"].to_numpy().astype("int64")
    n = len(ts)
    d = np.diff(ts); d = d[d > 0]
    fs = float(1e6 / np.median(d)) if len(d) else float("nan")
    span_s = (ts[-1] - ts[0]) / 1e6 if n > 1 else 0.0
    cov = (n / fs) / span_s if (span_s > 0 and fs == fs) else (1.0 if n else 0.0)
    segs = segments(ts, fs)
    ngaps = max(0, len(segs) - 1)
    # saturation
    sat = {}
    if modality == "scg_mwd":
        chans = [c for c in df.columns if c.startswith("c")]
        railed = np.zeros(n, bool)
        for c in chans:
            v = df[c].to_numpy()
            railed |= (np.abs(v) >= SAT_INT16)
            sat[c] = float((np.abs(v) >= SAT_INT16).mean())
        sat["any_axis"] = float(railed.mean())
        instrument = "MWD 3-axis accel (SCG)"; mod = "SCG"
    else:
        for c in [c for c in ("c1", "c2", "c3") if c in df.columns]:
            v = df[c].to_numpy()
            sat[c] = float((np.abs(v) >= BIOPAC_RAIL).mean()) if modality == "ecg_biopac" else 0.0
        instrument = "BioPac (12-lead set)" if modality == "ecg_biopac" else "Polar H10"
        mod = "ECG"
    meta = load_meta(os.path.dirname(p))
    reg.append(dict(modality=modality, mod=mod, instrument=instrument, dog=dog, stream=stream,
                    date=date, fs=fs, n=n, span_s=span_s, coverage=cov, ngaps=ngaps,
                    t0=int(ts[0]), t1=int(ts[-1]),
                    t0_local=str(local(ts[0])), t1_local=str(local(ts[-1])),
                    sat=sat, segs=segs))

# ----- SCG ∩ ECG overlap per dog (comparison-feasible windows) -----
def interval_overlap(A, B):
    out = []
    for a0, a1 in A:
        for b0, b1 in B:
            lo, hi = max(a0, b0), min(a1, b1)
            if hi > lo:
                out.append((lo, hi))
    return out

overlap = {}
for dog in sorted({r["dog"] for r in reg if r["mod"] == "SCG"}):
    scg = [r for r in reg if r["dog"] == dog and r["mod"] == "SCG"]
    if not scg:
        continue
    scg_segs = scg[0]["segs"]
    for ecgr in [r for r in reg if r["dog"] == dog and r["mod"] == "ECG"]:
        ov = interval_overlap(scg_segs, ecgr["segs"])
        sec = sum((b - a) for a, b in ov) / 1e6
        overlap.setdefault(dog, []).append(dict(ecg=ecgr["modality"], overlap_s=sec,
                                                 intervals=[(local(a).strftime("%H:%M:%S"),
                                                             local(b).strftime("%H:%M:%S")) for a, b in ov[:8]]))

json.dump({"registry": reg, "overlap": overlap}, open(os.path.join(OUT, "registry.json"), "w"), indent=2, default=str)

# ----- printed table -----
print(f"{'modality':<11}{'dog':<9}{'fs(Hz)':>8}{'n':>12}{'span(min)':>10}{'cov%':>7}{'gaps':>6}  window(local)        sat%")
for r in sorted(reg, key=lambda r: (r["dog"], r["modality"])):
    satv = r["sat"].get("any_axis", max([v for k, v in r["sat"].items()] or [0]))
    print(f"{r['modality']:<11}{r['dog']:<9}{r['fs']:>8.1f}{r['n']:>12,}{r['span_s']/60:>10.1f}"
          f"{100*r['coverage']:>7.1f}{r['ngaps']:>6}  {r['t0_local'][11:19]}->{r['t1_local'][11:19]}"
          f"  {100*satv:>5.1f}")
print("\n== SCG ∩ ECG overlap (comparison-feasible) ==")
for dog, lst in overlap.items():
    for o in lst:
        print(f"  {dog:<9} SCG ∩ {o['ecg']:<10} = {o['overlap_s']/60:6.1f} min   {o['intervals']}")

# ----------------------------- overview timeline figure -----------------------------
# Day-2 (2026-06-26) is the SCG-vs-ECG experiment; Day-1 Dog1/Dog2 (06-25) were an
# ECG-only pilot with no SCG -> kept in the table/JSON but off this timeline so the
# time axis stays readable.
DAY2 = "20260626"
DOGS = ["Dasty", "Chelten", "Chuck", "MWD2A65", "MWD5905"]
ROWS = [("ecg_biopac", "BioPac ECG", "#1f77b4"),
        ("ecg_polar",  "Polar H10 ECG", "#2ca02c"),
        ("scg_mwd",    "MWD SCG", "#d62728")]
present = [d for d in DOGS if any(r["dog"] == d and r["date"] == DAY2 for r in reg)]
lanes = []
for d in present:
    for modk, modlabel, col in ROWS:
        if any(r["dog"] == d and r["modality"] == modk and r["date"] == DAY2 for r in reg):
            lanes.append((d, modk, modlabel, col))

fig, ax = plt.subplots(figsize=(16, 12))
yticks, ylabels = [], []
for i, (d, modk, modlabel, col) in enumerate(lanes):
    y = len(lanes) - 1 - i
    yticks.append(y); ylabels.append(f"{d} · {modlabel}")
    r = next(r for r in reg if r["dog"] == d and r["modality"] == modk and r["date"] == DAY2)
    bars = [(dn(a)[0], max(dn(b)[0] - dn(a)[0], 1e-9)) for a, b in r["segs"]]
    ax.broken_barh(bars, (y - 0.38, 0.76), facecolors=col, edgecolor="none", alpha=0.9)
    # annotate fs + coverage at the right
    ax.text(1.005, y, f"{r['fs']:.0f} Hz · {100*r['coverage']:.0f}% · n={r['n']:,}",
            transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=8, color="0.25")
# highlight SCG∩ECG comparison windows behind the relevant lanes (gold)
for i, (d, modk, modlabel, col) in enumerate(lanes):
    if modk != "scg_mwd":
        continue
    y = len(lanes) - 1 - i
    ecg_ys = [len(lanes) - 1 - j for j, (dd, mk, _, _) in enumerate(lanes)
              if dd == d and mk in ("ecg_biopac", "ecg_polar")]
    if not ecg_ys:
        continue
    lo_y, hi_y = min(min(ecg_ys), y), max(max(ecg_ys), y)
    for o in overlap.get(d, []):
        # rebuild merged intervals in us for shading
        scg = next(r for r in reg if r["dog"] == d and r["mod"] == "SCG")
        ecgr = next(r for r in reg if r["dog"] == d and r["modality"] == o["ecg"])
        for a, b in interval_overlap(scg["segs"], ecgr["segs"]):
            if (b - a) / 1e6 < 30:    # ignore sub-30s slivers in the shading
                continue
            ax.add_patch(plt.Rectangle((dn(a)[0], lo_y - 0.45), dn(b)[0] - dn(a)[0],
                                       (hi_y - lo_y) + 0.9, facecolor="gold", alpha=0.18,
                                       edgecolor="goldenrod", lw=0.8, zorder=0))
ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=10)
ax.set_ylim(-0.6, len(lanes) - 0.4)
ax.xaxis.set_major_formatter(DateFormatter("%H:%M", tz=TZ))
ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 30)))
t0a = min(r["t0"] for r in reg if r["date"] == DAY2)
t1a = max(r["t1"] for r in reg if r["date"] == DAY2)
ax.set_xlim(dn(t0a - 6 * 60 * 1_000_000)[0], dn(t1a + 6 * 60 * 1_000_000)[0])
ax.set_xlabel("America/Chicago local time · 2026-06-26   (gold = SCG∩ECG comparison-feasible window)", fontsize=11)
ax.grid(True, axis="x", alpha=0.3)
ax.set_title("Dog ECG/SCG experiment (Day-2, 2026-06-26) — data coverage timeline\n"
             "bars = continuous segments · gaps are real (not interpolated)", fontsize=14, pad=12)
legend = [Patch(facecolor=c, label=l) for _, l, c in ROWS] + \
         [Patch(facecolor="gold", alpha=0.4, label="SCG∩ECG comparison window")]
ax.legend(handles=legend, loc="lower right", framealpha=0.95)
fig.tight_layout()
out = os.path.join(FIGS, "overview_timeline.png")
fig.savefig(out, dpi=200); print("\n->", out)
