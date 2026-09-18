#!/usr/bin/env python3
"""Consolidate every modality into ONE clean parquet hive (the 'much better shape'):

    hive/username=<modality>/device=<dog|MWDxxxx>/stream=<s>/date=<utc>/data_0.parquet

  modality = ecg_biopac (written by biopac_to_hive.py)
           | ecg_polar  (Polar H10 130 Hz ECG, from ecg_cache)
           | scg_mwd    (MWD 3-axis accel SCG + telemetry, from the R2 lake pull)

Device is keyed by DOG NAME for the three identified dogs so a dog's ECG and SCG
sit under the same device key across modalities (trivial per-dog joins). Unknown
MWD devices keep their hex id. Raw stays raw: parquet bodies are copied verbatim;
we only add JSON metadata sidecars (channel/stream meaning, fs, instrument).
"""
import os, json, glob, shutil
import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
HIVE = os.path.join(HERE, "hive")
ECG_CACHE = os.path.join(HERE, "ecg_cache")
LAKE = os.path.expanduser("~/.qsib-cache")

# hex suffix -> dog (from the BioPac/Polar filenames)
HEX2DOG = {"7D82": "Chuck", "50D1": "Chelten", "0868": "Dasty"}

# ECG_cache device -> (which dog, which experiment day label, instrument)
POLAR_DEVS = {  # device dir in ecg_cache -> dog name to use in hive
    "Chelten": "Chelten", "Chuck": "Chuck", "Dog1": "Dog1", "Dog2": "Dog2",
}


def lab(p):
    return {s.split("=", 1)[0]: s.split("=", 1)[1] for s in p.split("/") if "=" in s}


def fs_of(ts):
    d = np.diff(ts.astype("int64")); d = d[d > 0]
    return float(1e6 / np.median(d)) if len(d) else None


def copy_part(src_parquet, dst_part):
    os.makedirs(dst_part, exist_ok=True)
    shutil.copy(src_parquet, os.path.join(dst_part, "data_0.parquet"))


# ---------------------------------------------------------------- ECG (Polar) --
print("== ecg_polar ==")
for p in sorted(glob.glob(ECG_CACHE + "/username=*/device=*/stream=*/date=*/data_0.parquet")):
    L = lab(p)
    dog = POLAR_DEVS.get(L["device"])
    if dog is None:
        continue
    dst = os.path.join(HIVE, "username=ecg_polar", f"device={dog}",
                       f"stream={L['stream']}", f"date={L['date']}")
    copy_part(p, dst)
    df = pl.read_parquet(p)
    ts = df["ts"].to_numpy()
    meta = {
        "dog": dog, "instrument": "Polar H10", "modality": "ECG",
        "fs_hz": 130.0, "fs_est_hz": fs_of(ts), "n_samples": len(df),
        "channels": {"c1": {"name": "ECG", "unit": "uV"}},
        "note": ("Polar H10 single-lead ECG, 130 Hz only when chest-strap contact "
                 "is good; lead-off/poor-contact regions are flagged downstream."),
    }
    with open(os.path.join(dst, "channels.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  {dog}: {len(df):,} rows -> {dst}")

# ----------------------------------------------------------------- SCG (MWD) --
print("== scg_mwd ==")
STREAM_META = {
    "45": {"name": "accelerometer_scg", "fs_hz": 2000.0,
           "channels": {"c1": {"name": "accel_x", "unit": "counts_int16"},
                        "c2": {"name": "accel_y", "unit": "counts_int16"},
                        "c3": {"name": "accel_z", "unit": "counts_int16"}},
           "note": "3-axis accelerometer @2 kHz = the SCG. ±32768 = ADC saturation (motion/lead-off)."},
    "0":  {"name": "battery?", "channels": {"c1": {"name": "battery_mV?", "unit": "mV?"}},
           "note": "sparse, ~3.5-4.2 -> likely Li-ion battery voltage in mV (uncertain)."},
    "19": {"name": "telemetry_1hz?", "channels": {"c1": {"name": "telemetry?", "unit": "?"}},
           "note": "~1 Hz scalar, meaning uncertain (not HR-scaled)."},
    "56": {"name": "device_clock_counter?",
           "channels": {"c1": {"name": "device_epoch_us?", "unit": "us?"},
                        "c2": {"name": "counter?", "unit": "?"}},
           "note": "~2 Hz; c1 looks like a device epoch-us timestamp, c2 a counter. Housekeeping."},
}
# group lake partitions by device
devparts = {}
for p in sorted(glob.glob(LAKE + "/username=*/device=MWD*/stream=*/date=*/data_0.parquet")):
    L = lab(p)
    devparts.setdefault(L["device"], []).append(p)

for mwd, parts in sorted(devparts.items()):
    suffix = mwd[-4:]
    dog = HEX2DOG.get(suffix)
    devkey = dog if dog else f"MWD{suffix}"
    # skip pure power-on bursts (no stream45, or stream45 with <=40 rows total)
    s45 = [p for p in parts if lab(p)["stream"] == "45"]
    n45 = sum(pl.read_parquet(p, columns=["ts"]).height for p in s45)
    if n45 <= 100:
        print(f"  skip {mwd} ({devkey}): burst-only, stream45 n={n45}")
        continue
    streams_meta = {}
    for p in parts:
        L = lab(p)
        st = L["stream"]
        dst = os.path.join(HIVE, "username=scg_mwd", f"device={devkey}",
                           f"stream={st}", f"date={L['date']}")
        copy_part(p, dst)
        df = pl.read_parquet(p)
        sm = dict(STREAM_META.get(st, {"name": f"stream_{st}", "channels": {}}))
        sm["fs_est_hz"] = fs_of(df["ts"].to_numpy())
        sm["n_samples"] = len(df)
        streams_meta[st] = sm
    devmeta = {"dog": dog, "mwd_device": mwd, "hex": suffix,
               "modality": "SCG (3-axis accelerometer)", "streams": streams_meta}
    ddir = os.path.join(HIVE, "username=scg_mwd", f"device={devkey}")
    with open(os.path.join(ddir, "device.json"), "w") as f:
        json.dump(devmeta, f, indent=2)
    print(f"  {devkey} ({mwd}): streams={sorted(streams_meta)} scg45_n={n45:,}")

print("\nhive built at", HIVE)
