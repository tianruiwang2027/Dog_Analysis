"""
Loaders for the additional, non-Chelten/Chuck/Dasty domains used to pretrain the
query-based MIL matcher (teacher/mil_matcher.py) on more than one "shape" of SCG
signal: the epicardial canine/porcine accelerometer set (on-heart, invasive) and the
CEBS human dataset (chest-mounted, non-invasive, with real hand-verified R-peaks).

As with teacher/real_data.py, nothing here reads or uses any hand-clicked SCG label.
The CEBS annotations ARE used -- but only as the ECG-side beat anchor (the same role
Pan-Tompkins plays everywhere else in this project), never as an SCG label. Chelten
stays completely out of this file; it is the one recording pretraining is never
allowed to see, so it stays a fair blind validation target.
"""
from __future__ import annotations

import glob
import json

import numpy as np

from common import signal_utils
from common.real_data import RealRecording

try:
    import wfdb
except ImportError:  # pragma: no cover
    wfdb = None


def load_epicardial_dogs(paths: list[str], min_beats: int = 10) -> list[RealRecording]:
    """One RealRecording per JSON file (already pre-selected by the caller -- see
    run_multidomain_mil.py for how the 18-dog, longest-valid-recording selection was
    made). Skips any file whose 'ecg' channel is all-zero (a real data-quality issue
    found in this set, e.g. MV12's baseline file) or where our Pan-Tompkins detector
    finds fewer than min_beats beats -- a real failure mode on this set's LBBB
    (left-bundle-branch-block) intervention recordings, whose deliberately distorted
    QRS morphology this SNR-thresholded detector isn't tuned for, rather than a sign
    those recordings have no heartbeats at all."""
    recs = []
    skipped = []
    for path in paths:
        d = json.load(open(path))
        ecg = np.asarray(d["ecg"], dtype=np.float64)
        if ecg.size == 0 or float(ecg.std()) == 0.0:
            skipped.append((path, "all-zero ECG"))
            continue
        fs = float(d["sample_rate"])
        n_beats = len(signal_utils.detect_qrs(ecg, fs))
        if n_beats < min_beats:
            skipped.append((path, f"only {n_beats} beats detected"))
            continue
        ax = np.asarray(d["acc_x"], dtype=np.float64)
        ay = np.asarray(d["acc_y"], dtype=np.float64)
        az = np.asarray(d["acc_z"], dtype=np.float64)
        ax = ax - ax.mean()
        ay = ay - ay.mean()
        az = az - az.mean()
        x_scg = np.sqrt(ax**2 + ay**2 + az**2)
        n = len(ecg)
        ts = (np.arange(n) / fs * 1e6).astype(np.int64)
        ident = f"{d.get('identifier', 'unk')}_{d.get('intervention', '')}"
        recs.append(RealRecording(
            dog=ident, x_ecg=ecg, ts_ecg=ts, fs_ecg=fs,
            x_scg=x_scg, ts_scg=ts, fs_scg=fs,
            domain="dog_epicardial",
        ))
    if skipped:
        print(f"  [load_epicardial_dogs] skipped {len(skipped)} file(s):")
        for path, reason in skipped:
            print(f"    {path.split('/')[-1]}: {reason}")
    return recs


def load_human_cebs(record_names: list[str], base_dir: str) -> list[RealRecording]:
    """One RealRecording per WFDB record (e.g. 'b001', 'p001'). Uses the dataset's own
    hand-verified R-peak annotations (lead-I-derived, ann.chan == 0) as ecg_beats_hint
    instead of running our own detect_qrs -- this is real ground truth on the ECG side,
    the same trusted role Pan-Tompkins plays for the dog recordings, just better."""
    if wfdb is None:
        raise ImportError("wfdb is required for load_human_cebs (pip install wfdb)")
    recs = []
    for name in record_names:
        path = f"{base_dir}/{name}"
        rec = wfdb.rdrecord(path)
        ann = wfdb.rdann(path, "atr")
        fs = float(rec.fs)
        x_scg = rec.p_signal[:, rec.sig_name.index("SCG")].astype(np.float64)
        x_ecg = rec.p_signal[:, rec.sig_name.index("I")].astype(np.float64)
        n = len(x_scg)
        ts = (np.arange(n) / fs * 1e6).astype(np.int64)
        samp = np.asarray(ann.sample)
        chan = np.asarray(ann.chan)
        beat_hint = (samp[chan == 0] / fs * 1e6).astype(np.int64)
        recs.append(RealRecording(
            dog=f"human_{name}", x_ecg=x_ecg, ts_ecg=ts, fs_ecg=fs,
            x_scg=x_scg, ts_scg=ts, fs_scg=fs,
            domain="human", ecg_beats_hint=beat_hint,
        ))
    return recs
