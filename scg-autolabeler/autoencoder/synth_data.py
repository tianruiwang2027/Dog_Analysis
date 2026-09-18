"""
Synthetic multi-dog SCG+ECG generator.

This session has no access to the real hive parquet files, so the demo pipeline
(`demo_end_to_end.py`) runs against data generated here instead. It is built to
reproduce the specific failure mode this whole proposal targets: a per-dog SCG
amplitude-scale mismatch (PIPELINE_SUMMARY.md section 9.5's ~32x finding) plus
per-dog SCG morphology jitter and an unknown ECG<->SCG clock offset, while ECG QRS
morphology stays comparatively stable across dogs (section 6's actual empirical
finding). It is NOT a claim about what real SCG/ECG looks like -- it exists only to
give the code something to run against so shapes, losses, and the overall data flow
can be checked end to end before this ever touches real hive data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SynthRecording:
    dog: str
    fs: float
    duration_s: float
    x_ecg: np.ndarray
    ts_ecg: np.ndarray  # microseconds, ECG clock
    x_scg: np.ndarray
    ts_scg: np.ndarray  # microseconds, SCG clock (has an unknown offset vs ECG clock)
    true_beat_times_ecg_clock: np.ndarray  # ground truth, for evaluation only
    true_lag_us: float  # ecg_clock + true_lag_us -> scg_clock, for evaluation only
    scg_scale: float  # this dog's SCG amplitude scale, for evaluation only

    @property
    def fs_ecg(self) -> float:
        """Real recordings (teacher/real_data.py) may have different ECG/SCG sample
        rates (e.g. a 130Hz Polar strap vs a 2kHz accelerometer); synthetic recordings
        use one shared rate for both."""
        return self.fs

    @property
    def fs_scg(self) -> float:
        return self.fs


def _pulse(t, center, width, amp):
    return amp * np.exp(-0.5 * ((t - center) / width) ** 2)


def _make_beat_times(duration_s, hr_mean_bpm, hr_jitter_bpm, rng, dt=0.05, tau_s=4.0):
    """Smooth (Ornstein-Uhlenbeck) HR trajectory rather than i.i.d.-per-beat jitter --
    real physiological HR varies on a several-second timescale, not beat to beat, and
    the downstream HR-agreement metric (a fixed +-2s smoothing window, copied verbatim
    from PIPELINE_SUMMARY.md section 3) assumes that. Beats are then placed by
    integrating instantaneous rate (an inhomogeneous point process), not by drawing an
    independent RR interval each time."""
    n_steps = int(duration_s / dt)
    theta = dt / tau_s
    sigma = hr_jitter_bpm * np.sqrt(2 * theta)
    hr = np.empty(n_steps)
    hr[0] = hr_mean_bpm
    for i in range(1, n_steps):
        hr[i] = hr[i - 1] + theta * (hr_mean_bpm - hr[i - 1]) + sigma * rng.standard_normal()
    hr = np.clip(hr, 40, 220)

    phase = np.cumsum((hr / 60.0) * dt)
    t_grid = np.arange(n_steps) * dt
    n_beats = int(phase[-1])
    thresholds = np.arange(1, n_beats + 1)
    beat_times = np.interp(thresholds, phase, t_grid)
    return beat_times[(beat_times > 0.3) & (beat_times < duration_s - 0.3)]


def make_recording(
    dog: str,
    duration_s: float = 120.0,
    fs: float = 500.0,
    hr_mean_bpm: float = 90.0,
    hr_jitter_bpm: float = 8.0,
    scg_scale: float = 1.0,
    scg_morphology_jitter: float = 0.15,
    ecg_noise: float = 0.05,
    scg_noise: float = 0.08,
    n_scg_distractors_per_min: float = 6.0,
    true_lag_us: float = 3_000_000.0,
    seed: int = 0,
) -> SynthRecording:
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs

    beat_times = _make_beat_times(duration_s, hr_mean_bpm, hr_jitter_bpm, rng)

    # Carve out a couple of short beat-free "quiet" windows (motion/handling gaps --
    # the real recordings have these too, e.g. PIPELINE_SUMMARY.md section 5's
    # pre-settling segment) and drop any beat that falls inside one. At a typical RR
    # this dog's beats are packed too densely (RR often < the 0.5s Stage-1 refractory
    # window) for a distractor burst to be placed anywhere and still survive as its
    # own candidate -- these windows are where the synthetic distractors below live.
    n_quiet = max(1, int(duration_s // 30))
    quiet_windows = []
    for i in range(n_quiet):
        center = duration_s * (i + 1) / (n_quiet + 1)
        quiet_windows.append((center - 2.5, center + 2.5))
    in_quiet = np.zeros_like(beat_times, dtype=bool)
    for lo, hi in quiet_windows:
        in_quiet |= (beat_times >= lo) & (beat_times <= hi)
    beat_times = beat_times[~in_quiet]

    # --- ECG: QRS-like biphasic pulse per beat, stable morphology across dogs ---
    x_ecg = ecg_noise * rng.standard_normal(n)
    for bt in beat_times:
        x_ecg += _pulse(t, bt, 0.008, 3.0) - _pulse(t, bt + 0.02, 0.012, 1.0)

    # --- SCG: S1 (at beat) + S2 (140-280ms later), amplitude/shape vary per dog ---
    x_scg = scg_noise * scg_scale * rng.standard_normal(n)
    for bt in beat_times:
        s1_amp = scg_scale * (1.0 + scg_morphology_jitter * rng.standard_normal())
        s1_width = 0.010 * (1.0 + 0.3 * rng.standard_normal())
        x_scg += _pulse(t, bt, max(s1_width, 0.003), s1_amp)

        s2_gap = rng.uniform(0.14, 0.28)
        s2_amp = 0.5 * scg_scale * (1.0 + scg_morphology_jitter * rng.standard_normal())
        x_scg += _pulse(t, bt + s2_gap, 0.012, s2_amp)

    # distractor bursts: motion/handling artifacts unrelated to any real beat, placed
    # only inside the beat-free quiet windows carved out above, >=0.6s apart, so they
    # survive the frozen Stage-1 refractory (0.5s) as their own candidates instead of
    # being swallowed by a nearby real beat.
    n_distractors = int(n_scg_distractors_per_min * duration_s / 60.0)
    placed = []
    n_placed = 0
    attempts = 0
    while n_placed < n_distractors and attempts < n_distractors * 50:
        attempts += 1
        lo, hi = quiet_windows[rng.integers(len(quiet_windows))]
        dt = rng.uniform(lo + 0.3, hi - 0.3)
        if placed and min(abs(dt - p) for p in placed) < 0.6:
            continue
        x_scg += _pulse(t, dt, 0.02, scg_scale * rng.uniform(1.5, 2.5))
        placed.append(dt)
        n_placed += 1

    ts_ecg_us = (t * 1e6).astype(np.int64)
    ts_scg_us = ts_ecg_us + int(true_lag_us)  # SCG clock is ECG clock shifted by the lag

    return SynthRecording(
        dog=dog,
        fs=fs,
        duration_s=duration_s,
        x_ecg=x_ecg,
        ts_ecg=ts_ecg_us,
        x_scg=x_scg,
        ts_scg=ts_scg_us,
        true_beat_times_ecg_clock=(beat_times * 1e6).astype(np.int64),
        true_lag_us=true_lag_us,
        scg_scale=scg_scale,
    )


def make_dog_pool(n_train_dogs: int = 4, seed: int = 0) -> tuple[list[SynthRecording], SynthRecording]:
    """Returns (training pool with wide dog-to-dog scale/morphology variation, one
    held-out 'unlabeled' dog analogous to Dasty)."""
    rng = np.random.default_rng(seed)
    train = []
    for i in range(n_train_dogs):
        train.append(
            make_recording(
                dog=f"train_dog_{i}",
                duration_s=90.0,
                hr_mean_bpm=rng.uniform(70, 110),
                scg_scale=rng.uniform(0.3, 3.0),  # wide per-dog amplitude range
                scg_morphology_jitter=rng.uniform(0.1, 0.25),
                true_lag_us=rng.uniform(-5e6, 5e6),
                seed=seed * 100 + i,
            )
        )
    held_out = make_recording(
        dog="held_out_dog",
        duration_s=90.0,
        # NOTE: kept below the ~120bpm primary-candidate refractory ceiling
        # (PRIMARY_REFRACT_S=0.50, PIPELINE_SUMMARY.md section 1) on purpose, so this
        # demo isolates the OOD scale/morphology problem the teacher targets from the
        # separate, already-documented refractory-ceiling problem (section 6/9) that a
        # learned matcher can't fix by itself -- real Dasty (~123bpm) hits both at once.
        hr_mean_bpm=95.0,
        scg_scale=8.0,  # deliberately far outside the training pool's range
        scg_morphology_jitter=0.3,
        n_scg_distractors_per_min=14.0,  # stress precision harder than the training pool
        true_lag_us=1.7e6,
        seed=seed * 100 + 999,
    )
    return train, held_out
