# SCG heart-rate extraction

Extracting heart rate / HRV from seismocardiography (SCG) signals, using Chelten's
~37.6-minute hand-clicked SCG recording as the only real ground truth. Two independent
approaches were tried and are kept side by side here:

- **`autoencoder/`** — a joint SCG+ECG masked-autoencoder (MAE), pretrained on unlabeled
  multi-domain data (other dog recordings + a human SCG/ECG dataset), then fine-tuned
  with a small supervised head on 30 minutes of Chelten's real labels. Also includes the
  weakly-supervised MIL auto-labeling machinery the MAE pretraining and the distillation
  step both depend on, and a from-scratch (no pretraining) baseline for comparison.
- **`unet/`** — a 1D U-Net that predicts a per-sample peak-proximity mask directly on the
  continuous SCG envelope. No Stage-1 candidate detection is involved in training or
  inference; it just needs the envelope and the real hand-clicked labels.

`common/` holds the infrastructure both approaches share: Stage-1 SCG candidate
detection (`signal_utils.py`), the Chelten data loader (`real_data.py`), and the
supervised fine-tune/grading utilities used by both (`finetune_chelten.py`).

Each pipeline's write-up and results live in its own `results/` subfolder
(`autoencoder/results/`, `unet/results/`).

## Setup

```
pip install -r requirements.txt
```

### Data

The real recordings aren't in this repo. Put them in a `data/` folder at the repo root:

```
data/
├── Training Data/
│   ├── ecg_polar_chelten.parquet
│   ├── MWD_Chelten.parquet
│   ├── ecg_biopac_Chuck.parquet
│   ├── MWD_Chuck.parquet
│   ├── ecg_Dasty_Biopac.parquet
│   └── MWD_Dasty.parquet
├── qdaq-gui-annotations/
│   ├── Annotation Chelten SCG
│   └── Annotation Chelten
├── epicardially-attached-cardiac-accelerometer-data-from-canines-and-porcines-1.0.0/
│   └── accelerometer_data/...
└── Human-SCG-ECG-Data/
    └── (WFDB records: b001, b002, b003, p001, ...)
```

If your data lives somewhere else, set `SCG_DATA_ROOT` instead of moving anything
(see `common/config.py`).

## Reproducing the results

Autoencoder pipeline, run in order from the repo root (each step's output feeds the
next):

```
python autoencoder/run_option_a.py          # MAE pretraining + 30-min supervised fine-tune
python autoencoder/run_distill.py           # distill the fine-tuned teacher into a small MCU-sized net
python autoencoder/run_repeated_trials.py   # Monte Carlo eval: MAE-pretrained vs. from-scratch
python autoencoder/make_comparison_plot.py
python autoencoder/make_repeated_trials_plot.py
```

U-Net pipeline:

```
python unet/run_unet.py
python unet/make_unet_comparison_plot.py
```

See `autoencoder/results/OPTION_A_RESULTS.md`, `autoencoder/results/REPEATED_TRIALS_RESULTS.md`,
and `unet/results/UNET_RESULTS.md` for what was actually found, including a couple of
real bugs (BatchNorm miscalibration, test-set threshold leakage) hit and fixed along the
way, and the "Stage-1 ceiling" finding that turned out to matter more than either
pipeline's own precision/recall numbers.
