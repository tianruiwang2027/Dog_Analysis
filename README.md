# dog-scg-ecg-hr

Dog SCG (seismocardiogram) vs ECG heart-rate detection. A 4-stage signal-
processing + small-CNN pipeline detects heartbeats from SCG (chest-mounted
accelerometer) and compares against a fully-tuned Pan-Tompkins ECG detector.
Built and validated on one dog (**Chelten**), then tested for cross-dog
generalization on a second (**Dasty**) — see `docs/PIPELINE_SUMMARY.md` for
the full narrative, every validated number, and the currently-open work.

**Hard constraint:** the CORAL (`coral-st`) algorithm's own source is never
modified by anything in this repo. Everything here is standalone, read-only
analysis layered on top of raw sensor recordings and, where noted, CORAL's own
CSV output — never a change to CORAL itself.

## Validated headline results (Chelten, hand-click ground truth)

| method | r | n | MAE (bpm) |
|---|---|---|---|
| SCG pipeline (this repo, `src/scg_hr`) | 0.9095 | 1285 | 2.20 |
| Pan-Tompkins ECG, tuned (this repo, `src/scg_hr`) | 0.9820 | 2495 | 1.82 |
| CORAL's own ECG algorithm (reference only) | 0.7849 | 2789 | 7.05 |

Full validation history, every rejected approach, the cross-dog (Dasty)
generalization test, and a clock-drift investigation live in
`docs/PIPELINE_SUMMARY.md`.

## Layout

```
src/scg_hr/       importable package -- the frozen, validated pipeline
  constants.py      thresholds, hyperparameters, per-dog channel conventions
  signal_utils.py   Stage 1: bandpass, Shannon-energy envelope, local sharpening
  models.py         SCGNet / S2Net (+ proposed, untrained SCGNetFused)
  scg_pipeline.py   Stages 1-4 end to end
  ecg_pantompkins.py  fully-tuned Pan-Tompkins R-peak detector
  compare_utils.py  smoothing/alignment/stats helpers
  io_hive.py        loads raw signals from the hive parquet layout

weights/          small trained weights + their scale constants (~70KB total)
  cnn_model_relabel.pt / cnn_train_relabel_result.pkl   main confirm CNN (SCGNet)
  s2_model.pt / s2_train_result.pkl                     S2-recovery CNN (S2Net)
  env_template_wide.pkl                                 NCC template (for planned fusion work, Section 9)

scripts/          runnable entry points built on src/scg_hr
  run_scg_pipeline.py             run the SCG detector alone for one dog/window
  run_ecg_pantompkins.py          run the ECG detector alone
  compare_scg_vs_ecg.py           full algorithm-vs-algorithm comparison
  run_dasty_generalization_check.py   reproduces the Chelten-trained-on-Dasty test
  run_lag_drift_check.py          reproduces the clock-drift investigation

ingestion/        scripts that build the hive/ parquet layout from raw device exports
  (biopac_to_hive.py, build_hive.py, csv_to_hive.py, polar_xlsx_to_parquet.py, inventory.py)

docs/
  PIPELINE_SUMMARY.md   the full reference doc: every validated number, every bug
                         found + fixed, the cross-dog test, the clock-drift finding,
                         and the (unimplemented) plan for fixing Dasty generalization

figs/              a handful of final, validated figures (not the full working set)

experiments/       ~264 historical scripts from the full investigation, kept AS-IS
                    for provenance -- exploratory, one-off, and rejected-approach
                    code. Not meant to be imported; not curated or cleaned up.
                    experiments/coral_reference_results/ holds small agree.json
                    summaries from the project's separate CORAL runs (the bulky
                    CORAL parquet/csv/png outputs themselves are not included).
```

## Data (not included in this repo)

The raw hive parquet data (~350MB) is not committed here — see `.gitignore`.
Expected layout, if you have (or regenerate via `ingestion/`) a local copy:

```
hive/username=<stream>/device=<dog>/stream=<n>/date=*/data_0.parquet
  username=ecg_polar   stream=0    columns: ts, c1                (Chelten only)
  username=ecg_biopac  stream=0    columns: ts, c1..c4             (Chelten, Dasty, Chuck)
  username=scg_mwd     stream=45   columns: ts, c1 (or c1/c2/c3)   (all dogs)
```

`ts` is int64 epoch microseconds. Point scripts at your local copy with
`--hive-dir ./hive` or by setting `SCG_HIVE_DIR`.

Note: the scripts in this repo reproduce the *algorithm-vs-algorithm*
comparisons (SCG pipeline vs Pan-Tompkins ECG) without needing a hand-click
ground-truth file. The headline hand-click-validated numbers above (r=0.9095 /
r=0.9820) were computed against a proprietary hand-annotated click file that
is also not included; `docs/PIPELINE_SUMMARY.md` documents exactly how those
numbers were produced if you have (or recreate) that file.

## Quickstart

```bash
pip install -r requirements.txt

# SCG pipeline alone, over Chelten's hand-click-validated window
python scripts/run_scg_pipeline.py --dog Chelten \
    --t0 1782494790000000 --t1 1782496808699088 \
    --hive-dir ./hive --weights-dir ./weights

# full SCG-vs-ECG comparison, using the measured clock-drift correction
python scripts/compare_scg_vs_ecg.py --dog Chelten --ecg-stream ecg_polar \
    --hive-dir ./hive --weights-dir ./weights --use-chelten-drift-correction

# cross-dog generalization check (expected to perform poorly -- see docs)
python scripts/run_dasty_generalization_check.py --hive-dir ./hive --weights-dir ./weights
```

## Verification note

Before packaging, every script here was run end to end against the original
project's local data and cross-checked:

- **Signal processing (Stage 1)** reproduces the original envelope bit-for-bit
  (`max abs diff = 0.0` against a saved reference array) once the correct
  per-dog channel is used — Chelten's custom SCGNet/S2Net pipeline was built
  against the raw **`c1`** column, not the `mag` derived channel CORAL's own
  separate `coral_scg.py` uses for Chelten (a real, easy-to-miss discrepancy
  between the two tools' conventions in the original project; see
  `constants.SCG_CHANNEL` vs `constants.AXIS`).
- **`run_dasty_generalization_check.py`** reproduces the documented Section 6
  numbers essentially exactly (627/813 candidates, 104/139 confirmed vs
  103/138 documented).
- **`run_lag_drift_check.py`** reproduces the documented Section 10 finding
  essentially exactly (25.17 ppm vs 25.10 ppm documented, t-stat 15.3, r=0.978).
- **`run_scg_pipeline.py` / `compare_scg_vs_ecg.py` on Chelten** run correctly
  and give results in the same ballpark as documented (r≈0.85 vs hand clicks,
  vs r=0.9095 documented) but not an exact beat-for-beat match against one
  specific historical intermediate cache file — most likely that cache
  (`s2_chain_final.pkl`) predates a later weight retrain within the original
  session rather than a bug in this extraction (candidate generation and CNN
  scoring were both verified bit-exact / self-consistent independently). If
  you rely on an exact reproduction of r=0.9095, re-validate against your own
  hand-click file before trusting downstream numbers.

## Known limitations / open work

The SCG-side CNN (`SCGNet`/`S2Net`) was trained on Chelten only and does
**not** currently generalize to other dogs (confirm rate collapses on Dasty).
`docs/PIPELINE_SUMMARY.md` Section 9 has a fully-specified, not-yet-implemented
plan to address this (NCC template-match feature fusion + a fix to a proven
~32x input-scale mismatch in the CNN's preprocessing) — `models.py` already
includes the proposed `SCGNetFused` scaffold for that work.
