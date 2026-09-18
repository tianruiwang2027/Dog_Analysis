# Dog_Analysis

Combined repo for the Neurolux dog SCG/ECG heart-rate work:

- `scg-autolabeler/` — MAE-pretrained autoencoder + from-scratch 1D U-Net pipelines for
  detecting SCG heartbeats (see its own README).
- `best-pipeline-chelten/` — the validated 4-stage SCG-vs-ECG pipeline (r=0.9095 SCG,
  r=0.9820 Pan-Tompkins ECG vs hand clicks) (see its own README).

Raw data is not committed here (see `.gitignore`) — `data/` is a local symlink to the
shared dataset on disk, and `best-pipeline-chelten/timestamp_data/` holds local copies
of the raw hand-click annotation databases. Neither is pushed to GitHub.
