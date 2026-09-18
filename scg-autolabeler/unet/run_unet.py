"""
From-scratch U-Net baseline (no pretraining). Thin wrapper around
unet.unet_train.finetune_and_evaluate -- see results/UNET_RESULTS.md for the full
writeup of what this does and the bugs found along the way.

Run from anywhere: `python unet/run_unet.py` (or `python run_unet.py` from within
this folder) -- the sys.path bootstrap below makes the sibling `common`/`unet`
packages importable either way.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unet.unet_train import finetune_and_evaluate

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

N_EPOCHS = int(os.environ.get("N_EPOCHS", "40"))

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    finetune_and_evaluate(
        run_name="from_scratch",
        checkpoint_path=os.path.join(OUTPUT_DIR, "unet_checkpoint.pt"),
        best_path=os.path.join(OUTPUT_DIR, "unet_best.pt"),
        results_path=os.path.join(RESULTS_DIR, "unet_results.json"),
        n_epochs=N_EPOCHS,
        pretrained_state_dict=None,
    )
