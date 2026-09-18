"""Frozen CNN architectures: SCGNet (main confirm classifier) and S2Net (S2-recovery
classifier). Both are tiny (6,481 params each) by design -- a deliberately
conservative choice for a single-dog (Chelten-only) training set; see
docs/PIPELINE_SUMMARY.md Section 9.1 for why a larger architecture (e.g. 1D-UNet)
was rejected.

Weights live in weights/cnn_model_relabel.pt and weights/s2_model.pt; the scale
constant each was trained with lives alongside in the matching *_result.pkl.
"""
import torch
import torch.nn as nn


class SCGNet(nn.Module):
    """Main confirm classifier. cutoff = constants.CUTOFF_MAIN."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=15, padding=7)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=9, padding=4)
        self.pool = nn.MaxPool1d(4)
        self.drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(16 * 10, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.flatten(1)
        x = self.drop(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class S2Net(nn.Module):
    """S2-recovery classifier. Identical trunk to SCGNet, only dropout differs
    (0.5 vs 0.3). cutoff = constants.S2_CUTOFF.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=15, padding=7)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=9, padding=4)
        self.pool = nn.MaxPool1d(4)
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16 * 10, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.flatten(1)
        x = self.drop(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SCGNetFused(nn.Module):
    """PROPOSED, NOT YET TRAINED (see docs/PIPELINE_SUMMARY.md Section 9.6).

    Same conv trunk as SCGNet; concatenates n_extra auxiliary NCC template-
    match score(s) after flatten+dropout, before fc1. Kept here as a ready-to-
    train scaffold for the planned NCC-feature-fusion overfitting mitigation --
    do not assume this has been validated.
    """

    def __init__(self, n_extra=1):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=15, padding=7)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=9, padding=4)
        self.pool = nn.MaxPool1d(4)
        self.drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(16 * 10 + n_extra, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x, extra):
        # x: (B,1,L) raw env_sd snippet;  extra: (B, n_extra) e.g. [ncc_score]
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.flatten(1)
        x = self.drop(x)
        x = torch.cat([x, extra], dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
