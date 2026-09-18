"""
SCGNet / S2Net, copied verbatim from PIPELINE_SUMMARY.md Sections 1.Stage2/1.Stage3.

Deliberately unchanged: the whole point of the teacher/auto-labeler is that the deployed
MCU architecture never has to change, only the origin of its training labels does.
"""
from __future__ import annotations

import torch
import torch.nn as nn

HALF_N = 80  # snippet is env_sd[i-80 : i+81]


class SCGNet(nn.Module):
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
    # identical trunk to SCGNet, only the dropout differs (0.5 vs 0.3)
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


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
