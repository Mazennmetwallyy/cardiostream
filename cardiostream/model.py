"""Beat classifier: a small residual 1D-CNN over the waveform, fused with
RR-interval rhythm features. ~90k parameters, so CPU inference is well
under a millisecond per beat."""

from __future__ import annotations

import torch
from torch import nn

from .signal import CLASSES, N_RR, WINDOW


class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int, stride: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, stride, k // 2, bias=False),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv1d(c_out, c_out, k, 1, k // 2, bias=False),
            nn.BatchNorm1d(c_out),
        )
        self.skip = (
            nn.Identity()
            if c_in == c_out and stride == 1
            else nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm1d(c_out))
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


class BeatNet(nn.Module):
    def __init__(self, n_classes: int = len(CLASSES), n_rr: int = N_RR, dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 16, 7, 1, 3, bias=False), nn.BatchNorm1d(16), nn.ReLU(inplace=True)
        )
        self.blocks = nn.Sequential(ResBlock(16, 32, 7, 2), ResBlock(32, 48, 5, 2), ResBlock(48, 64, 5, 2))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.rr = nn.Sequential(nn.Linear(n_rr, 16), nn.ReLU(inplace=True))
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(64 + 16, 64), nn.ReLU(inplace=True), nn.Linear(64, n_classes)
        )

    def forward(self, beat: torch.Tensor, rr: torch.Tensor) -> torch.Tensor:
        z = self.pool(self.blocks(self.stem(beat))).flatten(1)
        return self.head(torch.cat([z, self.rr(rr)], dim=1))


def example_inputs(batch: int = 2):
    return torch.randn(batch, 1, WINDOW), torch.rand(batch, N_RR) + 0.5
