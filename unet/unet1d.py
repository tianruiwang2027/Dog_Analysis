"""
A minimal, from-scratch 1D U-Net for time-series/segmentation tasks, sized for
SCG/ECG beat-and-fiducial-point segmentation of the kind described in the recent
literature (e.g. PMC13210897, the Frontiers 2025 beat-to-beat delineation paper).

Not wired into the rest of this project -- this is a standalone, documented
reference implementation to adapt, not a drop-in replacement for ShapeMIL. See
the module docstring at the bottom (`build_peak_proximity_mask`) for how you'd
turn Chelten's existing point-click labels into the per-sample training targets
a segmentation network like this actually needs.

Architecture (depth=4 by default):
    input (B, C_in, L)
      -> encoder stage 1 (channels: C_in -> 32)   -- kept for skip connection
      -> downsample (maxpool /2)
      -> encoder stage 2 (32 -> 64)                -- kept for skip connection
      -> downsample (maxpool /2)
      -> encoder stage 3 (64 -> 128)                -- kept for skip connection
      -> downsample (maxpool /2)
      -> encoder stage 4 (128 -> 256)               -- kept for skip connection
      -> downsample (maxpool /2)
      -> bottleneck (256 -> 512)
      -> upsample /2 -> concat skip 4 -> decoder stage 4 (512+256 -> 256)
      -> upsample /2 -> concat skip 3 -> decoder stage 3 (256+128 -> 128)
      -> upsample /2 -> concat skip 2 -> decoder stage 2 (128+64  -> 64)
      -> upsample /2 -> concat skip 1 -> decoder stage 1 (64+32   -> 32)
      -> 1x1 conv -> (B, n_classes, L)   <- SAME length as the input
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two (Conv1d -> BatchNorm -> ReLU) layers, padding chosen so the temporal
    length is unchanged. This is the basic repeated unit at every U-Net stage."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 9):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


def _match_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Skip connections can be off by 1-2 samples after repeated /2 and *2
    resizing when the input length isn't a clean multiple of 2**depth. This
    center-crops or pads x so it lines up exactly with target_len before
    concatenation -- a standard, unglamorous U-Net implementation detail."""
    diff = x.shape[-1] - target_len
    if diff > 0:
        left = diff // 2
        return x[..., left:left + target_len]
    elif diff < 0:
        pad = -diff
        left = pad // 2
        right = pad - left
        return F.pad(x, (left, right))
    return x


class UNet1D(nn.Module):
    """
    in_channels: number of input signal channels (e.g. 1 for SCG alone, 2 for
        SCG+ECG stacked as separate channels).
    n_classes: number of per-sample output classes. Use 1 (with BCEWithLogits)
        for binary "is this sample near a real peak" segmentation -- the
        practical starting point when you only have point-click labels like
        Chelten's, rather than full multi-fiducial-point interval labels.
    base_channels: channel count at the first encoder stage; doubles each
        stage down. 32 is a reasonable default; shrink it (e.g. 16) for a
        lighter model if you eventually want to distill this down toward
        MCU-sized, the way scgnet_distilled_chelten.pt was distilled from
        the MIL teacher earlier in this project.
    depth: number of downsample/upsample stages (4 matches the ~5-stage
        encoders used in the SCG U-Net papers cited in this project's
        literature review).
    """

    def __init__(self, in_channels: int = 1, n_classes: int = 1,
                 base_channels: int = 32, depth: int = 4, kernel_size: int = 9):
        super().__init__()
        self.depth = depth

        enc_channels = [base_channels * (2 ** i) for i in range(depth)]
        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for ch in enc_channels:
            self.encoders.append(ConvBlock(prev_ch, ch, kernel_size))
            prev_ch = ch
        self.pool = nn.MaxPool1d(2)

        self.bottleneck = ConvBlock(prev_ch, prev_ch * 2, kernel_size)

        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        dec_in = prev_ch * 2
        for ch in reversed(enc_channels):
            self.upsamples.append(nn.ConvTranspose1d(dec_in, ch, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(ch + ch, ch, kernel_size))  # +ch from skip concat
            dec_in = ch

        self.out_conv = nn.Conv1d(enc_channels[0], n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, in_channels, length) -> (batch, n_classes, length)"""
        skips = []
        h = x
        for enc in self.encoders:
            h = enc(h)
            skips.append(h)
            h = self.pool(h)

        h = self.bottleneck(h)

        for up, dec, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            h = up(h)
            h = _match_length(h, skip.shape[-1])
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        h = _match_length(h, x.shape[-1])
        return self.out_conv(h)


def build_peak_proximity_mask(length: int, click_indices, tol_samples: int) -> torch.Tensor:
    """The practical bridge from point-click labels (what Chelten's annotation
    files actually give you) to the per-sample segmentation target a U-Net
    needs. Returns a (length,) float tensor: 1.0 within tol_samples of any
    real click, 0.0 (background) everywhere else -- directly analogous to the
    +-60ms matching tolerance already used in teacher/finetune_chelten.py, just
    applied to every sample instead of only to Stage-1 candidate positions.

    Class imbalance warning: with tol_samples set to e.g. 60ms at 800Hz sampling
    (fs=800 -> ~48 samples either side of each click), the positive class will
    still be a small fraction of an 85-minute recording. Use
    nn.BCEWithLogitsLoss(pos_weight=...) or a Dice/Tversky loss rather than
    plain unweighted BCE, or the network will trivially learn to predict
    "background everywhere."""
    mask = torch.zeros(length)
    for idx in click_indices:
        lo = max(0, int(idx) - tol_samples)
        hi = min(length, int(idx) + tol_samples + 1)
        mask[lo:hi] = 1.0
    return mask


if __name__ == "__main__":
    # Smoke test: confirm output shape matches input length for both an
    # exact-power-of-2 length and an awkward one (exercising _match_length).
    for in_ch, n_cls, L in [(1, 1, 4096), (2, 1, 4001), (1, 5, 8192)]:
        model = UNet1D(in_channels=in_ch, n_classes=n_cls, base_channels=16, depth=4)
        x = torch.randn(2, in_ch, L)
        y = model(x)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"in_ch={in_ch} n_classes={n_cls} L={L} -> out shape {tuple(y.shape)} "
              f"({n_params:,} params)")
        assert y.shape == (2, n_cls, L), "output length must match input length"
    print("OK")
