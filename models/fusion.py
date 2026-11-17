"""SSMA block, independently implemented from Valada et al., Sec. 4.1.

Reference: https://arxiv.org/abs/1808.03833 (v3), Eqs. 10--12.
One 3x3 bottleneck gate at the encoder output; this is NOT AdapNet++.
The ungated control retains dormant gate weights for exactly paired common
initialization. Dormant weights are excluded from optimization and active counts.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .residual import Residual


class Fusion(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(2*channels, max(1, channels//reduction), 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(max(1, channels//reduction), 2*channels, 3, padding=1),
            nn.Sigmoid())
        self.project = nn.Sequential(nn.Conv2d(2*channels, channels, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(channels), nn.ReLU())

    def forward(self, rgb, depth, use_gate=True, return_gate=False):
        joined = torch.cat([rgb, depth], dim=1)
        gate = self.gate(joined) if use_gate else torch.ones_like(joined)
        out = self.project(joined*gate)
        return (out, gate) if return_gate else out


def encoder(inputs, channels, stage_widths=None):
    first, second, third = stage_widths or (channels, channels * 2, channels * 4)
    return nn.Sequential(
        nn.Conv2d(inputs, first, 9, padding=4, bias=False),
        nn.BatchNorm2d(first), nn.ReLU(),
        nn.Conv2d(first, second, 4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(second), nn.ReLU(),
        nn.Conv2d(second, third, 4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(third), nn.ReLU(),
    )


class DualGraspNet(nn.Module):
    def __init__(self, channels=32, residual_blocks=5, use_gate=True, early_fusion=False):
        super().__init__()
        self.use_gate = use_gate
        self.early_fusion = early_fusion
        if early_fusion:
            if use_gate:
                raise ValueError("Early fusion does not support gating")
            # Match the active capacity of the two encoders and fusion projection.
            widths = (48, 96, 202)
            self.rgbd_encoder = encoder(4, channels, widths)
            self.projection = nn.Sequential(
                nn.Conv2d(widths[-1], channels * 4, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels * 4), nn.ReLU(),
            )
        else:
            self.rgb_encoder = encoder(3, channels)
            self.depth_encoder = encoder(1, channels)
            self.fusion = Fusion(channels * 4)
        self.residual = nn.Sequential(*(Residual(channels*4) for _ in range(residual_blocks)))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels*4, channels*2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels*2), nn.ReLU(),
            nn.ConvTranspose2d(channels*2, channels, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU())
        self.head = nn.Conv2d(channels, 4, 3, padding=1)
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        if not use_gate and not early_fusion:
            self.fusion.gate.requires_grad_(False)

    def forward(self, x, return_gate=False):
        if self.early_fusion:
            if return_gate:
                raise ValueError("Early fusion has no gate to return")
            fused = self.projection(self.rgbd_encoder(x))
        else:
            fused, gate = self.fusion(self.rgb_encoder(x[:, :3]), self.depth_encoder(x[:, 3:]),
                                      self.use_gate, return_gate=True)
        out = self.head(self.decoder(self.residual(fused)))
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return (out, gate) if return_gate else out
