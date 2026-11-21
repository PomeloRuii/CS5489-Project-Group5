import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "baseline"
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
from project import SmallUNet


class FixedSRM(nn.Module):
    """Three zero-sum filters from RGB-N Fig. 4, applied to luminance."""

    def __init__(self):
        super().__init__()
        kernels = torch.tensor([
            [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0],
             [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]],
            [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
             [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0],
             [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        ], dtype=torch.float32)
        divisors = torch.tensor([4., 12., 2.]).view(3, 1, 1, 1)
        self.register_buffer("kernels", kernels[:, None] / divisors, persistent=False)
        self.register_buffer("luma", torch.tensor([.299, .587, .114]).view(1, 3, 1, 1),
                             persistent=False)

    def raw(self, rgb):
        luminance = (rgb * self.luma).sum(dim=1, keepdim=True)
        return F.conv2d(F.pad(luminance, (2, 2, 2, 2), mode="reflect"), self.kernels)

    @staticmethod
    def normalize(residual):
        return (residual * (255 / 3)).clamp(-1, 1)

    def forward(self, rgb):
        return self.normalize(self.raw(rgb))


class SRMUNet(SmallUNet):
    """Retain V1 and extend only its first convolution from 3 to 6 channels."""

    def __init__(self):
        super().__init__()
        initial_rgb = self.encoder1[0]
        # Preserve V1's CPU RNG state and therefore its shuffled training order.
        with torch.random.fork_rng(devices=[]):
            expanded = nn.Conv2d(6, 16, 3, padding=1)
        with torch.no_grad():
            expanded.weight[:, :3].copy_(initial_rgb.weight)
            expanded.bias.copy_(initial_rgb.bias)
        self.encoder1[0] = expanded
        self.srm = FixedSRM()

    def forward(self, rgb):
        features = torch.cat((rgb, self.srm(rgb)), dim=1)
        return super().forward(features)
