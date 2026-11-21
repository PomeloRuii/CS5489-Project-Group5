"""Extend V1 by one encoder/decoder scale with a matching skip connection."""
import sys
from pathlib import Path

import torch
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "baseline"
sys.path.insert(0, str(BASE))
from project import SmallUNet, conv_block


class DeepUNet(SmallUNet):
    def __init__(self):
        super().__init__()
        # The old bridge becomes the third skip. Preserve the shared initialization
        # and the CPU RNG state used by the default training DataLoader.
        with torch.random.fork_rng(devices=[]):
            self.deep_bridge = conv_block(64, 128)
            self.decoder3 = conv_block(128 + 64, 64)

    def forward(self, x):
        first = self.encoder1(x)
        second = self.encoder2(F.max_pool2d(first, 2))
        third = self.bridge(F.max_pool2d(second, 2))
        x = self.deep_bridge(F.max_pool2d(third, 2))
        for skip, decoder in ((third, self.decoder3), (second, self.decoder2), (first, self.decoder1)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat((x, skip), dim=1))
        return self.output(x)
