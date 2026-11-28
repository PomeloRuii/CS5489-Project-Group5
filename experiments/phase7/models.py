"""Pretrained multi-scale segmentation architectures for CS5489 Phase 7.

Uses only PyTorch and torchvision (matching the TA reference environment).
All models register ImageNet normalization buffers internally and accept
RGB float tensors in [0, 1] of shape (B, 3, H, W), outputting logits of shape (B, 1, H, W).
At inference time, instantiate with pretrained=False and load model.pth completely offline.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as tv_models


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def decode_rle(rle: str, height: int, width: int) -> np.ndarray:
    """Decode 1-based start/length pairs in column-major order."""
    tokens = str(rle).split()
    if len(tokens) % 2:
        raise ValueError(f"Expected start/length pairs, got {rle!r}")
    pixels = np.zeros(height * width, dtype=np.uint8)
    previous_end = 0
    for start, length in zip(tokens[::2], tokens[1::2]):
        begin = int(start) - 1
        end = begin + int(length)
        if begin < previous_end or end <= begin or end > pixels.size:
            raise ValueError(f"Invalid RLE run {start} {length} for {height}x{width}")
        pixels[begin:end] = 1
        previous_end = end
    return pixels.reshape((height, width), order="F")


def srm_kernels() -> torch.Tensor:
    """Three standard 5x5 SRM high-pass residual kernels applied to luminance."""
    k1 = np.array([
        [0, 0, 0, 0, 0],
        [0, -1, 2, -1, 0],
        [0, 2, -4, 2, 0],
        [0, -1, 2, -1, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.float32) / 4.0
    k2 = np.array([
        [-1, 2, -2, 2, -1],
        [2, -6, 8, -6, 2],
        [-2, 8, -12, 8, -2],
        [2, -6, 8, -6, 2],
        [-1, 2, -2, 2, -1],
    ], dtype=np.float32) / 12.0
    k3 = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, -2, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.float32) / 2.0
    # Apply to RGB luminance (0.299, 0.587, 0.114) -> shape (3, 3, 5, 5)
    luma = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    kernels = np.stack([k1, k2, k3], axis=0)  # (3, 5, 5)
    weight = kernels[:, None, :, :] * luma[None, :, None, None]  # (3, 3, 5, 5)
    return torch.from_numpy(weight)


class SRMResidual(nn.Module):
    """Fixed high-pass forensic residual extractor with truncated activation."""

    def __init__(self, clip_value: float = 3.0):
        super().__init__()
        self.register_buffer("weight", srm_kernels())
        self.clip_value = clip_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is RGB in [0, 1]; scale to [0, 255] like classic SRM then truncate & scale
        res = F.conv2d(x * 255.0, self.weight, padding=2)
        return res.clamp(-self.clip_value, self.clip_value) / self.clip_value


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.block(x)


class ResNet34UNet(nn.Module):
    """Pretrained ResNet-34 encoder with full-resolution U-Net decoder."""

    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = tv_models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.resnet34(weights=weights)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # stride 2, 64
        self.pool = backbone.maxpool
        self.enc1 = backbone.layer1  # stride 4, 64
        self.enc2 = backbone.layer2  # stride 8, 128
        self.enc3 = backbone.layer3  # stride 16, 256
        self.enc4 = backbone.layer4  # stride 32, 512

        self.dec3 = DecoderBlock(512, 256, 256, dropout=0.1)
        self.dec2 = DecoderBlock(256, 128, 128, dropout=0.05)
        self.dec1 = DecoderBlock(128, 64, 64)
        self.dec0 = DecoderBlock(64, 64, 32)
        self.head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def backbone_parameters(self):
        for m in (self.stem, self.enc1, self.enc2, self.enc3, self.enc4):
            yield from m.parameters()

    def decoder_parameters(self):
        for m in (self.dec3, self.dec2, self.dec1, self.dec0, self.head):
            yield from m.parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        xn = (x - self.mean) / self.std
        s0 = self.stem(xn)           # H/2
        s1 = self.enc1(self.pool(s0))  # H/4
        s2 = self.enc2(s1)           # H/8
        s3 = self.enc3(s2)           # H/16
        s4 = self.enc4(s3)           # H/32

        d3 = self.dec3(s4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        d0 = self.dec0(d1, s0)
        d0 = F.interpolate(d0, size=(h, w), mode="bilinear", align_corners=False)
        return self.head(d0)


class ConvNeXtUNet(nn.Module):
    """Pretrained ConvNeXt (tiny or small) encoder + high-resolution detail/SRM stem + U-Net decoder."""

    def __init__(self, variant: str = "small", pretrained: bool = False, use_srm: bool = False):
        super().__init__()
        self.variant = variant
        self.use_srm = use_srm
        if variant == "tiny":
            weights = tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = tv_models.convnext_tiny(weights=weights)
            dims = (96, 192, 384, 768)
        elif variant == "small":
            weights = tv_models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = tv_models.convnext_small(weights=weights)
            dims = (96, 192, 384, 768)
        else:
            raise ValueError(f"Unsupported ConvNeXt variant: {variant}")

        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        # ConvNeXt features indices:
        # 0: stem (stride 4), 1: stage1 (96)
        # 2: down (stride 8), 3: stage2 (192)
        # 4: down (stride 16), 5: stage3 (384)
        # 6: down (stride 32), 7: stage4 (768)
        f = backbone.features
        self.stage1 = nn.Sequential(f[0], f[1])
        self.stage2 = nn.Sequential(f[2], f[3])
        self.stage3 = nn.Sequential(f[4], f[5])
        self.stage4 = nn.Sequential(f[6], f[7])

        # Parallel high-resolution spatial detail (+ optional SRM residual) branch
        self.srm = SRMResidual() if use_srm else None
        in_detail = 6 if use_srm else 3
        self.detail1 = nn.Sequential(
            nn.Conv2d(in_detail, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.Conv2d(24, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
        )
        self.detail2 = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
        )

        self.dec3 = DecoderBlock(dims[3], dims[2], 256, dropout=0.1)
        self.dec2 = DecoderBlock(256, dims[1], 128, dropout=0.05)
        self.dec1 = DecoderBlock(128, dims[0], 64)
        self.dec_half = DecoderBlock(64, 48, 48)
        self.dec_full = DecoderBlock(48, 24, 32)
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def backbone_parameters(self):
        for m in (self.stage1, self.stage2, self.stage3, self.stage4):
            yield from m.parameters()

    def decoder_parameters(self):
        for m in (self.detail1, self.detail2, self.dec3, self.dec2, self.dec1, self.dec_half, self.dec_full, self.head):
            yield from m.parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = (x - self.mean) / self.std
        s1 = self.stage1(xn)  # H/4
        s2 = self.stage2(s1)  # H/8
        s3 = self.stage3(s2)  # H/16
        s4 = self.stage4(s3)  # H/32

        detail_in = torch.cat([xn, self.srm(x)], dim=1) if self.srm is not None else xn
        f1 = self.detail1(detail_in)  # H, W
        f2 = self.detail2(f1)         # H/2, W/2

        d3 = self.dec3(s4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        dh = self.dec_half(d1, f2)
        df = self.dec_full(dh, f1)
        return self.head(df)


def build_architecture(arch: str, pretrained: bool = False) -> nn.Module:
    if arch == "resnet34_unet":
        return ResNet34UNet(pretrained=pretrained)
    if arch == "convnext_tiny_unet":
        return ConvNeXtUNet(variant="tiny", pretrained=pretrained, use_srm=False)
    if arch == "convnext_small_unet":
        return ConvNeXtUNet(variant="small", pretrained=pretrained, use_srm=False)
    if arch == "convnext_small_srm_unet":
        return ConvNeXtUNet(variant="small", pretrained=pretrained, use_srm=True)
    raise ValueError(f"Unknown architecture: {arch}")


def image_to_tensor(image: Image.Image, size: int) -> torch.Tensor:
    resized = image.resize((size, size), Image.Resampling.BILINEAR)
    pixels = np.array(resized, dtype=np.uint8)
    return torch.from_numpy(pixels).permute(2, 0, 1).float().div_(255.0)


@torch.inference_mode()
def predict_probability(
    model: nn.Module,
    image: Image.Image,
    device: torch.device,
    size: int,
    tta: bool = True,
) -> torch.Tensor:
    """Return full-resolution foreground probability map on device."""
    x = image_to_tensor(image, size).unsqueeze(0).to(device)
    logits = model(x)
    prob = logits.sigmoid()
    if tta:
        x_flip = torch.flip(x, dims=[-1])
        prob_flip = torch.flip(model(x_flip).sigmoid(), dims=[-1])
        prob = 0.5 * (prob + prob_flip)
    prob_full = F.interpolate(
        prob, size=(image.height, image.width), mode="bilinear", align_corners=False
    )
    return prob_full[0, 0]


@torch.inference_mode()
def predict_mask(
    model: nn.Module,
    image: Image.Image,
    device: torch.device,
    size: int,
    threshold: float = 0.5,
    tta: bool = True,
) -> np.ndarray:
    prob = predict_probability(model, image, device, size, tta=tta)
    return (prob >= threshold).cpu().numpy().astype(np.uint8)
