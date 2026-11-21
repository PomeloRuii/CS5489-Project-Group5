from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset


EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def decode_rle(rle, height, width):
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


def read_records(image_dir, labels_csv, id_column, rle_column):
    paths = {}
    for path in Path(image_dir).iterdir():
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            if path.stem in paths:
                raise ValueError(f"Duplicate image ID {path.stem!r}")
            paths[path.stem] = path
    table = pd.read_csv(labels_csv, dtype=str, keep_default_na=False)
    records = []
    seen = set()
    for identifier, rle in table[[id_column, rle_column]].itertuples(index=False, name=None):
        name = Path(identifier)
        key = name.stem if name.suffix.lower() in EXTENSIONS else identifier
        if key in seen:
            raise ValueError(f"Duplicate annotation ID {key!r}")
        if key not in paths:
            raise ValueError(f"Missing image for annotation ID {key!r}")
        seen.add(key)
        records.append((paths[key], rle))
    if paths.keys() - seen:
        raise ValueError(f"Images without annotations: {sorted(paths.keys() - seen)[:5]}")
    return sorted(records)


def load_pair(record):
    path, rle = record
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    try:
        mask = decode_rle(rle, rgb.height, rgb.width)
    except ValueError as error:
        raise ValueError(f"Invalid annotation for {path.name}: {error}") from error
    return rgb, mask


def image_tensor(image, size):
    resized = image.resize((size, size), Image.Resampling.BILINEAR)
    pixels = np.array(resized, dtype=np.uint8)
    return torch.from_numpy(pixels).permute(2, 0, 1).float().div_(255)


class MaskDataset(Dataset):
    def __init__(self, records, size):
        self.records = records
        self.size = size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image, mask = load_pair(self.records[index])
        resized = Image.fromarray(mask).resize(
            (self.size, self.size), Image.Resampling.NEAREST
        )
        target = torch.from_numpy(np.array(resized)).unsqueeze(0).float()
        return image_tensor(image, self.size), target


class CachedMaskDataset(Dataset):
    def __init__(self, cache, indices):
        self.images = np.load(cache / "images.npy", mmap_mode="c")
        self.masks = np.load(cache / "masks.npy", mmap_mode="c")
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        image = torch.from_numpy(self.images[i]).permute(2, 0, 1).float().div_(255)
        target = torch.from_numpy(self.masks[i]).unsqueeze(0).float()
        return image, target


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class SmallUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = conv_block(3, 16)
        self.encoder2 = conv_block(16, 32)
        self.bridge = conv_block(32, 64)
        self.decoder2 = conv_block(64 + 32, 32)
        self.decoder1 = conv_block(32 + 16, 16)
        self.output = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        first = self.encoder1(x)
        second = self.encoder2(F.max_pool2d(first, 2))
        x = self.bridge(F.max_pool2d(second, 2))
        x = F.interpolate(x, size=second.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decoder2(torch.cat((x, second), dim=1))
        x = F.interpolate(x, size=first.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decoder1(torch.cat((x, first), dim=1))
        return self.output(x)


def segmentation_loss(logits, targets):
    probabilities = logits.sigmoid().flatten(1)
    foreground = targets.flatten(1)
    overlap = (probabilities * foreground).sum(dim=1)
    dice = (2 * overlap + 1) / (probabilities.sum(1) + foreground.sum(1) + 1)
    return F.binary_cross_entropy_with_logits(logits, targets) + 1 - dice.mean()


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = 0
    tp = fp = fn = 0
    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            if training:
                optimizer.zero_grad()
            logits = model(images)
            loss = segmentation_loss(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * len(images)
            if not training:
                guessed = logits >= 0
                actual = targets.bool()
                tp += (guessed & actual).sum().item()
                fp += (guessed & ~actual).sum().item()
                fn += (~guessed & actual).sum().item()
    average = loss_sum / len(loader.dataset)
    if training:
        return average
    union = tp + fp + fn
    return average, tp / union if union else 1.0


@torch.inference_mode()
def predict_mask(model, image, device, size):
    inputs = image_tensor(image, size).unsqueeze(0).to(device)
    logits = F.interpolate(
        model(inputs), size=(image.height, image.width),
        mode="bilinear", align_corners=False,
    )
    return (logits[0, 0] >= 0).cpu().numpy().astype(np.uint8)
