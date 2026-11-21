from itertools import product

import numpy as np
import torch

SIZE = 256
STRIDE = 192
INFERENCE_BATCH = 32


def pad_image(rgb, mask=None):
    h, w = rgb.shape[:2]
    dh, dw = max(SIZE - h, 0), max(SIZE - w, 0)
    pad = ((dh // 2, dh - dh // 2), (dw // 2, dw - dw // 2))
    if dh or dw:
        rgb = np.pad(rgb, (*pad, (0, 0)), mode="reflect" if min(h, w) > 1 else "edge")
        if mask is not None:
            mask = np.pad(mask, pad, mode="constant")
    return rgb, mask, (pad[0][0], pad[1][0])


def starts(length):
    last = length - SIZE
    anchors = list(range(0, last + 1, STRIDE))
    if anchors[-1] != last:
        anchors.append(last)
    return anchors


@torch.inference_mode()
def sliding_probability(rgb, predict_batch, device):
    """predict_batch takes BCHW float RGB and returns BHW probabilities."""
    original_h, original_w = rgb.shape[:2]
    rgb, _, (top, left) = pad_image(rgb)
    h, w = rgb.shape[:2]
    source = torch.from_numpy(np.array(rgb, copy=True)).to(device=device, dtype=torch.float32)
    source = source.permute(2, 0, 1).div_(255)
    y_starts, x_starts = starts(h), starts(w)
    coords = list(product(y_starts, x_starts))
    w1 = torch.hann_window(SIZE, periodic=False, device=device).clamp_min(.10)
    weight = w1[:, None] * w1[None, :]
    wy, wx = torch.zeros(h, device=device), torch.zeros(w, device=device)
    for y in y_starts:
        wy[y:y + SIZE].add_(w1)
    for x in x_starts:
        wx[x:x + SIZE].add_(w1)
    canvas = torch.zeros((h, w), device=device)
    for start in range(0, len(coords), INFERENCE_BATCH):
        batch = coords[start:start + INFERENCE_BATCH]
        patches = torch.stack([source[:, y:y + SIZE, x:x + SIZE] for y, x in batch])
        probs = predict_batch(patches)
        for p, (y, x) in zip(probs, batch):
            canvas[y:y + SIZE, x:x + SIZE].add_(p * weight)
    canvas.div_(wy[:, None]).div_(wx[None, :])
    native = canvas[top:top + original_h, left:left + original_w].cpu().numpy()
    return np.clip(native, 0, 1), len(coords)


@torch.inference_mode()
def predict_binary_mask(model, image, device):
    p, _ = sliding_probability(np.asarray(image.convert("RGB")), lambda x: model(x)[:, 0].sigmoid(), device)
    return (p >= .5).astype(np.uint8)
