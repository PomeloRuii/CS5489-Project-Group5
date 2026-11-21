import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from diagnose import BASE, CACHE, HERE, snapshot_v1
from project import CachedMaskDataset, SmallUNet, run_epoch, segmentation_loss


def weighted_loss(logits, targets, positive_weight):
    probabilities = logits.sigmoid().flatten(1)
    foreground = targets.flatten(1)
    overlap = (probabilities * foreground).sum(dim=1)
    dice = (2 * overlap + 1) / (probabilities.sum(1) + foreground.sum(1) + 1)
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=positive_weight)
    return bce + 1 - dice.mean()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def train():
    snapshot_v1(HERE)
    out = HERE / "weighted"
    out.mkdir(exist_ok=True)
    if (out / "training_history.csv").exists():
        raise FileExistsError("Phase 2 training history already exists")
    torch.set_num_threads(4)
    device = torch.device("cuda")
    assert torch.cuda.is_available()
    seed, epochs, batch_size, lr, image_size = 5489, 10, 8, 0.001, 256
    torch.manual_seed(seed)
    split = json.loads((BASE / "split.json").read_text())
    image_index = pd.read_csv(CACHE / "images.csv", dtype={"ID": str})
    index = {identifier: i for i, identifier in enumerate(image_index.ID)}
    train_indices = [index[i] for i in split["train"]]
    val_indices = [index[i] for i in split["validation"]]
    train_set, val_set = CachedMaskDataset(CACHE, train_indices), CachedMaskDataset(CACHE, val_indices)
    pixel_total = len(train_set) * image_size**2
    foreground_pixels = sum(np.count_nonzero(train_set.masks[i]) for i in train_indices)
    # Temper inverse prevalence using only training labels.
    weight = math.sqrt((pixel_total - foreground_pixels) / foreground_pixels)
    config = {
        "model": "baseline/project.py:SmallUNet", "initialization": "random, default PyTorch",
        "external_data": False, "pretrained_weights": False, "seed": seed,
        "split": "../baseline/split.json", "train_images": len(train_set),
        "validation_images": len(val_set), "input_size": image_size,
        "training_epochs": epochs, "batch_size": batch_size, "shuffle_train": True,
        "shuffle_validation": False, "dataloader_workers": 0, "cpu_threads": 4,
        "optimizer": "Adam", "optimizer_arguments": {
            "lr": lr, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0,
        },
        "input_processing": "bilinear RGB resize, divide by 255; nearest-neighbor binary mask resize",
        "augmentation": "none", "precision": "float32 (no AMP)",
        "training_loss": "positive-weighted BCEWithLogits + unchanged per-image soft Dice loss",
        "dice_smoothing": 1, "train_fg_ratio": foreground_pixels / pixel_total,
        "bce_pos_weight": weight, "weight_rule": "sqrt(training background pixels / foreground pixels)",
        "validation_loss": "unweighted BCEWithLogits + soft Dice, same as V1",
        "selection_metric": "micro foreground IoU at 256x256 with threshold 0.5",
        "threshold": 0.5, "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "python": sys.version.split()[0],
        "runtime_scope": "train and validation passes, synchronized, excluding preprocessing and file writes",
        "peak_memory_scope": "torch.cuda.max_memory_allocated, not total device VRAM",
    }
    save_json(out / "config.json", config)
    print(f"Training BCE pos_weight={weight:.6f}.", flush=True)

    # V1 did not log memory. Measure two of its training steps on a temporary model.
    torch.cuda.reset_peak_memory_stats(device)
    probe = SmallUNet().to(device)
    probe_optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    probe_loader = iter(DataLoader(train_set, batch_size=batch_size, shuffle=False))
    for _ in range(2):
        x, target = (t.to(device) for t in next(probe_loader))
        probe_optimizer.zero_grad()
        loss = segmentation_loss(probe(x), target)
        loss.backward()
        probe_optimizer.step()
    torch.cuda.synchronize()
    baseline_peak = torch.cuda.max_memory_allocated(device)
    del probe, probe_optimizer, probe_loader, x, target, loss
    torch.cuda.empty_cache()
    save_json(HERE / "v1" / "memory_probe.json", {
        "method": "two 8-image Adam steps on a temporary V1 architecture at 256x256, including Adam state",
        "peak_allocated_bytes": baseline_peak, "historical_peak_available": False,
    })

    # Restore the seed after the temporary memory probe to match the V1 run.
    torch.manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(val_set, batch_size=batch_size)
    torch.cuda.reset_peak_memory_stats(device)
    model = SmallUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    positive_weight = torch.tensor(weight, device=device)
    history, best_iou, best_epoch = [], -1.0, None
    for epoch in range(1, epochs + 1):
        torch.cuda.synchronize()
        start = time.perf_counter()
        model.train()
        loss_sum = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = weighted_loss(logits, targets, positive_weight)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(images)
        del images, targets, logits, loss
        val_loss, val_iou = run_epoch(model, validation_loader, device)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        history.append({
            "epoch": epoch, "train_weighted_loss": loss_sum / len(train_set),
            "val_loss": val_loss, "val_iou": val_iou, "seconds": seconds,
        })
        print(f"Epoch {epoch}/{epochs}: val loss {val_loss:.4f}, val IoU {val_iou:.4f}, {seconds:.1f}s", flush=True)
        if val_iou > best_iou:
            best_iou, best_epoch = val_iou, epoch
            torch.save({"model_state": model.state_dict(), "image_size": image_size}, out / "model.pth")
        pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
    peak = torch.cuda.max_memory_allocated(device)
    save_json(out / "training_summary.json", {
        "best_epoch": best_epoch, "best_val_iou_256": best_iou,
        "training_seconds": sum(row["seconds"] for row in history),
        "peak_allocated_bytes": peak,
    })

    v1 = pd.read_csv(BASE / "training_history.csv")
    v2 = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), layout="constrained")
    axes[0].plot(v1.epoch, v1.train_loss, "o-", label="V1 unweighted")
    axes[0].plot(v2.epoch, v2.train_weighted_loss, "o-", label="Weighted BCE")
    axes[0].set_ylabel("Training loss (different objectives)")
    for ax, column, label in zip(axes[1:], ("val_loss", "val_iou"),
                                 ("Shared validation loss", "Validation IoU (256×256)")):
        ax.plot(v1.epoch, v1[column], "o-", label="V1")
        ax.plot(v2.epoch, v2[column], "o-", label="Weighted BCE")
        ax.set_ylabel(label)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=.25)
        ax.legend()
    fig.savefig(out / "training_curves.png", dpi=150)
    plt.close(fig)
    snapshot_v1(HERE)


if __name__ == "__main__":
    train()
