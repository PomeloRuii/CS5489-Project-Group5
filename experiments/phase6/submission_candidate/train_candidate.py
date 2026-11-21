"""Reproduce the frozen training run from the course archive or its 256 cache."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from project import CachedMaskDataset, SmallUNet, run_epoch


def datasets(cache, split_path):
    split = json.loads(split_path.read_text(encoding="utf-8"))
    index = pd.read_csv(cache / "images.csv", dtype={"ID": str})
    rows = {identifier: row for row, identifier in enumerate(index.ID)}
    assert len(rows) == len(index) == 5000
    assert len(split["train"]) == 4500 and len(split["validation"]) == 500
    assert set(split["train"]).isdisjoint(split["validation"])
    assert set(split["train"]) | set(split["validation"]) == set(rows)
    return tuple(CachedMaskDataset(cache, [rows[i] for i in split[key]])
                 for key in ("train", "validation"))


def weighted_epoch(model, loader, device, optimizer, weight):
    model.train()
    total = 0.
    for images, truth in loader:
        images, truth = images.to(device), truth.to(device)
        optimizer.zero_grad()
        logits = model(images)
        probabilities = logits.sigmoid().flatten(1)
        target = truth.flatten(1)
        overlap = (probabilities * target).sum(dim=1)
        dice = (2 * overlap + 1) / (probabilities.sum(1) + target.sum(1) + 1)
        loss = F.binary_cross_entropy_with_logits(logits, truth, pos_weight=weight) + 1 - dice.mean()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(images)
    return total / len(loader.dataset)


def train(cache, split_path, out, model_factory=SmallUNet, weighted=True, model_details=None):
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, val_set = datasets(cache, split_path)
    if weighted:
        foreground = sum(np.count_nonzero(train_set.masks[i]) for i in train_set.indices)
        weight = math.sqrt((len(train_set) * 256**2 - foreground) / foreground)
        positive_weight = torch.tensor(weight, device=device)
    out.mkdir(parents=True, exist_ok=False)
    seed, epochs, batch_size, lr = 5489, 10, 8, 1e-3
    torch.manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = model_factory().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    config = {
        "model": type(model).__name__, "model_details": model_details,
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "train_images": len(train_set), "validation_images": len(val_set),
        "input_size": 256, "seed": seed, "training_epochs": epochs,
        "batch_size": batch_size, "train_shuffle": True, "validation_shuffle": False,
        "dataloader_workers": 0, "cpu_threads": 4, "augmentation": "none",
        "optimizer": "Adam", "optimizer_arguments": optimizer.defaults,
        "scheduler": None, "precision": "float32", "pretrained_weights": False,
        "external_data": False, "input_processing": "RGB bilinear / 255, binary mask nearest",
        "train_loss": ("weighted BCE" if weighted else "unweighted BCE") + " + per-image soft Dice, smoothing 1",
        "val_loss": "unweighted BCE + per-image soft Dice, smoothing 1",
        "pos_weight": weight if weighted else None, "threshold": 0.5,
        "selection": "maximum 256-grid validation foreground micro-IoU, earliest tie",
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2), flush=True)
    history, best_iou, best_epoch = [], -1., None
    for epoch in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        if weighted:
            loss = weighted_epoch(model, train_loader, device, optimizer, positive_weight)
        else:
            loss = run_epoch(model, train_loader, device, optimizer)
        val_loss, iou = run_epoch(model, val_loader, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        row = {"epoch": epoch, "train_loss": loss, "val_loss": val_loss, "val_iou": iou,
               "seconds": time.perf_counter() - start}
        history.append(row)
        print(row, flush=True)
        if iou > best_iou:
            best_iou, best_epoch = iou, epoch
            torch.save({"model_state": model.state_dict(), "image_size": 256}, out / "model.pth")
        pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
    summary = {"best_epoch": best_epoch, "best_val_iou_256": best_iou,
               "training_seconds": sum(row["seconds"] for row in history),
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("cache256"))
    parser.add_argument("--out", type=Path, default=Path("reproduction"))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if not all((args.cache / filename).is_file() for filename in ("images.csv", "images.npy", "masks.npy")):
        from prepare_data import prepare_data
        prepare_data(args.archive, args.cache, size=256)
    train(args.cache, Path(__file__).with_name("split.json"), args.out)
