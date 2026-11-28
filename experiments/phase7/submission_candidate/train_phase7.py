"""Controlled training pipeline for CS5489 Phase 7 pretrained segmentation models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
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
from torch.utils.data import DataLoader, Dataset

from models import build_architecture


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AugmentedCacheDataset(Dataset):
    def __init__(self, cache_dir: Path, indices: list[int], train: bool = False):
        self.cache_dir = Path(cache_dir)
        self.indices = indices
        self.train = train
        self.images = None
        self.masks = None
        with open(self.cache_dir / "cache_info.json", encoding="utf-8") as f:
            self.image_size = int(json.load(f)["size"])

    def _ensure_open(self):
        if self.images is None:
            self.images = np.load(self.cache_dir / "images.npy", mmap_mode="r")
            self.masks = np.load(self.cache_dir / "masks.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        self._ensure_open()
        i = self.indices[idx]
        # Copy out of read-only memmap into contiguous tensors
        img = torch.from_numpy(np.array(self.images[i], copy=True)).permute(2, 0, 1).float().div_(255.0)
        msk = torch.from_numpy(np.array(self.masks[i], copy=True)).unsqueeze(0).float()

        if self.train:
            # 1. Horizontal flip (matches inference TTA)
            if random.random() < 0.5:
                img = torch.flip(img, dims=[2])
                msk = torch.flip(msk, dims=[2])

            # 2. Random resized crop / scale jitter (preserves upright prior)
            if random.random() < 0.6:
                _, h, w = img.shape
                scale = random.uniform(0.70, 0.98)
                ratio = random.uniform(0.90, 1.11)
                ch = min(h, max(32, int(round(h * math.sqrt(scale / ratio)))))
                cw = min(w, max(32, int(round(w * math.sqrt(scale * ratio)))))
                top = random.randint(0, h - ch)
                left = random.randint(0, w - cw)
                img = F.interpolate(
                    img[:, top : top + ch, left : left + cw].unsqueeze(0),
                    size=(h, w), mode="bilinear", align_corners=False,
                ).squeeze(0)
                msk = F.interpolate(
                    msk[:, top : top + ch, left : left + cw].unsqueeze(0),
                    size=(h, w), mode="nearest",
                ).squeeze(0)

            # 3. Mild photometric jitter (brightness, contrast, saturation)
            if random.random() < 0.5:
                b = random.uniform(0.88, 1.12)
                c = random.uniform(0.88, 1.12)
                s = random.uniform(0.88, 1.12)
                img = img * b
                mean_c = img.mean(dim=(1, 2), keepdim=True)
                img = (img - mean_c) * c + mean_c
                gray = (0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]).unsqueeze(0)
                img = (img - gray) * s + gray
                img = img.clamp_(0.0, 1.0)

        return img, msk


def compute_loss(logits: torch.Tensor, targets: torch.Tensor, loss_type: str = "bce_dice") -> torch.Tensor:
    probs = logits.sigmoid()
    p_flat = probs.flatten(1)
    t_flat = targets.flatten(1)

    # Per-image soft Dice (protects small-edit images)
    inter_img = (p_flat * t_flat).sum(dim=1)
    dice_img = (2.0 * inter_img + 1.0) / (p_flat.sum(dim=1) + t_flat.sum(dim=1) + 1.0)
    loss_dice_img = 1.0 - dice_img.mean()

    # Batch micro soft Dice (aligns with global micro-IoU)
    inter_micro = (p_flat * t_flat).sum()
    dice_micro = (2.0 * inter_micro + 1.0) / (p_flat.sum() + t_flat.sum() + 1.0)
    loss_dice_micro = 1.0 - dice_micro

    if loss_type == "bce_dice":
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        return 0.5 * bce + 0.25 * loss_dice_img + 0.25 * loss_dice_micro
    if loss_type == "focal_dice":
        bce_raw = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce_raw)
        focal = (((1.0 - pt) ** 2.0) * bce_raw).mean()
        return 0.5 * focal + 0.25 * loss_dice_img + 0.25 * loss_dice_micro
    raise ValueError(f"Unsupported loss_type: {loss_type}")


@torch.inference_mode()
def evaluate_grid(model: torch.nn.Module, loader: DataLoader, device: torch.device, use_bf16: bool) -> dict:
    model.eval()
    loss_sum = 0.0
    tp = fp = fn = 0
    total_pixels = 0
    img_ious = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(images)
            loss = compute_loss(logits.float(), targets, "bce_dice")
        loss_sum += loss.item() * len(images)
        pred = logits >= 0.0
        truth = targets >= 0.5
        tp_b = (pred & truth).sum(dim=(1, 2, 3))
        fp_b = (pred & ~truth).sum(dim=(1, 2, 3))
        fn_b = (~pred & truth).sum(dim=(1, 2, 3))
        union_b = tp_b + fp_b + fn_b
        iou_b = torch.where(union_b > 0, tp_b.float() / union_b.clamp(min=1).float(), torch.ones_like(tp_b, dtype=torch.float32))
        img_ious.extend(iou_b.cpu().tolist())
        tp += int(tp_b.sum().item())
        fp += int(fp_b.sum().item())
        fn += int(fn_b.sum().item())
        total_pixels += int(truth.numel())

    union = tp + fp + fn
    micro_iou = tp / union if union else 1.0
    micro_dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    pred_fg = (tp + fp) / total_pixels
    return {
        "val_loss": loss_sum / len(loader.dataset),
        "val_micro_iou": micro_iou,
        "val_mean_img_iou": float(np.mean(img_ious)),
        "val_micro_dice": micro_dice,
        "val_precision": precision,
        "val_recall": recall,
        "val_pred_fg": pred_fg,
    }


def save_curves(history_df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), layout="constrained")
    axes[0].plot(history_df["epoch"], history_df["train_loss"], "o-", markersize=3, label="Train")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], "o-", markersize=3, label="Val (BCE+Dice)")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title=f"{title}: Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(history_df["epoch"], history_df["val_micro_iou"] * 100, "o-", markersize=3, label="Micro-IoU")
    axes[1].plot(history_df["epoch"], history_df["val_mean_img_iou"] * 100, "s--", markersize=3, label="Mean Image IoU")
    axes[1].set(xlabel="Epoch", ylabel="IoU (%)", title=f"{title}: Validation IoU")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[2].plot(history_df["epoch"], history_df["val_precision"] * 100, "o-", markersize=3, label="Precision")
    axes[2].plot(history_df["epoch"], history_df["val_recall"] * 100, "o-", markersize=3, label="Recall")
    axes[2].plot(history_df["epoch"], history_df["val_pred_fg"] * 100, "d--", markersize=3, label="Pred FG %")
    axes[2].set(xlabel="Epoch", ylabel="Percentage (%)", title=f"{title}: P / R / FG")
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def train(args) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    split = json.loads(args.split.read_text(encoding="utf-8"))
    index_df = pd.read_csv(args.cache / "images.csv", dtype={"ID": str})
    id_to_row = {identifier: row for row, identifier in enumerate(index_df["ID"])}
    assert len(id_to_row) == 5000
    assert len(split["train"]) == 4500 and len(split["validation"]) == 500
    assert set(split["train"]).isdisjoint(split["validation"])

    train_indices = [id_to_row[i] for i in split["train"]]
    val_indices = [id_to_row[i] for i in split["validation"]]

    train_ds = AugmentedCacheDataset(args.cache, train_indices, train=True)
    val_ds = AugmentedCacheDataset(args.cache, val_indices, train=False)

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
        drop_last=True, generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
    )

    model = build_architecture(args.arch, pretrained=True).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    backbone_params = list(model.backbone_parameters())
    decoder_params = list(model.decoder_parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr},
            {"params": decoder_params, "lr": args.lr * args.decoder_lr_mult},
        ],
        weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    args.out.mkdir(parents=True, exist_ok=True)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    image_size = int(train_ds.image_size)
    config = {
        "arch": args.arch,
        "pretrained_weights": "torchvision IMAGENET1K_V1",
        "external_data": False,
        "parameters": n_params,
        "split_sha256": hashlib.sha256(args.split.read_bytes()).hexdigest(),
        "train_images": len(train_ds),
        "validation_images": len(val_ds),
        "input_size": image_size,
        "seed": args.seed,
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "batch_size": args.batch_size,
        "optimizer": "AdamW",
        "backbone_lr": args.lr,
        "decoder_lr": args.lr * args.decoder_lr_mult,
        "weight_decay": args.weight_decay,
        "scheduler": "linear_warmup_cosine",
        "loss": args.loss,
        "augmentation": "hflip(0.5), random_resized_crop(0.6, scale=0.70-0.98), mild_color_jitter(0.5)",
        "precision": "bfloat16_amp" if use_bf16 else "float32",
        "selection_metric": f"micro foreground IoU at {image_size}x{image_size} with threshold 0.5",
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"=== Starting {args.arch} ({n_params:,} params) at {image_size}x{image_size} ===", flush=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history = []
    best_iou = -1.0
    best_epoch = -1
    ckpt_path = args.ckpt_dir / "model.pth"

    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        n_seen = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(images)
                loss = compute_loss(logits.float(), targets, args.loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss_sum += loss.item() * len(images)
            n_seen += len(images)

        val_metrics = evaluate_grid(model, val_loader, device, use_bf16)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        sec = time.perf_counter() - t0
        cur_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": cur_lr,
            "train_loss": train_loss_sum / n_seen,
            **val_metrics,
            "seconds": round(sec, 2),
        }
        history.append(row)
        print(
            f"Ep {epoch:02d}/{args.epochs} | tr_loss={row['train_loss']:.4f} | "
            f"val_loss={row['val_loss']:.4f} | val_mIoU={row['val_micro_iou']:.2%} | "
            f"val_imgIoU={row['val_mean_img_iou']:.2%} | P={row['val_precision']:.2%} | "
            f"R={row['val_recall']:.2%} | FG={row['val_pred_fg']:.2%} | {sec:.1f}s",
            flush=True,
        )

        if row["val_micro_iou"] > best_iou:
            best_iou = row["val_micro_iou"]
            best_epoch = epoch
            torch.save(
                {
                    "arch": args.arch,
                    "image_size": image_size,
                    "threshold": 0.5,
                    "tta": True,
                    "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                },
                ckpt_path,
            )

        hist_df = pd.DataFrame(history)
        hist_df.to_csv(args.out / "training_history.csv", index=False)

    peak_mem = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    summary = {
        "arch": args.arch,
        "best_epoch": best_epoch,
        f"best_val_micro_iou_{image_size}": best_iou,
        "best_val_mean_img_iou": float(history[best_epoch - 1]["val_mean_img_iou"]),
        "training_seconds": round(sum(r["seconds"] for r in history), 2),
        "peak_allocated_bytes": peak_mem,
        "checkpoint_sha256": hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),
        "checkpoint_bytes": ckpt_path.stat().st_size,
    }
    (args.out / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_curves(pd.DataFrame(history), args.out / "training_curves.png", args.arch)
    print(f"=== Finished {args.arch}: best epoch {best_epoch}, val_mIoU={best_iou:.2%}, ckpt={ckpt_path} ===", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", type=str, required=True, choices=["resnet34_unet", "convnext_tiny_unet", "convnext_small_unet", "convnext_small_srm_unet"])
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--decoder-lr-mult", type=float, default=5.0)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--loss", type=str, default="bce_dice", choices=["bce_dice", "focal_dice"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=5489)
    train(parser.parse_args())
