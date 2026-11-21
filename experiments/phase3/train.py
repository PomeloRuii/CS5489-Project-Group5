import hashlib
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from prepare import BASE, CACHE, HERE, PHASE2
from project import CachedMaskDataset, SmallUNet, run_epoch


def train():
    if not (HERE / "data_audit.json").is_file():
        raise RuntimeError("Run prepare.py and complete its data checks first")
    out = HERE / "resolution384"
    out.mkdir(exist_ok=True)
    if (out / "training_history.csv").exists():
        raise FileExistsError("A Phase 3 training history already exists")
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("The RTX 5090 CUDA environment is required for this controlled run")
    device = torch.device("cuda")
    seed, epochs, batch_size, lr, image_size = 5489, 10, 8, 1e-3, 384
    torch.manual_seed(seed)
    split_path = BASE / "split.json"
    split = json.loads(split_path.read_text())
    image_index = pd.read_csv(CACHE / "images.csv", dtype={"ID": str})
    lookup = {identifier: i for i, identifier in enumerate(image_index.ID)}
    groups = {key: [lookup[identifier] for identifier in values] for key, values in split.items()}
    train_set = CachedMaskDataset(CACHE, groups["train"])
    val_set = CachedMaskDataset(CACHE, groups["validation"])
    assert (len(train_set), len(val_set)) == (4500, 500)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    config = {
        "sole_experimental_change": "input size 256 -> 384",
        "split": "../../baseline/split.json", "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "train_images": len(train_set), "validation_images": len(val_set), "seed": seed,
        "model": "baseline/project.py:SmallUNet", "initialization": "fresh PyTorch defaults",
        "input_size": image_size, "training_epochs": epochs, "batch_size": batch_size,
        "train_shuffle": True, "validation_shuffle": False, "dataloader_workers": 0, "cpu_threads": 4,
        "optimizer": "Adam", "optimizer_arguments": {
            "lr": lr, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0,
        },
        "loss": "unweighted BCEWithLogits + per-image soft Dice, smoothing 1",
        "bce_pos_weight": None, "augmentation": "none", "precision": "float32 (no AMP)",
        "input_processing": "bilinear RGB resize, divide by 255; nearest-neighbor binary mask resize",
        "threshold": 0.5,
        "selection_metric": "highest validation foreground micro-IoU at input resolution, ties keep earliest epoch",
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
        "runtime_scope": "train and validation passes, synchronized, excluding preprocessing and file writes",
        "peak_memory_scope": "torch.cuda.max_memory_allocated, not total device VRAM",
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    torch.cuda.reset_peak_memory_stats(device)
    model = SmallUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history, best_iou, best_epoch = [], -1.0, None
    for epoch in range(1, epochs + 1):
        torch.cuda.synchronize()
        start = time.perf_counter()
        train_loss = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_iou = run_epoch(model, val_loader, device)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_iou": val_iou, "seconds": seconds,
            "peak_allocated_bytes_so_far": torch.cuda.max_memory_allocated(device),
        })
        print(f"Epoch {epoch:02d}/{epochs}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, "
              f"val IoU {val_iou:.4f}, {seconds:.1f}s", flush=True)
        if val_iou > best_iou:
            best_iou, best_epoch = val_iou, epoch
            torch.save({"model_state": model.state_dict(), "image_size": image_size}, out / "model.pth")
        pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
    summary = dict(
        best_epoch=best_epoch, best_val_iou_384=best_iou,
        training_seconds=sum(row["seconds"] for row in history),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
    )
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)

    v1 = pd.read_csv(BASE / "training_history.csv")
    weighted = pd.read_csv(PHASE2 / "weighted" / "training_history.csv")
    high = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), layout="constrained")
    for df, name, train_column in (
        (v1, "V1 256", "train_loss"),
        (weighted, "Weighted BCE 256", "train_weighted_loss"),
        (high, "V1 loss 384", "train_loss"),
    ):
        for panel, column in zip(axes, (train_column, "val_loss", "val_iou")):
            panel.plot(df.epoch, df[column], "o-", label=name)
    for panel, label in zip(axes, ("Train loss (Weighted BCE differs)", "Unweighted validation loss",
                                   "Validation IoU at model input resolution")):
        panel.set(xlabel="Epoch", ylabel=label)
        panel.grid(alpha=.25)
        panel.legend()
    fig.savefig(out / "training_curves.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    train()
