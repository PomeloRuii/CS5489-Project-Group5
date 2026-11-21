import hashlib
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from preflight import SEED, loaders
from srm_model import BASE, HERE, SRMUNet
from project import run_epoch


def train():
    check = json.loads((HERE / "preflight.json").read_text())
    assert check["overfit"]["final"]["iou"] >= .5
    out = HERE / "srm256"
    out.mkdir(exist_ok=True)
    if (out / "training_history.csv").exists():
        raise FileExistsError("Phase 4 training history already exists")
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), "Use the cs5489 CUDA environment"
    device = torch.device("cuda")
    torch.manual_seed(SEED)
    _, data = loaders()
    train_loader = DataLoader(data["train"], batch_size=8, shuffle=True)
    val_loader = DataLoader(data["validation"], batch_size=8)
    config = {
        "model": "SRMUNet (SmallUNet with 6-input first convolution)",
        "feature_method": "after-resize fixed RGB-N Fig. 4 filters on BT.601 luma",
        "srm_kernel_divisors": [4, 12, 2], "srm_boundary": "2-pixel reflect padding",
        "srm_scale_clip": "multiply by 255 / 3; clamp [-1,1]",
        "parameters_v1": check["parameters_v1"], "parameters_srm": check["parameters_srm"],
        "seed": SEED, "initialization": "V1 initialization for all shared entries; new input channel weights use Conv2d initialization",
        "split": "../../baseline/split.json", "split_sha256": hashlib.sha256((BASE / "split.json").read_bytes()).hexdigest(),
        "train_images": len(data["train"]), "validation_images": len(data["validation"]),
        "input_size": 256, "epochs": 10, "batch_size": 8, "train_shuffle": True,
        "validation_shuffle": False, "dataloader_workers": 0, "cpu_threads": 4,
        "optimizer": "Adam", "lr": .001, "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0,
        "loss": "unweighted BCEWithLogits + per-image soft Dice, smoothing 1",
        "bce_pos_weight": None, "augmentation": "none", "precision": "float32 (no AMP)",
        "rgb_resize": "PIL bilinear; divide by 255", "mask_resize": "PIL nearest",
        "external_data": False, "pretrained_weights": False, "threshold": .5,
        "selection_metric": "validation foreground micro-IoU at 256×256; ties keep earliest epoch",
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    torch.cuda.reset_peak_memory_stats(device)
    model = SRMUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    rows, best_iou, best_epoch = [], -1, None
    for epoch in range(1, 11):
        torch.cuda.synchronize()
        start = time.perf_counter()
        train_loss = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_iou = run_epoch(model, val_loader, device)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        rows.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_iou": val_iou,
            "seconds": seconds, "peak_allocated_bytes_so_far": torch.cuda.max_memory_allocated(device),
        })
        if val_iou > best_iou:
            best_iou, best_epoch = val_iou, epoch
            torch.save({"model_state": model.state_dict(), "image_size": 256}, out / "model.pth")
        pd.DataFrame(rows).to_csv(out / "training_history.csv", index=False)
        print(f"Epoch {epoch:02d}/10: train loss {train_loss:.4f}, val loss {val_loss:.4f}, "
              f"val IoU {val_iou:.4f}, {seconds:.1f}s", flush=True)
    summary = dict(best_epoch=best_epoch, best_val_iou_256=best_iou,
                   training_seconds=sum(row["seconds"] for row in rows),
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3), layout="constrained")
    for label, frame in (("V1 RGB", pd.read_csv(BASE / "training_history.csv")), ("RGB + SRM", pd.DataFrame(rows))):
        for ax, column, title in zip(axes, ("train_loss", "val_loss", "val_iou"),
                                    ("Train BCE + Dice", "Validation BCE + Dice", "Validation micro-IoU at 256")):
            ax.plot(frame.epoch, frame[column], "o-", label=label)
            ax.set(xlabel="Epoch", ylabel=title)
            ax.grid(alpha=.25)
            ax.legend()
    fig.savefig(out / "training_curves.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    train()
