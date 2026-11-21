import hashlib
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from prepare import BASE, CACHE, EPOCHS, HERE, SEED
from project import SmallUNet, run_epoch
from sliding import SIZE, STRIDE, INFERENCE_BATCH


class PatchDataset(Dataset):
    def __init__(self, path, validation=False):
        self.data = np.load(path, mmap_mode="c")
        self.epoch = 0
        self.validation = validation

    def __len__(self):
        return self.data.shape[0] * (self.data.shape[1] if self.validation else 1)

    def __getitem__(self, index):
        if self.validation:
            i, j = divmod(index, self.data.shape[1])
        else:
            i, j = index, self.epoch
        packed = self.data[i, j]
        x = torch.from_numpy(packed[:, :, :3]).permute(2, 0, 1).float().div_(255)
        y = torch.from_numpy(packed[:, :, 3]).unsqueeze(0).float()
        return x, y


def train(group, device):
    out = HERE / group
    out.mkdir(exist_ok=True)
    if (out / "training_history.csv").exists():
        raise FileExistsError(f"{group} training history already exists")
    torch.manual_seed(SEED)
    train_data = PatchDataset(CACHE / f"{group}.npy")
    val_data = PatchDataset(CACHE / "val.npy", validation=True)
    batch_size = 8
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    held_out = DataLoader(val_data, batch_size=batch_size)
    assert (len(train_data), len(val_data), len(loader)) == (4500, 2500, 563)
    torch.cuda.reset_peak_memory_stats(device)
    model = SmallUNet().to(device)
    initial_hash = hashlib.sha256(b"".join(
        tensor.detach().cpu().numpy().tobytes() for tensor in model.state_dict().values()
    )).hexdigest()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    config = {
        "group": group, "sole_controlled_difference": "crop selection rule",
        "requested_crop_ratios": {"uniform": 1.} if group == "A" else {"uniform": .5, "tamper": .3, "background": .2},
        "raw_crop_size": SIZE, "split": "../../baseline/split.json",
        "split_sha256": hashlib.sha256((BASE / "split.json").read_bytes()).hexdigest(),
        "model": "SmallUNet", "initialization_sha256": initial_hash,
        "seed": SEED, "initialize_from_checkpoint": False, "model_parameters": sum(p.numel() for p in model.parameters()),
        "loss": "V1 unweighted BCEWithLogits + soft Dice, smoothing 1",
        "optimizer": "Adam", "lr": .001, "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0,
        "scheduler": None, "augmentation": None, "precision": "float32",
        "batch_size": batch_size, "shuffle_train": True, "shuffle_validation": False,
        "epochs": EPOCHS, "training_crops_per_epoch": len(train_data), "optimizer_steps_per_epoch": len(loader),
        "images_train": 4500, "images_validation": 500, "validation_crops": len(val_data),
        "checkpoint_selection": "fixed five coordinate-based validation crops per image; pooled foreground IoU at .5; earliest tie",
        "native_inference": {"stride": STRIDE, "batch": INFERENCE_BATCH, "threshold": .5,
                             "weight": "separable nonperiodic Hann, each axis floored at .1; average sigmoid probabilities"},
        "pretrained_weights": False, "external_data": False,
        "device": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    history, best, best_epoch = [], -1, None
    with (out / "training.log").open("x", encoding="utf-8") as log:
        for epoch in range(1, EPOCHS + 1):
            train_data.epoch = epoch - 1
            torch.cuda.synchronize()
            start = time.perf_counter()
            loss = run_epoch(model, loader, device, optimizer)
            val_loss, val_iou = run_epoch(model, held_out, device)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            history.append({
                "epoch": epoch, "train_loss": loss, "val_loss": val_loss, "val_crop_micro_iou": val_iou,
                "seconds": seconds, "peak_allocated_bytes_so_far": torch.cuda.max_memory_allocated(device),
            })
            line = (f"{group} {epoch:02d}/{EPOCHS}: train loss {loss:.4f}, val loss {val_loss:.4f}, "
                    f"val crop IoU {val_iou:.4f}, {seconds:.1f}s")
            print(line, flush=True)
            print(line, file=log, flush=True)
            if val_iou > best:
                best, best_epoch = val_iou, epoch
                torch.save({"model_state": model.state_dict(), "image_size": SIZE}, out / "model.pth")
            pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
        summary = {
            "best_epoch": best_epoch, "best_val_crop_micro_iou": best,
            "train_validation_seconds": sum(row["seconds"] for row in history),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "updates": EPOCHS * len(loader),
        }
        print(json.dumps(summary, indent=2), file=log)
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return initial_hash


def curves():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
    for name in ("A", "B"):
        frame = pd.read_csv(HERE / name / "training_history.csv")
        for ax, key, title in zip(axes, ("train_loss","val_loss","val_crop_micro_iou"),
                                   ("Train loss", "Fixed validation crop loss", "Fixed validation crop IoU")):
            ax.plot(frame.epoch, frame[key], "o-", label=name)
            ax.set(xlabel="Epoch", ylabel=title)
            ax.grid(alpha=.3)
            ax.legend()
    fig.savefig(HERE / "training_curves.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    assert all((HERE / name).is_file() for name in (
        "data_preflight.json", "sliding_preflight.json", "real_inference_preflight.json"
    ))
    torch.set_num_threads(4)
    device = torch.device("cuda")
    a = train("A", device)
    torch.cuda.empty_cache()
    b = train("B", device)
    assert a == b, "The two models did not start from identical weights"
    curves()
