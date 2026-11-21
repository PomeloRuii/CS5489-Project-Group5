import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch

from project import SmallUNet, decode_rle, predict_mask


def evaluate(archive, directory):
    """Evaluate the saved checkpoint on full-resolution held-out images."""
    split = json.loads((directory / "split.json").read_text(encoding="utf-8"))
    if set(split["train"]) & set(split["validation"]):
        raise ValueError("Overlap between train and validation IDs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    checkpoint = torch.load(directory / "model.pth", map_location=device, weights_only=True)
    model = SmallUNet().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    history = pd.read_csv(directory / "training_history.csv")
    best = history.loc[history["val_iou"].idxmax()]
    rows = []
    with ZipFile(archive) as zipped, ThreadPoolExecutor(max_workers=4) as executor:
        index, = [name for name in zipped.namelist() if name.endswith("/train.csv")]
        root = index.removesuffix("train.csv")
        labels = pd.read_csv(
            zipped.open(index), dtype={"ID": str, "annotation": str}, keep_default_na=False
        ).set_index("ID")

        def load(identifier):
            label = labels.loc[identifier]
            with Image.open(io.BytesIO(zipped.read(f"{root}train/images/{identifier}.webp"))) as image:
                image = image.convert("RGB")
            target = decode_rle(label.annotation, image.height, image.width)
            return identifier, image, target

        ids = split["validation"]
        for start in range(0, len(ids), 16):
            for identifier, image, target in executor.map(load, ids[start:start + 16]):
                prediction = predict_mask(model, image, device, checkpoint["image_size"])
                tp = int(np.count_nonzero(prediction & target))
                actual, guessed = int(np.count_nonzero(target)), int(np.count_nonzero(prediction))
                union = actual + guessed - tp
                rows.append({
                    "ID": identifier, "height": image.height, "width": image.width,
                    "target_pixels": actual, "predicted_pixels": guessed, "intersection": tp,
                    "union": union, "iou": tp / union if union else 1.0,
                })
            if len(rows) % 80 == 0 or len(rows) == len(ids):
                print(f"Full-resolution validation: {len(rows)}/{len(ids)}", flush=True)
        table = pd.DataFrame(rows)
        table.to_csv(directory / "validation_by_image.csv", index=False)

        ranked = table.sort_values("iou")
        examples = ranked.iloc[[0, len(ranked) // 4, len(ranked) // 2, -1]]
        fig, axes = plt.subplots(4, 3, figsize=(10, 11), layout="constrained")
        for ax, example in zip(axes, examples.itertuples(index=False)):
            _, image, target = load(example.ID)
            prediction = predict_mask(model, image, device, checkpoint["image_size"])
            image.thumbnail((400, 320), Image.Resampling.BILINEAR)
            ax[0].imshow(image)
            ax[0].set_title(f"{example.ID} · IoU {example.iou:.3f}")
            for panel, mask, title in zip(ax[1:], (target, prediction), ("Reference", "Prediction")):
                preview = Image.fromarray(mask).resize(image.size, Image.Resampling.NEAREST)
                panel.imshow(preview, cmap="gray", vmin=0, vmax=1)
                panel.set_title(title)
            for panel in ax:
                panel.axis("off")
        fig.savefig(directory / "validation_examples.png", dpi=150)
        plt.close(fig)
    intersection, union = int(table.intersection.sum()), int(table.union.sum())
    # The grouped IoU uses the combined pixel counts for the selected checkpoint.
    total_predicted = int(table.predicted_pixels.sum())
    total_target = int(table.target_pixels.sum())
    metrics = {
        "training_images": len(split["train"]),
        "validation_images": len(table),
        "training_resolution": checkpoint["image_size"],
        "epochs": len(history),
        "best_epoch": int(best["epoch"]),
        "val_iou_resized_micro": float(best["val_iou"]),
        "val_iou_full_resolution_micro": intersection / union if union else 1.0,
        "val_iou_full_resolution_mean_per_image": float(table["iou"].mean()),
        "val_dice_full_resolution_micro": 2 * intersection / (total_predicted + total_target),
        "val_empty_predictions": int(table.predicted_pixels.eq(0).sum()),
        "val_zero_overlap_images": int(table.intersection.eq(0).sum()),
        "training_seconds": float(history.seconds.sum()),
    }
    (directory / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    evaluate(here.parent / "AIGC Detection Dataset 2026.zip", here)
