import argparse
import hashlib
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "baseline"
sys.path.insert(0, str(BASE))
from project import SmallUNet, decode_rle, image_tensor

THRESHOLDS = np.arange(1, 10) / 10
WEIGHTS = BASE / "model.pth"
ARCHIVE = HERE.parent / "AIGC Detection Dataset 2026.zip"
CACHE = BASE / "cache256"


def scores(tp, fp, fn):
    return {
        "TP": int(tp), "FP": int(fp), "FN": int(fn),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "dice": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 1.0,
        "iou": float(tp / (tp + fp + fn)) if tp + fp + fn else 1.0,
    }


def snapshot_v1(out):
    current = {
        p.relative_to(BASE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(BASE.iterdir()) if p.is_file()
    }
    manifest = out / "v1_input_sha256.json"
    if manifest.exists():
        assert current == json.loads(manifest.read_text()), "Baseline V1 changed"
    else:
        manifest.write_text(json.dumps(current, indent=2), encoding="utf-8")


@torch.inference_mode()
def evaluate(checkpoint_path, out):
    out.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    assert torch.cuda.is_available()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    size = checkpoint["image_size"]
    model = SmallUNet().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    split = json.loads((BASE / "split.json").read_text())
    ids = split["validation"]
    image_ids = pd.read_csv(CACHE / "images.csv", dtype={"ID": str})
    index = {identifier: i for i, identifier in enumerate(image_ids.ID)}
    cache_x = np.load(CACHE / "images.npy", mmap_mode="r")
    cache_y = np.load(CACHE / "masks.npy", mmap_mode="r")
    train_pixels = sum(np.count_nonzero(cache_y[index[k]]) for k in split["train"])
    threshold_logits = torch.logit(torch.tensor(THRESHOLDS, device=device, dtype=torch.float32))
    sums = np.zeros((len(THRESHOLDS), 2), dtype=np.int64)  # positives predicted, TP
    resized = np.zeros(3, dtype=np.int64)  # TP, FP, FN at 0.5
    total_pixels = 0
    rows, samples_p, samples_y = [], [], []

    with ZipFile(ARCHIVE) as zipped, ThreadPoolExecutor(max_workers=4) as executor:
        csv, = [name for name in zipped.namelist() if name.endswith("/train.csv")]
        root = csv.removesuffix("train.csv")
        labels = pd.read_csv(zipped.open(csv), dtype={"ID": str}, keep_default_na=False).set_index("ID")

        def load(identifier):
            row = labels.loc[identifier]
            with Image.open(io.BytesIO(zipped.read(f"{root}train/images/{identifier}.webp"))) as image:
                image = image.convert("RGB")
            mask = decode_rle(row.annotation, image.height, image.width)
            return identifier, image, mask

        for start in range(0, len(ids), 16):
            for identifier, image, mask in executor.map(load, ids[start:start + 16]):
                x = image_tensor(image, size)
                i = index[identifier]
                if len(rows) < 12:
                    assert np.array_equal((x.permute(1, 2, 0) * 255).round().byte().numpy(), cache_x[i])
                    target_resize = np.asarray(Image.fromarray(mask).resize(
                        (size, size), Image.Resampling.NEAREST
                    ))
                    assert np.array_equal(target_resize, cache_y[i])
                target = torch.from_numpy(np.ascontiguousarray(mask)).to(device=device, dtype=torch.bool)
                logits = model(x.unsqueeze(0).to(device))
                small_pred = logits[0, 0].cpu().numpy() >= 0
                small_target = cache_y[i].astype(bool)
                resized += [
                    np.count_nonzero(small_pred & small_target),
                    np.count_nonzero(small_pred & ~small_target),
                    np.count_nonzero(~small_pred & small_target),
                ]
                native = F.interpolate(logits, size=mask.shape, mode="bilinear", align_corners=False)[0, 0]
                threshold_counts = []
                for threshold in threshold_logits:
                    predicted = native >= threshold
                    threshold_counts.append(torch.stack((
                        predicted.sum(dtype=torch.int32),
                        (predicted & target).sum(dtype=torch.int32),
                    )))
                counts = torch.stack(threshold_counts).cpu().numpy()
                sums += counts
                predicted_count, tp = map(int, counts[4])
                true_count = int(np.count_nonzero(mask))
                total_pixels += mask.size
                rows.append({
                    "ID": identifier, "pixels": mask.size,
                    "target_fraction": true_count / mask.size,
                    "predicted_fraction": predicted_count / mask.size,
                    **scores(tp, predicted_count - tp, true_count - tp),
                })
                phase = len(rows) * 53 % 256
                samples_p.append(native.flatten()[phase::256].sigmoid().cpu().numpy())
                samples_y.append(mask.ravel()[phase::256])
            if len(rows) % 80 == 0 or len(rows) == len(ids):
                print(f"{out.name}: evaluated {len(rows)}/{len(ids)}", flush=True)

        if checkpoint_path != WEIGHTS:
            old = pd.read_csv(HERE / "v1" / "validation_by_image.csv", dtype={"ID": str})
            paired = old[["ID", "iou"]].merge(pd.DataFrame(rows), on="ID", suffixes=("_v1", "_weighted"))
            paired["difference"] = paired.iou_weighted - paired.iou_v1
            v1_ranked = paired.sort_values(["iou_v1", "ID"])
            examples = pd.concat([
                v1_ranked.iloc[[len(paired)//4, len(paired)//2, 3*len(paired)//4, 19*len(paired)//20]],
                paired.loc[[paired.difference.idxmax(), paired.difference.idxmin()]],
            ])
            v1 = SmallUNet().to(device)
            v1.load_state_dict(torch.load(WEIGHTS, map_location=device, weights_only=True)["model_state"])
            v1.eval()
            fig, axes = plt.subplots(len(examples), 4, figsize=(11, 13), layout="constrained")
            for ax, row in zip(axes, examples.itertuples()):
                _, image, target = load(row.ID)
                x = image_tensor(image, size).unsqueeze(0).to(device)
                predictions = []
                for net in (v1, model):
                    logits = F.interpolate(net(x), size=target.shape, mode="bilinear", align_corners=False)
                    predictions.append((logits[0, 0] >= 0).cpu().numpy().astype(np.uint8))
                image.thumbnail((360, 270))
                ax[0].imshow(image)
                ax[0].set_title(row.ID)
                titles = ("Reference", f"V1 IoU {row.iou_v1:.3f}", f"Weighted IoU {row.iou_weighted:.3f}")
                for panel, mask, title in zip(ax[1:], (target, *predictions), titles):
                    panel.imshow(Image.fromarray(mask).resize(image.size, Image.Resampling.NEAREST),
                                 cmap="gray", vmin=0, vmax=1)
                    panel.set_title(title)
                for panel in ax:
                    panel.axis("off")
            fig.savefig(out / "validation_examples.png", dpi=130)
            plt.close(fig)

    table = pd.DataFrame(rows)
    table.to_csv(out / "validation_by_image.csv", index=False)
    positives = int(table.TP.sum() + table.FN.sum())
    sweep = pd.DataFrame([
        {"threshold": t, "predicted_fraction": float(pred / total_pixels),
         **scores(tp, pred - tp, positives - tp)}
        for t, (pred, tp) in zip(THRESHOLDS, sums)
    ])
    sweep.to_csv(out / "threshold_sweep.csv", index=False)
    grouped = []
    bins = [0, .01, .05, .15, .30, 1.01]
    for lo, hi in zip(bins, bins[1:]):
        b = table[(table.target_fraction >= lo) & (table.target_fraction < hi)]
        if len(b):
            grouped.append({
                "target_fraction_range": f"[{lo:g},{min(hi, 1):g}{']' if hi > 1 else ')'}",
                "images": len(b), "empty_predictions": int(b.predicted_fraction.eq(0).sum()),
                "mean_image_iou": b.iou.mean(), "mean_target_fraction": b.target_fraction.mean(),
                "mean_predicted_fraction": b.predicted_fraction.mean(),
                **scores(b.TP.sum(), b.FP.sum(), b.FN.sum()),
            })
    pd.DataFrame(grouped).to_csv(out / "area_bins.csv", index=False)
    prob, mask = np.concatenate(samples_p), np.concatenate(samples_y)
    edges = np.linspace(0, 1, 21)
    histogram = pd.DataFrame({"lower": edges[:-1], "upper": edges[1:]})
    quantiles = {}
    for name, selected in (("all", prob), ("foreground", prob[mask == 1]), ("background", prob[mask == 0])):
        histogram[name] = np.histogram(selected, bins=edges)[0]
        quantiles[name] = dict(zip(("p05", "p25", "p50", "p75", "p95"), np.quantile(selected, [.05, .25, .5, .75, .95]).tolist()))
    histogram.to_csv(out / "probability_histogram.csv", index=False)
    result = {
        "validation_images": len(ids),
        "training_foreground_fraction_256": train_pixels / (len(split["train"]) * size**2),
        "native_pixels": total_pixels,
        "native_target_fraction": positives / total_pixels,
        "mean_image_target_fraction": float(table.target_fraction.mean()),
        "mean_image_predicted_fraction": float(table.predicted_fraction.mean()),
        "native_fixed_threshold": sweep.iloc[4].to_dict(),
        "native_fixed_threshold_mean_image_iou": float(table.iou.mean()),
        "native_empty_predictions": int(table.predicted_fraction.eq(0).sum()),
        "native_zero_overlap": int(table.TP.eq(0).sum()),
        "native_all_foreground": scores(positives, total_pixels - positives, 0),
        "native_best_sweep": sweep.loc[sweep.iou.idxmax()].to_dict(),
        "resized_fixed_threshold": scores(*resized),
        "probability_sample_pixels": len(prob),
        "probability_quantiles": quantiles,
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["v1", "weighted"])
    args = parser.parse_args()
    snapshot_v1(HERE)
    folder = HERE / args.model
    evaluate(WEIGHTS if args.model == "v1" else folder / "model.pth", folder)
