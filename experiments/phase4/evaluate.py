import json
import sys
from concurrent.futures import ThreadPoolExecutor
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import auc, precision_recall_curve

from preflight import ARCHIVE, CACHE, loaders
from srm_model import BASE, HERE, SRMUNet
from project import CachedMaskDataset, SmallUNet, image_tensor, predict_mask

P2, P3 = HERE.parent / "phase2", HERE.parent / "phase3"
sys.path.append(str(P3))
from prepare import archive_index, read_pair

OUT = HERE / "srm256"
MODELS = {
    "V1": BASE / "model.pth",
    "Weighted": P2 / "weighted/model.pth",
    "384": P3 / "resolution384/model.pth",
    "SRM": OUT / "model.pth",
}
THRESHOLDS = np.arange(1, 10) / 10


def load_model(name, device):
    checkpoint = torch.load(MODELS[name], map_location=device, weights_only=True)
    model = (SRMUNet() if name == "SRM" else SmallUNet()).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model.eval(), checkpoint["image_size"]


def metrics(tp, fp, fn):
    return {
        "TP": int(tp), "FP": int(fp), "FN": int(fn),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.,
        "dice": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 1.,
        "iou": float(tp / (tp + fp + fn)) if tp + fp + fn else 1.,
    }


def aggregate(frame):
    return {
        "images": len(frame), **metrics(frame.TP.sum(), frame.FP.sum(), frame.FN.sum()),
        "macro_iou": float(frame.iou.mean()),
        "predicted_fraction": float((frame.TP + frame.FP).sum() / frame.pixels.sum()),
        "target_fraction": float((frame.TP + frame.FN).sum() / frame.pixels.sum()),
        "empty_predictions": int((frame.TP + frame.FP).eq(0).sum()),
        "zero_overlap": int(frame.TP.eq(0).sum()),
    }


def prediction_figure(load, tables, device):
    selections = pd.read_csv(P3 / "resolution384/visualization_selection.csv", dtype={"ID": str})
    ids = selections.ID.tolist()
    models = [load_model(name, device) for name in MODELS]
    fig, axes = plt.subplots(len(ids), 6, figsize=(15, 10), layout="constrained")
    tables = {name: frame.set_index("ID") for name, frame in tables.items()}
    values = []
    for panels, identifier in zip(axes, ids):
        _, image, target = load(identifier)
        predictions = [predict_mask(net, image, device, size) for net, size in models]
        image.thumbnail((325, 250), Image.Resampling.BILINEAR)
        panels[0].imshow(image)
        panels[0].set_title(identifier)
        row = {"ID": identifier, "target_fraction": float(target.mean())}
        for panel, mask, name in zip(panels[1:], [target, *predictions], ["Reference", *MODELS]):
            preview = Image.fromarray(mask).resize(image.size, Image.Resampling.NEAREST)
            panel.imshow(preview, cmap="gray", vmin=0, vmax=1)
            if name in MODELS:
                val = float(tables[name].loc[identifier, "iou"])
                row[f"{name}_iou"] = val
                panel.set_title(f"{name} · IoU {val:.3f}")
            else:
                panel.set_title(name)
        for panel in panels:
            panel.axis("off")
        values.append(row)
    fig.savefig(OUT / "validation_examples.png", dpi=125)
    plt.close(fig)
    pd.DataFrame(values).to_csv(OUT / "visualization_cases.csv", index=False)


@torch.inference_mode()
def native_eval(device, ids):
    net, size = load_model("SRM", device)
    thresholds = torch.logit(torch.tensor(THRESHOLDS, dtype=torch.float32, device=device))
    totals = np.zeros((9, 2), dtype=np.int64)
    rows = []
    previous = {
        "V1": P2 / "v1/validation_by_image.csv",
        "Weighted": P2 / "weighted/validation_by_image.csv",
        "384": P3 / "resolution384/validation_by_image.csv",
    }
    tables = {name: pd.read_csv(path, dtype={"ID": str}) for name, path in previous.items()}
    with ZipFile(ARCHIVE) as zipped, ThreadPoolExecutor(max_workers=4) as executor:
        root, labels = archive_index(zipped)
        def load(identifier):
            image, mask = read_pair(zipped, root, identifier, labels.loc[identifier])
            return identifier, image, mask

        for start in range(0, len(ids), 16):
            for identifier, image, target in executor.map(load, ids[start:start + 16]):
                logits = net(image_tensor(image, size).unsqueeze(0).to(device))
                logits = F.interpolate(logits, size=target.shape, mode="bilinear", align_corners=False)[0, 0]
                actual = torch.from_numpy(np.ascontiguousarray(target)).to(device=device, dtype=torch.bool)
                batch = []
                for threshold in thresholds:
                    guess = logits >= threshold
                    batch.append(torch.stack((guess.sum(dtype=torch.int32), (guess & actual).sum(dtype=torch.int32))))
                counts = torch.stack(batch).cpu().numpy()
                totals += counts
                pred, tp = map(int, counts[4])
                truth = np.count_nonzero(target)
                rows.append({
                    "ID": identifier, "pixels": target.size, "target_fraction": truth / target.size,
                    "predicted_fraction": pred / target.size, **metrics(tp, pred - tp, truth - tp),
                })
            if len(rows) % 80 == 0 or len(rows) == len(ids):
                print(f"Native evaluation {len(rows)}/{len(ids)}", flush=True)
        tables["SRM"] = pd.DataFrame(rows)
        table = tables["SRM"]
        for frame in tables.values():
            assert frame.ID.tolist() == ids
            assert np.array_equal(frame.TP + frame.FN, table.TP + table.FN)
            assert np.array_equal(frame.pixels, table.pixels)
        prediction_figure(load, tables, device)

    table.to_csv(OUT / "validation_by_image.csv", index=False)
    counts = (table.TP + table.FN).sum()
    pixels = table.pixels.sum()
    sweep = pd.DataFrame([
        {"threshold": threshold, "predicted_fraction": pred / pixels, **metrics(tp, pred - tp, counts - tp)}
        for threshold, (pred, tp) in zip(THRESHOLDS, totals)
    ])
    sweep.to_csv(OUT / "threshold_sweep.csv", index=False)
    others = pd.read_csv(P3 / "resolution384/threshold_comparison.csv")
    pd.concat([others, sweep.assign(model="SRM")], ignore_index=True).to_csv(OUT / "threshold_comparison.csv", index=False)
    groups = []
    for name, frame in tables.items():
        for lo, hi in ((0, .01), (.01, .05), (.05, .15), (.15, .3), (.3, 1.01)):
            group = frame[frame.target_fraction.ge(lo) & frame.target_fraction.lt(hi)]
            groups.append({"model": name, "area": f"[{lo:g},{min(hi, 1):g}{']' if hi > 1 else ')'}",
                           **aggregate(group)})
    pd.DataFrame(groups).to_csv(OUT / "area_bins_comparison.csv", index=False)
    summary = {name: aggregate(frame) for name, frame in tables.items()}
    pd.DataFrame.from_dict(summary, orient="index").rename_axis("model").to_csv(OUT / "native_comparison.csv")
    delta = table.iou - tables["V1"].iou
    result = {"all_models_native": summary,
              "all_foreground": metrics(counts, pixels - counts, 0),
              "srm_best_threshold": sweep.loc[sweep.iou.idxmax()].to_dict(),
              "srm_vs_v1_images": {"higher": int(delta.gt(0).sum()), "same": int(delta.eq(0).sum()),
                                   "lower": int(delta.lt(0).sum())}}
    return result


@torch.inference_mode()
def pr_eval(device, ids, data256):
    # Exact pooled curves on 500 × 256 × 256 pixels per model.
    y = np.array(data256.masks[data256.indices], dtype=np.uint8).ravel()
    high_cache = P3 / "cache384"
    high_index = {identifier: i for i, identifier in enumerate(pd.read_csv(high_cache / "images.csv", dtype={"ID": str}).ID)}
    data384 = CachedMaskDataset(high_cache, [high_index[identifier] for identifier in ids])
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ax.axhline(y.mean(), color="gray", linestyle=":", label="Foreground prevalence")
    records, texture_rows = [], []
    for name in MODELS:
        net, size = load_model(name, device)
        values = np.empty(y.size, dtype=np.float32)
        cursor = 0
        for images, targets in DataLoader(data384 if size == 384 else data256, batch_size=8):
            images = images.to(device)
            logits = net(images)
            if size != 256:
                logits = F.interpolate(logits, (256, 256), mode="bilinear", align_corners=False)
            p = logits.sigmoid().cpu().numpy().ravel()
            values[cursor:cursor + p.size] = p
            cursor += p.size
            if name == "SRM":
                energy = net.srm(images).abs().mean(1).cpu().numpy()
                bg = ~targets[:, 0].numpy().astype(bool)
                pred = p.reshape(energy.shape) >= .5
                for group in (bg & (energy < .5), bg & (energy >= .5)):
                    texture_rows.append((np.count_nonzero(group), np.count_nonzero(pred & group)))
        assert cursor == y.size
        precision, recall, _ = precision_recall_curve(y, values)
        ap = float(-np.sum(np.diff(recall) * precision[:-1]))
        area = float(auc(recall, precision))
        pos = values[y == 1]
        neg = values[y == 0]
        records.append({
            "model": name, "pixels": y.size, "foreground_prevalence": y.mean(),
            "PR_AUC_trapezoid": area, "average_precision": ap,
            "foreground_probability_mean": float(pos.mean()), "foreground_probability_median": float(np.median(pos)),
            "background_probability_mean": float(neg.mean()), "background_probability_median": float(np.median(neg)),
        })
        subsample = np.linspace(0, len(recall) - 1, min(3000, len(recall)), dtype=int)
        ax.plot(recall[subsample], precision[subsample], label=f"{name}: {area:.3f}")
        print("Common 256-grid PR", records[-1], flush=True)
        del precision, recall, _, pos, neg, values
    ax.set(xlabel="Pixel foreground recall", ylabel="Pixel foreground precision", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    ax.grid(alpha=.25)
    fig.savefig(OUT / "pr_curves.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(records).to_csv(OUT / "pr_metrics.csv", index=False)
    bins = np.asarray(texture_rows).reshape(-1, 2, 2).sum(axis=0)
    texture = [
        dict(group=name, background_pixels=int(n), fp=int(fp), false_positive_rate=float(fp / n))
        for name, (n, fp) in zip(("abs_mean_residual_lt_0.5", "abs_mean_residual_ge_0.5"), bins)
    ]
    return records, texture


if __name__ == "__main__":
    torch.set_num_threads(4)
    split, ds = loaders()
    device = torch.device("cuda")
    result = native_eval(device, split["validation"])
    result["pr_at_shared_256_grid"], result["srm_texture_background_check"] = pr_eval(device, split["validation"], ds["validation"])
    (OUT / "validation_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
