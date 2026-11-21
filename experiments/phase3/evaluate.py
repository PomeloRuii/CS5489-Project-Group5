import json
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

from prepare import ARCHIVE, BASE, CACHE, HERE, PHASE2, archive_index, read_pair
from project import SmallUNet, image_tensor, predict_mask

OUT = HERE / "resolution384"
THRESHOLDS = np.arange(1, 10) / 10


def score(tp, fp, fn):
    return {
        "TP": int(tp), "FP": int(fp), "FN": int(fn),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "dice": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 1.0,
        "iou": float(tp / (tp + fp + fn)) if tp + fp + fn else 1.0,
    }


def model_from(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    net = SmallUNet().to(device)
    net.load_state_dict(checkpoint["model_state"])
    return net.eval(), checkpoint["image_size"]


def summarize(df):
    return {
        "images": len(df), **score(df.TP.sum(), df.FP.sum(), df.FN.sum()),
        "mean_image_iou": float(df.iou.mean()),
        "empty_predictions": int((df.TP + df.FP).eq(0).sum()),
        "zero_overlap": int(df.TP.eq(0).sum()),
        "target_fraction": float((df.TP + df.FN).sum() / df.pixels.sum()),
        "predicted_fraction": float((df.TP + df.FP).sum() / df.pixels.sum()),
    }


def group_tables(tables):
    ranges = [(0, .01), (.01, .05), (.05, .15), (.15, .30), (.30, 1.01)]
    grouped = []
    for name, table in tables.items():
        for low, high in ranges:
            group = table[table.target_fraction.ge(low) & table.target_fraction.lt(high)]
            label = f"[{low:g},{min(high, 1):g}{']' if high > 1 else ')'}"
            grouped.append({"model": name, "target_fraction_range": label, **summarize(group)})
    return pd.DataFrame(grouped)


def bootstrap_difference(v1, new):
    # Pair resamples by image. Intervals describe sampling uncertainty for these fixed checkpoints.
    rng = np.random.default_rng(5489)
    result = {}
    for key, keep in (("all", v1.target_fraction.ge(0)), ("lt_1_percent", v1.target_fraction.lt(.01))):
        left, right = v1.loc[keep], new.loc[keep]
        ids = rng.integers(0, len(left), (2000, len(left)))
        def micro(df):
            tp = df.TP.to_numpy()[ids].sum(axis=1)
            total = df[["TP", "FP", "FN"]].sum(axis=1).to_numpy()[ids].sum(axis=1)
            return tp / total
        delta = micro(right) - micro(left)
        result[key] = np.quantile(delta, [.025, .975]).tolist()
    return result


def plot_examples(load, tables, device):
    joined = tables["384"].set_index("ID")
    small = joined[joined.target_fraction.lt(.01)].sort_index()
    success = small[small.iou.ge(.25)].sort_values("iou")
    failures = small[small.iou.eq(0)]
    rng = np.random.default_rng(5489)
    selected = [
        ("small: median success (IoU >= 0.25)", success.index[len(success)//2]) if len(success)
        else ("small: highest IoU (no case >= 0.25)", small.iou.idxmax()),
        ("small: random zero-overlap case", rng.choice(failures.index)) if len(failures)
        else ("small: lowest IoU", small.iou.idxmin()),
        ("small: random case", rng.choice(small.index)),
        ("1–5%: random case", rng.choice(joined[joined.target_fraction.between(.01, .05, inclusive="left")].index)),
        (">=30%: random case", rng.choice(joined[joined.target_fraction.ge(.30)].index)),
    ]
    nets = [
        model_from(BASE / "model.pth", device),
        model_from(PHASE2 / "weighted" / "model.pth", device),
        model_from(OUT / "model.pth", device),
    ]
    indexed = [tables[name].set_index("ID") for name in ("V1", "Weighted", "384")]
    fig, axes = plt.subplots(len(selected), 5, figsize=(13, 11), layout="constrained")
    notes = []
    for panels, (category, identifier) in zip(axes, selected):
        _, image, target = load(identifier)
        preds = [predict_mask(net, image, device, size) for net, size in nets]
        image.thumbnail((360, 270), Image.Resampling.BILINEAR)
        panels[0].imshow(image)
        panels[0].set_title(f"{identifier} · {joined.loc[identifier, 'target_fraction']:.2%}")
        notes.append({"ID": identifier, "selection": category, **{
            f"{name}_iou": float(df.loc[identifier, "iou"])
            for name, df in zip(("v1", "weighted", "384"), indexed)
        }})
        titles = ["Reference"] + [f"{name}\nIoU {df.loc[identifier, 'iou']:.3f}"
                                  for name, df in zip(("V1 256", "Weighted 256", "V1 loss 384"), indexed)]
        for panel, mask, title in zip(panels[1:], [target, *preds], titles):
            panel.imshow(Image.fromarray(mask).resize(image.size, Image.Resampling.NEAREST),
                         cmap="gray", vmin=0, vmax=1)
            panel.set_title(title)
        for panel in panels:
            panel.axis("off")
    fig.savefig(OUT / "validation_examples.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(notes).to_csv(OUT / "visualization_selection.csv", index=False)


@torch.inference_mode()
def evaluate():
    torch.set_num_threads(4)
    device = torch.device("cuda")
    split = json.loads((BASE / "split.json").read_text())
    ids = split["validation"]
    training = json.loads((OUT / "training_summary.json").read_text())
    net, size = model_from(OUT / "model.pth", device)
    assert size == 384
    lookup = {identifier: i for i, identifier in enumerate(pd.read_csv(CACHE / "images.csv", dtype={"ID": str}).ID)}
    cache_masks = np.load(CACHE / "masks.npy", mmap_mode="r")
    thresholds = torch.logit(torch.tensor(THRESHOLDS, device=device, dtype=torch.float32))
    totals = np.zeros((len(THRESHOLDS), 2), dtype=np.int64)
    rows, resized = [], np.zeros(3, dtype=np.int64)
    tables = {
        name: pd.read_csv(PHASE2 / folder / "validation_by_image.csv", dtype={"ID": str})
        for name, folder in (("V1", "v1"), ("Weighted", "weighted"))
    }
    with ZipFile(ARCHIVE) as zipped, ThreadPoolExecutor(max_workers=4) as executor:
        root, labels = archive_index(zipped)
        def load(identifier):
            image, target = read_pair(zipped, root, identifier, labels.loc[identifier])
            return identifier, image, target

        for start in range(0, len(ids), 16):
            for identifier, image, target in executor.map(load, ids[start:start + 16]):
                logits = net(image_tensor(image, size).unsqueeze(0).to(device))
                small_pred = logits[0, 0].cpu().numpy() >= 0
                small_target = cache_masks[lookup[identifier]].astype(bool)
                resized += [np.count_nonzero(small_pred & small_target),
                            np.count_nonzero(small_pred & ~small_target),
                            np.count_nonzero(~small_pred & small_target)]
                binary = torch.from_numpy(np.ascontiguousarray(target)).to(device=device, dtype=torch.bool)
                native = F.interpolate(logits, size=target.shape, mode="bilinear", align_corners=False)[0, 0]
                counts = []
                for threshold in thresholds:
                    guess = native >= threshold
                    counts.append(torch.stack((guess.sum(dtype=torch.int32), (guess & binary).sum(dtype=torch.int32))))
                counts = torch.stack(counts).cpu().numpy()
                totals += counts
                predicted, tp = map(int, counts[4])
                truth = int(np.count_nonzero(target))
                rows.append({
                    "ID": identifier, "pixels": target.size, "target_fraction": truth / target.size,
                    "predicted_fraction": predicted / target.size,
                    **score(tp, predicted - tp, truth - tp),
                })
            if len(rows) % 80 == 0 or len(rows) == len(ids):
                print(f"Native evaluation {len(rows)}/{len(ids)}", flush=True)
        table = pd.DataFrame(rows)
        tables["384"] = table
        for previous in (tables["V1"], tables["Weighted"]):
            assert previous.ID.tolist() == table.ID.tolist() == ids
            assert np.array_equal(previous.pixels, table.pixels)
            assert np.array_equal(previous.TP + previous.FN, table.TP + table.FN)
        plot_examples(load, tables, device)

    table.to_csv(OUT / "validation_by_image.csv", index=False)
    groups = group_tables(tables)
    groups.to_csv(OUT / "area_bins_comparison.csv", index=False)
    truth, pixels = (table.TP + table.FN).sum(), table.pixels.sum()
    sweep = pd.DataFrame([
        {"threshold": t, "predicted_fraction": float(pred / pixels), **score(tp, pred - tp, truth - tp)}
        for t, (pred, tp) in zip(THRESHOLDS, totals)
    ])
    sweep.to_csv(OUT / "threshold_sweep.csv", index=False)
    all_sweeps = []
    for name, folder in (("V1", "v1"), ("Weighted", "weighted")):
        all_sweeps.append(pd.read_csv(PHASE2 / folder / "threshold_sweep.csv").assign(model=name))
    all_sweeps.append(sweep.assign(model="384"))
    pd.concat(all_sweeps, ignore_index=True).to_csv(OUT / "threshold_comparison.csv", index=False)
    fixed = {name: summarize(df) for name, df in tables.items()}
    assert np.isclose(score(*resized)["iou"], training["best_val_iou_384"], atol=1e-6)
    pd.DataFrame.from_dict(fixed, orient="index").rename_axis("model").to_csv(OUT / "native_comparison.csv")
    diff = table.iou - tables["V1"].iou
    summary = dict(
        native_fixed_threshold=fixed,
        native_all_foreground=score(truth, pixels - truth, 0),
        resized_fixed_threshold_384=score(*resized),
        native_best_threshold_384=sweep.loc[sweep.iou.idxmax()].to_dict(),
        paired_count_vs_v1={"higher": int((diff > 0).sum()), "same": int((diff == 0).sum()), "lower": int((diff < 0).sum())},
        paired_bootstrap_95_percent_native_micro_iou_delta_vs_v1=bootstrap_difference(tables["V1"], table),
    )
    (OUT / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    evaluate()
