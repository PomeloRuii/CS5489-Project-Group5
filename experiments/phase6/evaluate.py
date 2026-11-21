"""Evaluate the archived full-image candidates and the new depth control."""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, precision_recall_curve
from torch.utils.data import DataLoader

from architecture_audit import BASE, HERE, PATHS, ROOT, TABLES
from deep_unet import DeepUNet, SmallUNet
from project import predict_mask

sys.path.extend((str(ROOT / "phase3"), str(HERE / "submission_candidate")))
from prepare import ARCHIVE, archive_index, read_pair
from train_candidate import datasets


HISTORIES = {
    "V1": BASE / "training_history.csv",
    "Weighted": ROOT / "phase2/weighted/training_history.csv",
    "384": ROOT / "phase3/resolution384/training_history.csv",
}
BINS = ((0, .01), (.01, .05), (.05, .15), (.15, .3), (.3, 1.01))


def counts_metrics(tp, fp, fn):
    return {"TP": int(tp), "FP": int(fp), "FN": int(fn),
            "precision": tp / (tp + fp) if tp + fp else 0.,
            "recall": tp / (tp + fn) if tp + fn else 0.,
            "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.}


def summarize(frame):
    tp, fp, fn = frame[["TP", "FP", "FN"]].sum()
    return {"images": len(frame), **counts_metrics(tp, fp, fn),
            "mean_image_iou": frame.iou.mean(), "mean_image_dice": frame.dice.mean(),
            "predicted_fraction": (tp + fp) / frame.pixels.sum(),
            "target_fraction": (tp + fn) / frame.pixels.sum(),
            "empty_predictions": int((frame.TP + frame.FP).eq(0).sum()),
            "zero_overlap": int(frame.TP.eq(0).sum())}


def load_models(checkpoints, device):
    nets = {}
    for name, path in checkpoints.items():
        saved = torch.load(path, map_location=device, weights_only=True)
        net = DeepUNet() if name == "Deep256" else SmallUNet()
        net.load_state_dict(saved["model_state"], strict=True)
        nets[name] = net.to(device).eval(), saved["image_size"]
    return nets


def native_eval(nets, device, ids, out):
    rows = {name: [] for name in nets}
    names = list(nets)
    with ZipFile(ARCHIVE) as zipped, ThreadPoolExecutor(max_workers=4) as executor:
        prefix, labels = archive_index(zipped)

        def load(identifier):
            return identifier, *read_pair(zipped, prefix, identifier, labels.loc[identifier])

        for identifier in ids[:2]:
            _, image, _ = load(identifier)
            for model, size in nets.values():
                predict_mask(model, image, device, size)
        for start in range(0, len(ids), 12):
            for index, (identifier, image, truth) in enumerate(
                    executor.map(load, ids[start:start + 12]), start):
                actual = np.count_nonzero(truth)
                for name in names[index % len(names):] + names[:index % len(names)]:
                    model, size = nets[name]
                    begin = time.perf_counter()
                    guess = predict_mask(model, image, device, size)
                    seconds = time.perf_counter() - begin
                    pred = np.count_nonzero(guess)
                    tp = np.count_nonzero(guess & truth)
                    rows[name].append({"ID": identifier, "pixels": truth.size,
                                       "target_fraction": actual / truth.size,
                                       "predicted_fraction": pred / truth.size, "inference_seconds": seconds,
                                       **counts_metrics(tp, pred - tp, actual - tp)})
            if (start + 12) % 120 == 0 or start + 12 >= len(ids):
                print(f"Native validation: {min(start + 12, len(ids))}/{len(ids)}", flush=True)
    frames = {name: pd.DataFrame(data) for name, data in rows.items()}
    archive_match = {}
    for name, path in TABLES.items():
        historical = pd.read_csv(path, dtype={"ID": str})
        current = frames[name]
        assert current.ID.tolist() == historical.ID.tolist() == ids
        assert (current.TP + current.FN).equals(historical.TP + historical.FN)
        errors = current[["TP", "FP", "FN"]] - historical[["TP", "FP", "FN"]]
        archive_match[name] = {"rows_identical": int(errors.eq(0).all(axis=1).sum()),
                               "max_pixel_diff": int(errors.abs().to_numpy().max())}
    template = frames["V1"]
    target = template.TP + template.FN
    assert all((frame.TP + frame.FN).equals(target) and frame.pixels.equals(template.pixels)
               for frame in frames.values())
    all_foreground = template.copy()
    all_foreground["TP"], all_foreground["FP"], all_foreground["FN"] = target, template.pixels - target, 0
    all_foreground["iou"] = target / template.pixels
    all_foreground["dice"] = 2 * target / (template.pixels + target)
    summary = {name: {**summarize(frame), "inference_seconds": frame.inference_seconds.sum()}
               for name, frame in frames.items()}
    summary["All foreground"] = summarize(all_foreground)
    groups = []
    for name, frame in {**frames, "All foreground": all_foreground}.items():
        for lo, hi in BINS:
            group = frame[frame.target_fraction.ge(lo) & frame.target_fraction.lt(hi)]
            groups.append({"model": name, "target_fraction_range": f"[{lo:g},{min(hi, 1):g}{']' if hi > 1 else ')'}",
                           **summarize(group)})
    pd.concat([f.assign(model=n) for n, f in frames.items()], ignore_index=True).to_csv(
        out / "validation_by_image.csv", index=False)
    pd.DataFrame(groups).to_csv(out / "area_bins.csv", index=False)
    return summary, archive_match


@torch.inference_mode()
def common_grid_pr(net, device):
    _, val_set = datasets(BASE / "cache256", BASE / "split.json")
    truth = np.array(val_set.masks[val_set.indices], dtype=np.uint8).ravel()
    values = np.empty(truth.size, dtype=np.float32)
    offset = 0
    for images, _ in DataLoader(val_set, batch_size=8):
        probabilities = net(images.to(device)).sigmoid().cpu().numpy().ravel()
        values[offset:offset + probabilities.size] = probabilities
        offset += probabilities.size
    assert offset == truth.size
    precision, recall, _ = precision_recall_curve(truth, values)
    historical = pd.read_csv(ROOT / "phase4/srm256/pr_metrics.csv").set_index("model")
    scores = {name: float(historical.loc[name, "PR_AUC_trapezoid"]) for name in PATHS}
    scores["Deep256"] = float(auc(recall, precision))
    return scores


def training_curves(histories, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), layout="constrained")
    for name, path in histories.items():
        history = pd.read_csv(path)
        train_column = "train_weighted_loss" if name == "Weighted" else "train_loss"
        for ax, column in zip(axes, (train_column, "val_loss", "val_iou")):
            ax.plot(history.epoch, history[column], "o-", label=name)
    for ax, title in zip(axes, ("Training loss (Weighted differs)", "Unweighted validation loss",
                                "Validation micro-IoU at input resolution")):
        ax.set(xlabel="Epoch", ylabel=title)
        ax.grid(alpha=.25)
        ax.legend()
    fig.savefig(out / "training_curves.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "deep256")
    out = parser.parse_args().out
    histories = {**HISTORIES, "Deep256": out / "training_history.csv"}
    torch.set_num_threads(4)
    device = torch.device("cuda")
    nets = load_models({**PATHS, "Deep256": out / "model.pth"}, device)
    ids = json.loads((BASE / "split.json").read_text())["validation"]
    summary, archive_match = native_eval(nets, device, ids, out)
    scores = common_grid_pr(nets["Deep256"][0], device)
    for name, path in histories.items():
        history = pd.read_csv(path)
        best = history.loc[history.val_iou.idxmax()]
        model, size = nets[name]
        summary[name].update(parameters=sum(p.numel() for p in model.parameters()),
                             input_size=size, checkpoint_epoch=int(best.epoch),
                             val_iou_input_grid=best.val_iou, training_seconds=history.seconds.sum(),
                             pr_auc_256_grid=scores[name])
    pd.DataFrame.from_dict(summary, orient="index").rename_axis("model").to_csv(
        out / "native_comparison.csv")
    training_curves(histories, out)
    result = {"threshold": 0.5, "historical_pixel_counts_check": archive_match,
              "time_scope": "sum over 500 decoded RGB images, CPU resize through returned native mask, "
                            "two warmups per model, excluding image decoding and metric computation",
              "pr_scope": "exact pooled trapezoidal pixel PR-AUC on the same 500 x 256 x 256 grid",
              "results": summary}
    (out / "validation_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
