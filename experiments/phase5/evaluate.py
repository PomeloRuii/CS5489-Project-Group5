import json
import time
from concurrent.futures import ThreadPoolExecutor
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch

from prepare import ARCHIVE, BASE, HERE, ROOT, SEED, archive_index, read_pair
from project import SmallUNet
from sliding import sliding_probability

GROUPS = ("A", "B")
BINS = 4096
THRESHOLDS = np.arange(1, 10) / 10


def score(tp, fp, fn):
    def ratio(n, d, otherwise=0.):
        return float(n / d) if d else otherwise
    return {
        "TP": int(tp), "FP": int(fp), "FN": int(fn),
        "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
        "dice": ratio(2 * tp, 2 * tp + fp + fn, 1.),
        "iou": ratio(tp, tp + fp + fn, 1.),
    }


def group_summary(frame):
    return {
        "images": len(frame), **score(frame.TP.sum(), frame.FP.sum(), frame.FN.sum()),
        "macro_iou": float(frame.iou.mean()),
        "predicted_fraction": float((frame.TP + frame.FP).sum() / frame.pixels.sum()),
        "zero_overlap_images": int(frame.TP.eq(0).sum()),
        "empty_predictions": int((frame.TP + frame.FP).eq(0).sum()),
    }


def bootstrap(a, b):
    rng = np.random.default_rng(SEED)
    results = {}
    for name, include in (("all", a.target_fraction.ge(0)), ("under_1_percent", a.target_fraction.lt(.01))):
        x, y = a[include], b[include]
        ix = rng.integers(0, len(x), (3000, len(x)))
        estimates, point_estimates = [], []
        for frame in (x, y):
            tp, fp, fn, pixels = (frame[k].to_numpy()[ix].sum(axis=1) for k in ("TP","FP","FN","pixels"))
            estimates.append({
                "micro_iou": tp / (tp + fp + fn), "macro_iou": frame.iou.to_numpy()[ix].mean(axis=1),
                "dice": 2*tp / (2*tp + fp + fn),
                "precision": np.divide(tp, tp+fp, out=np.zeros_like(tp, dtype=float), where=tp+fp > 0),
                "recall": tp / (tp + fn), "background_fpr": fp / (pixels - tp - fn),
            })
            s = group_summary(frame)
            point_estimates.append({
                "micro_iou": s["iou"], "macro_iou": s["macro_iou"], "dice": s["dice"],
                "precision": s["precision"], "recall": s["recall"],
                "background_fpr": s["FP"] / (frame.pixels.sum() - s["TP"] - s["FN"]),
            })
        results[name] = {"count": len(x), "B_minus_A": {
            metric: {
                "point": float(point_estimates[1][metric] - point_estimates[0][metric]),
                "ci95": np.quantile(estimates[1][metric] - estimates[0][metric], [.025,.975]).tolist(),
            }
            for metric in estimates[0]
        }}
    return results


@torch.inference_mode()
def evaluate():
    torch.set_num_threads(4)
    device = torch.device("cuda")
    ids = json.loads((BASE / "split.json").read_text())["validation"]
    models = {}
    for name in GROUPS:
        net = SmallUNet().to(device)
        checkpoint = torch.load(HERE / name / "model.pth", map_location=device, weights_only=True)
        assert checkpoint["image_size"] == 256
        net.load_state_dict(checkpoint["model_state"])
        models[name] = net.eval()
    rows = {name: [] for name in GROUPS}
    counts = {name: np.zeros((len(THRESHOLDS), 2), dtype=np.int64) for name in GROUPS}
    hist = {name: np.zeros((2, BINS + 1), dtype=np.int64) for name in GROUPS}
    timings = {name: [] for name in GROUPS}
    expected = pd.read_csv(ROOT / "phase3/foreground_by_image.csv", dtype={"ID": str}).set_index("ID")
    chosen = pd.read_csv(ROOT / "phase3/resolution384/visualization_selection.csv", dtype={"ID": str}).ID.tolist()
    figure_cache = {}
    with ZipFile(ARCHIVE) as zipped, ThreadPoolExecutor(max_workers=4) as pool:
        root, labels = archive_index(zipped)
        def load(identifier):
            rgb, mask = read_pair(zipped, root, identifier, labels.loc[identifier])
            return identifier, np.asarray(rgb), mask

        for start in range(0, len(ids), 16):
            for identifier, image, mask in pool.map(load, ids[start:start+16]):
                n = len(rows["A"])
                gt = mask.astype(bool)
                true_pixels = int(gt.sum())
                assert true_pixels == expected.loc[identifier, "native_foreground"]
                # Fixed native-grid background texture check, independent of predictions and labels for its threshold.
                green = image[::2, ::2, 1].astype(np.int16)
                edge = np.zeros(green.shape, dtype=bool)
                edge[:, 1:] |= np.abs(np.diff(green, axis=1)) >= 20
                edge[1:] |= np.abs(np.diff(green, axis=0)) >= 20
                background = ~gt[::2, ::2]
                high, low = edge & background, ~edge & background
                high_n, low_n = np.count_nonzero(high), np.count_nonzero(low)
                shots = {}
                # Alternate timing order to reduce systematic first-model warm-up bias.
                for name in (GROUPS if n % 2 == 0 else GROUPS[::-1]):
                    net = models[name]
                    torch.cuda.synchronize()
                    now = time.perf_counter()
                    proba, tiles = sliding_probability(image, lambda x: net(x)[:, 0].sigmoid(), device)
                    seconds = time.perf_counter() - now
                    timings[name].append(seconds)
                    assert proba.shape == gt.shape and np.isfinite(proba).all()
                    predicted = proba >= .5
                    positive_scores = proba[gt]
                    tp, claimed = np.count_nonzero(predicted & gt), np.count_nonzero(predicted)
                    rows[name].append({
                        "ID": identifier, "pixels": gt.size, "height": gt.shape[0], "width": gt.shape[1],
                        "target_fraction": true_pixels / gt.size, "tiles": tiles, "inference_seconds": seconds,
                        "high_texture_background": high_n, "low_texture_background": low_n,
                        "high_texture_fp": np.count_nonzero(predicted[::2, ::2] & high),
                        "low_texture_fp": np.count_nonzero(predicted[::2, ::2] & low),
                        **score(tp, claimed - tp, true_pixels - tp),
                    })
                    for i, threshold in enumerate(THRESHOLDS):
                        counts[name][i] += (np.count_nonzero(proba >= threshold), np.count_nonzero(positive_scores >= threshold))
                    hist[name][0] += np.bincount((proba.ravel() * BINS).astype(np.int32), minlength=BINS+1)
                    hist[name][1] += np.bincount((positive_scores * BINS).astype(np.int32), minlength=BINS+1)
                    if identifier in chosen:
                        shots[name] = Image.fromarray(predicted.astype(np.uint8))
                if identifier in chosen:
                    preview = Image.fromarray(image)
                    preview.thumbnail((360, 270), Image.Resampling.BILINEAR)
                    figure_cache[identifier] = (
                        preview, Image.fromarray(mask).resize(preview.size, Image.Resampling.NEAREST),
                        *(shots[name].resize(preview.size, Image.Resampling.NEAREST) for name in GROUPS),
                    )
            if len(rows["A"]) % 80 == 0 or len(rows["A"]) == len(ids):
                print(f"Paired sliding-window inference completed for {len(rows['A'])}/{len(ids)}", flush=True)

    frames = {name: pd.DataFrame(rows[name]) for name in GROUPS}
    assert all(frame.ID.tolist() == ids for frame in frames.values())
    a, b = (frames[name] for name in GROUPS)
    assert np.array_equal(a.tiles, b.tiles) and np.array_equal(a.TP + a.FN, b.TP + b.FN)
    new_metrics = {}
    all_groups, all_thresholds, pr_rows = [], [], []
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    for name, frame in frames.items():
        frame.to_csv(HERE / name / "validation_by_image.csv", index=False)
        m = group_summary(frame)
        m["predicted_tiles"] = int(frame.tiles.sum())
        m["inference_seconds"] = float(sum(timings[name]))
        m["inference_median_seconds"] = float(np.median(timings[name]))
        m["high_texture_fpr"] = float(frame.high_texture_fp.sum() / frame.high_texture_background.sum())
        m["low_texture_fpr"] = float(frame.low_texture_fp.sum() / frame.low_texture_background.sum())
        tp_truth = m["TP"] + m["FN"]
        np.testing.assert_array_equal(counts[name][4], [m["TP"] + m["FP"], m["TP"]])
        assert hist[name][0].sum() == frame.pixels.sum() and hist[name][1].sum() == tp_truth
        h = hist[name][:, ::-1]
        valid = h[0] > 0
        prediction, positive = np.cumsum(h[:, valid], axis=1)
        recall, precision = np.r_[0, positive / tp_truth], np.r_[1, positive / prediction]
        m["pr_auc_binned"] = float(np.trapezoid(precision, recall))
        m["ap_binned"] = float(np.sum(np.diff(recall) * precision[1:]))
        ax.plot(recall, precision, label=f"{name}: {m['pr_auc_binned']:.3f}")
        pr_rows.extend({"group": name, "recall": r, "precision": p} for r,p in zip(recall, precision))
        np.save(HERE / name / "native_probability_bin_counts.npy", hist[name])
        for i, threshold in enumerate(THRESHOLDS):
            pred, tp = counts[name][i]
            all_thresholds.append({"group": name, "threshold": threshold, **score(tp, pred-tp, tp_truth-tp)})
        for lo,hi in ((0,.01),(.01,.05),(.05,.15),(.15,.30),(.30,1.01)):
            bucket = frame[frame.target_fraction.ge(lo) & frame.target_fraction.lt(hi)]
            all_groups.append({"group": name, "area": f"[{lo:g},{min(hi,1):g}{']' if hi>1 else ')'}",
                               **group_summary(bucket)})
        new_metrics[name] = m
    ax.axhline((a.TP + a.FN).sum() / a.pixels.sum(), linestyle=":", color="gray", label="Foreground prevalence")
    ax.set(xlabel="Pixel recall", ylabel="Pixel precision", xlim=(0,1), ylim=(0,1))
    ax.legend()
    ax.grid(alpha=.25)
    fig.savefig(HERE / "pr_curves.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(pr_rows).to_csv(HERE / "pr_curve_points.csv", index=False)
    pd.DataFrame(all_thresholds).to_csv(HERE / "threshold_sweep.csv", index=False)
    pd.DataFrame(all_groups).to_csv(HERE / "area_bins.csv", index=False)
    (HERE / "bootstrap.json").write_text(json.dumps(bootstrap(a,b),indent=2),encoding="utf-8")
    paired = a[["ID","iou"]].merge(b[["ID","iou"]],on="ID",suffixes=("_A","_B"))
    paired["iou_B_minus_A"] = paired.iou_B - paired.iou_A
    paired.to_csv(HERE / "paired_by_image.csv", index=False)
    fig, axes = plt.subplots(len(chosen),4,figsize=(11,12),layout="constrained")
    by_id = paired.set_index("ID")
    for identifier, panels in zip(chosen, axes):
        values = by_id.loc[identifier]
        for panel, image, title in zip(panels, figure_cache[identifier],
                       (identifier,"Reference",f"A · IoU {values.iou_A:.3f}",f"B · IoU {values.iou_B:.3f}")):
            panel.imshow(image, cmap="gray", vmin=0, vmax=1)
            panel.set_title(title)
            panel.axis("off")
    fig.savefig(HERE / "validation_examples.png", dpi=140)
    plt.close(fig)
    (HERE / "validation_summary.json").write_text(json.dumps(new_metrics, indent=2), encoding="utf-8")
    print(json.dumps(new_metrics, indent=2), flush=True)


if __name__ == "__main__":
    evaluate()
