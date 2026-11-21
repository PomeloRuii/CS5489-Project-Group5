"""Recompute original-resolution validation metrics from the archived counts."""
import csv
import json
from pathlib import Path


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
RECORDS = {
    "V1": "baseline/validation_by_image.csv",
    "Weighted BCE": "phase2/weighted/validation_by_image.csv",
    "384": "phase3/resolution384/validation_by_image.csv",
    "RGB + SRM": "phase4/srm256/validation_by_image.csv",
    "Uniform Crop": "phase5/A/validation_by_image.csv",
    "Mask-aware Crop": "phase5/B/validation_by_image.csv",
    "Deep256": "phase6/deep256/validation_by_image.csv",
}


def counts(row):
    if "TP" in row:
        return tuple(int(row[key]) for key in ("TP", "FP", "FN", "pixels"))
    tp = int(row["intersection"])
    return (tp, int(row["predicted_pixels"]) - tp, int(row["target_pixels"]) - tp,
            int(row["height"]) * int(row["width"]))


def iou(tp, fp, fn):
    union = tp + fp + fn
    return tp / union if union else 1.


def summary(records):
    tp, fp, fn, _ = (sum(column) for column in zip(*records))
    small = [row for row in records if (row[0] + row[2]) * 100 < row[3]]
    small_tp, small_fp, small_fn, _ = (sum(column) for column in zip(*small))
    return {
        "micro_iou": iou(tp, fp, fn),
        "macro_iou": sum(iou(*row[:3]) for row in records) / len(records),
        "precision": tp / (tp + fp) if tp + fp else 0.,
        "recall": tp / (tp + fn) if tp + fn else 0.,
        "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.,
        "small_iou": iou(small_tp, small_fp, small_fn),
        "small_n": len(small),
    }


def recompute():
    split = json.loads((EXPERIMENTS / "baseline/split.json").read_text(encoding="utf-8"))
    reference = None
    results = {}
    for model, filename in RECORDS.items():
        with (EXPERIMENTS / filename).open(newline="", encoding="utf-8") as stream:
            rows = [row for row in csv.DictReader(stream) if "model" not in row or row["model"] == model]
        by_id = {row["ID"]: counts(row) for row in rows}
        if len(rows) != len(by_id) or set(by_id) != set(split["validation"]):
            raise ValueError(f"Unexpected validation IDs in {filename}")
        truth = {key: (tp + fn, pixels) for key, (tp, _, fn, pixels) in by_id.items()}
        if reference is None:
            reference = truth
        if truth != reference:
            raise ValueError(f"Original masks or image dimensions differ in {filename}")
        results[model] = summary(list(by_id.values()))
    results["All foreground"] = summary([(n, pixels - n, 0, pixels) for n, pixels in reference.values()])
    return results


if __name__ == "__main__":
    print("Model                  micro-IoU  macro-IoU      Dice  Precision     Recall    <1% IoU")
    for name, stats in recompute().items():
        values = (stats[key] for key in ("micro_iou", "macro_iou", "dice", "precision", "recall", "small_iou"))
        print(f"{name:<23}" + " ".join(f"{value:>10.2%}" for value in values))
