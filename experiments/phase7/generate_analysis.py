"""Aggregate Phase 1-7 + Teammate metrics, inspect failure cases, and plot comparison figures."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from models import build_architecture

REPO = Path(__file__).resolve().parents[2]
P7 = REPO / "experiments" / "phase7"
DEFAULT_EXPORT_P7 = REPO.parent / "Course_Project_export" / "phase7"


def load_p7_row(label: str, subdir: str, mode: str = "tta_050") -> dict:
    s = json.loads((P7 / subdir / "validation_summary.json").read_text(encoding="utf-8"))
    m = s[mode]
    ab = m["area_bins"]
    pr = s["pr_auc_grid_tta"] if mode.startswith("tta") else s["pr_auc_grid_base"]
    return {
        "stage": label,
        "micro_iou": m["micro_iou"],
        "mean_img_iou": m["mean_img_iou"],
        "micro_dice": m["micro_dice"],
        "mean_img_dice": m["mean_img_dice"],
        "precision": m["precision"],
        "recall": m["recall"],
        "pred_fg": m["pred_fg"],
        "pix_acc": m["pix_acc"],
        "pr_auc": pr,
        "bin_<1%": ab["<1%"],
        "bin_1-5%": ab["1-5%"],
        "bin_5-15%": ab["5-15%"],
        "bin_15-30%": ab["15-30%"],
        "bin_30-100%": ab["30-100%"],
        "empty_imgs": m["empty_imgs"],
        "zero_tp_imgs": m["zero_tp_imgs"],
    }


def main(export_p7: Path = DEFAULT_EXPORT_P7):
    # Historical Phase 1-6 + Teammate rows (verified on exact same 500 validation images)
    rows = [
        {
            "stage": "All-Foreground Baseline",
            "micro_iou": 0.130691, "mean_img_iou": 0.123068, "micro_dice": 0.231169, "mean_img_dice": np.nan,
            "precision": 0.130691, "recall": 1.0, "pred_fg": 1.0, "pix_acc": 0.130691, "pr_auc": np.nan,
            "bin_<1%": 0.0053, "bin_1-5%": 0.0272, "bin_5-15%": 0.0838, "bin_15-30%": 0.2092, "bin_30-100%": 0.5537,
            "empty_imgs": 0, "zero_tp_imgs": 0,
        },
        {
            "stage": "Phase 1: V1 SmallUNet 256 (0.50)",
            "micro_iou": 0.133633, "mean_img_iou": 0.143007, "micro_dice": 0.235760, "mean_img_dice": 0.190514,
            "precision": 0.376311, "recall": 0.171649, "pred_fg": 0.059613, "pix_acc": 0.854562, "pr_auc": 0.244838,
            "bin_<1%": 0.025350, "bin_1-5%": 0.152967, "bin_5-15%": 0.132655, "bin_15-30%": 0.224454, "bin_30-100%": 0.110992,
            "empty_imgs": 68, "zero_tp_imgs": 191,
        },
        {
            "stage": "Phase 2: Weighted BCE 256 (0.50)",
            "micro_iou": 0.140543, "mean_img_iou": 0.143851, "micro_dice": 0.246449, "mean_img_dice": 0.200079,
            "precision": 0.276617, "recall": 0.222214, "pred_fg": 0.104987, "pix_acc": 0.822405, "pr_auc": 0.228509,
            "bin_<1%": 0.019765, "bin_1-5%": 0.104440, "bin_5-15%": 0.133725, "bin_15-30%": 0.239536, "bin_30-100%": 0.141809,
            "empty_imgs": 45, "zero_tp_imgs": 133,
        },
        {
            "stage": "Phase 3: V1 SmallUNet 384 (0.50)",
            "micro_iou": 0.132048, "mean_img_iou": 0.149113, "micro_dice": 0.233290, "mean_img_dice": 0.194732,
            "precision": 0.378193, "recall": 0.168667, "pred_fg": 0.058285, "pix_acc": 0.855110, "pr_auc": 0.228547,
            "bin_<1%": 0.026083, "bin_1-5%": 0.153116, "bin_5-15%": 0.129645, "bin_15-30%": 0.224017, "bin_30-100%": 0.108020,
            "empty_imgs": 107, "zero_tp_imgs": 225,
        },
        {
            "stage": "Phase 4: RGB+SRM SmallUNet 256 (0.50)",
            "micro_iou": 0.1135, "mean_img_iou": 0.1260, "micro_dice": 0.2039, "mean_img_dice": np.nan,
            "precision": np.nan, "recall": np.nan, "pred_fg": np.nan, "pix_acc": np.nan, "pr_auc": np.nan,
            "bin_<1%": np.nan, "bin_1-5%": np.nan, "bin_5-15%": np.nan, "bin_15-30%": np.nan, "bin_30-100%": np.nan,
            "empty_imgs": np.nan, "zero_tp_imgs": np.nan,
        },
        {
            "stage": "Phase 5B: Mask-Aware Crop (0.50)",
            "micro_iou": 0.1051, "mean_img_iou": 0.1175, "micro_dice": 0.1901, "mean_img_dice": np.nan,
            "precision": 0.4052, "recall": 0.1242, "pred_fg": 0.0401, "pix_acc": np.nan, "pr_auc": 0.2002,
            "bin_<1%": np.nan, "bin_1-5%": np.nan, "bin_5-15%": np.nan, "bin_15-30%": np.nan, "bin_30-100%": np.nan,
            "empty_imgs": np.nan, "zero_tp_imgs": np.nan,
        },
        {
            "stage": "Phase 6: 3-Scale UNet 256 (0.50)",
            "micro_iou": 0.1145, "mean_img_iou": 0.1297, "micro_dice": 0.2055, "mean_img_dice": np.nan,
            "precision": 0.3884, "recall": 0.1397, "pred_fg": 0.0470, "pix_acc": np.nan, "pr_auc": 0.2457,
            "bin_<1%": 0.0231, "bin_1-5%": 0.1247, "bin_5-15%": 0.1223, "bin_15-30%": 0.2115, "bin_30-100%": 0.0859,
            "empty_imgs": np.nan, "zero_tp_imgs": np.nan,
        },
        {
            "stage": "Teammate Tier-0 (CPU Weighted, 0.35+TTA+fill)",
            "micro_iou": 0.179002, "mean_img_iou": 0.154803, "micro_dice": 0.303650, "mean_img_dice": 0.232504,
            "precision": 0.201874, "recall": 0.612395, "pred_fg": 0.396456, "pix_acc": 0.632921, "pr_auc": 0.225658,
            "bin_<1%": 0.010524, "bin_1-5%": 0.049758, "bin_5-15%": 0.139197, "bin_15-30%": 0.308993, "bin_30-100%": 0.443092,
            "empty_imgs": 0, "zero_tp_imgs": 8,
        },
        load_p7_row("Phase 7A: ResNet34-UNet 512 (Base@0.50)", "resnet34_512", "base_050"),
        load_p7_row("Phase 7A: ResNet34-UNet 512 (TTA@0.50)", "resnet34_512", "tta_050"),
        load_p7_row("Phase 7B: ConvNeXt-Small Detail-UNet 512 (Base@0.50)", "convnext_small_512", "base_050"),
        load_p7_row("Phase 7B: ConvNeXt-Small Detail-UNet 512 (TTA@0.50)", "convnext_small_512", "tta_050"),
        load_p7_row("Phase 7C: ConvNeXt-Small+SRM Detail-UNet 512 (Base@0.50)", "convnext_small_srm_512", "base_050"),
        load_p7_row("Phase 7C: ConvNeXt-Small+SRM Detail-UNet 512 (TTA@0.50)", "convnext_small_srm_512", "tta_050"),
    ]
    comp = pd.DataFrame(rows)
    comp.to_csv(P7 / "phase_comparison.csv", index=False)

    # Plot progression & area-bin chart (named training_curves-style or phase7_comparison.png)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), layout="constrained")
    pick = [
        ("Phase 1\nV1 256", 0.1336, 0.1430, 0.2358),
        ("Phase 2\nWeighted", 0.1405, 0.1439, 0.2464),
        ("Teammate\nTier-0", 0.1790, 0.1548, 0.3037),
        ("Phase 7A\nResNet34", 0.4523, 0.4209, 0.6229),
        ("Phase 7C\nCNX+SRM", 0.8051, 0.7092, 0.8920),
        ("Phase 7B\nConvNeXt-S", 0.8101, 0.7137, 0.8951),
    ]
    x = np.arange(len(pick))
    w = 0.26
    axes[0].bar(x - w, [p[1] * 100 for p in pick], w, label="Native micro-IoU")
    axes[0].bar(x, [p[2] * 100 for p in pick], w, label="Mean Image IoU")
    axes[0].bar(x + w, [p[3] * 100 for p in pick], w, label="Native micro-Dice")
    axes[0].axhline(13.07, color="gray", linestyle="--", linewidth=1, label="All-FG prior (13.07%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([p[0] for p in pick], fontsize=8.5)
    axes[0].set_ylabel("Score (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_title("Validation Progression (500 Native-Res Images)")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    bins = ["<1%", "1-5%", "5-15%", "15-30%", "30-100%"]
    bx = np.arange(len(bins))
    for label, row_name, style in [
        ("Phase 2 Weighted (0.50)", "Phase 2: Weighted BCE 256 (0.50)", "o--"),
        ("Teammate Tier-0 (0.35)", "Teammate Tier-0 (CPU Weighted, 0.35+TTA+fill)", "s--"),
        ("Phase 7A ResNet34 (TTA)", "Phase 7A: ResNet34-UNet 512 (TTA@0.50)", "^-"),
        ("Phase 7C ConvNeXt-S+SRM", "Phase 7C: ConvNeXt-Small+SRM Detail-UNet 512 (TTA@0.50)", "d-"),
        ("Phase 7B ConvNeXt-S (Winner)", "Phase 7B: ConvNeXt-Small Detail-UNet 512 (TTA@0.50)", "o-"),
    ]:
        r = comp.loc[comp["stage"] == row_name].iloc[0]
        vals = [r[f"bin_{b}"] * 100 for b in bins]
        axes[1].plot(bx, vals, style, linewidth=2, markersize=5, label=label)
    axes[1].set_xticks(bx)
    axes[1].set_xticklabels(bins)
    axes[1].set_xlabel("Ground-Truth Edited Area Fraction")
    axes[1].set_ylabel("Native micro-IoU (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Performance by Edited Region Size")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    fig.savefig(P7 / "phase7_comparison.png", dpi=150)
    plt.close(fig)

    # Failure & success analysis on winning model (Phase 7B)
    df = pd.read_csv(P7 / "convnext_small_512" / "validation_by_image.csv", dtype={"ID": str})
    print("=== Phase 7B Per-Image IoU Distribution (500 Val Images) ===")
    print(df["iou"].describe(percentiles=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]))
    print("IoU >= 0.80:", int((df["iou"] >= 0.80).sum()), "/ 500")
    print("IoU >= 0.50:", int((df["iou"] >= 0.50).sum()), "/ 500")
    print("IoU <  0.10:", int((df["iou"] < 0.10).sum()), "/ 500")
    print("Zero TP:", int((df["TP"] == 0).sum()), "/ 500 | Empty pred:", int(((df["TP"] + df["FP"]) == 0).sum()), "/ 500")

    worst = df.sort_values("iou").head(12)
    print("\n=== Worst 12 Failure Cases (Phase 7B) ===")
    print(worst[["ID", "target_fraction", "predicted_fraction", "iou", "dice", "TP", "FP", "FN"]].to_string(index=False))

    small_best = df[df["target_fraction"] < 0.01].sort_values("iou", ascending=False).head(8)
    print("\n=== Best 8 Small-Edit (<1%) Cases (Phase 7B) ===")
    print(small_best[["ID", "target_fraction", "predicted_fraction", "iou", "dice"]].to_string(index=False))

    # Diagnose failure modes
    low = df[df["iou"] < 0.20]
    failure_stats = {
        "total_val": len(df),
        "iou_ge_080": int((df["iou"] >= 0.80).sum()),
        "iou_ge_050": int((df["iou"] >= 0.50).sum()),
        "iou_lt_020": len(low),
        "low_iou_median_target_frac": float(low["target_fraction"].median()),
        "all_val_median_target_frac": float(df["target_fraction"].median()),
        "low_iou_under_1pct_count": int((low["target_fraction"] < 0.01).sum()),
        "low_iou_empty_pred_count": int(((low["TP"] + low["FP"]) == 0).sum()),
        "low_iou_false_alarm_only_count": int(((low["TP"] == 0) & (low["FP"] > 0)).sum()),
        "low_iou_underpredict_count": int((low["predicted_fraction"] < low["target_fraction"] * 0.5).sum()),
        "low_iou_overpredict_count": int((low["predicted_fraction"] > low["target_fraction"] * 2.0).sum()),
    }
    (P7 / "failure_analysis.json").write_text(json.dumps(failure_stats, indent=2), encoding="utf-8")
    print("\n=== Failure Mode Summary ===")
    print(json.dumps(failure_stats, indent=2))

    # Measure inference VRAM & CPU/GPU latency for TA compatibility check (RTX A1000 8 GB)
    ckpt_file = export_p7 / "convnext_small_512" / "model.pth"
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=True)
    m = build_architecture(ckpt["arch"], pretrained=False)
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    x_cpu = torch.rand(1, 3, 512, 512)
    with torch.inference_mode():
        _ = m(x_cpu)
        tc0 = time.perf_counter()
        _ = m(x_cpu)
        _ = m(torch.flip(x_cpu, dims=[-1]))
        cpu_tta_ms = (time.perf_counter() - tc0) * 1000.0

    m = m.cuda()
    x_gpu = x_cpu.cuda()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        _ = m(x_gpu)
        torch.cuda.synchronize()
        tg0 = time.perf_counter()
        _ = m(x_gpu)
        _ = m(torch.flip(x_gpu, dims=[-1]))
        torch.cuda.synchronize()
        gpu_tta_ms = (time.perf_counter() - tg0) * 1000.0
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    perf = {
        "checkpoint_mb": round(ckpt_file.stat().st_size / (1024 ** 2), 2),
        "inference_peak_vram_mb": round(peak_mb, 2),
        "gpu_tta_ms_per_image": round(gpu_tta_ms, 2),
        "cpu_tta_ms_per_image": round(cpu_tta_ms, 1),
    }
    (P7 / "ta_compatibility_probe.json").write_text(json.dumps(perf, indent=2), encoding="utf-8")
    print("\n=== TA Hardware Compatibility Probe ===")
    print(json.dumps(perf, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_P7)
    args = parser.parse_args()
    main(args.export_dir)
