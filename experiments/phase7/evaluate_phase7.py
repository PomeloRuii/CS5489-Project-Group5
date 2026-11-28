"""Native-resolution validation evaluator with 5-fold OOF threshold calibration for CS5489 Phase 7."""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F

from models import build_architecture, decode_rle, image_to_tensor

BINS = [
    (0.0, 0.01, "<1%"),
    (0.01, 0.05, "1-5%"),
    (0.05, 0.15, "5-15%"),
    (0.15, 0.30, "15-30%"),
    (0.30, 1.01, "30-100%"),
]
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]


def pr_auc_from_hist(pos_hist: np.ndarray, neg_hist: np.ndarray) -> float:
    tp = np.cumsum(pos_hist[::-1])
    fp = np.cumsum(neg_hist[::-1])
    rec = tp / pos_hist.sum()
    prec = tp / np.maximum(tp + fp, 1)
    rec = np.concatenate([[0.0], rec])
    prec = np.concatenate([[1.0], prec])
    return float(np.trapezoid(prec, rec))


def summarize_array(arr: np.ndarray, gt_fracs: np.ndarray) -> dict:
    tp_i, fp_i, fn_i, pix_i = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    tp, fp, fn, pix = int(tp_i.sum()), int(fp_i.sum()), int(fn_i.sum()), int(pix_i.sum())
    tn = pix - tp - fp - fn
    union_i = tp_i + fp_i + fn_i
    img_ious = np.where(union_i > 0, tp_i / np.maximum(union_i, 1), 1.0)
    denom_d = 2 * tp_i + fp_i + fn_i
    img_dices = np.where(denom_d > 0, 2 * tp_i / np.maximum(denom_d, 1), 1.0)
    bins_out = {}
    for lo, hi, label in BINS:
        sel = (gt_fracs >= lo) & (gt_fracs < hi)
        btp, bfp, bfn = int(arr[sel, 0].sum()), int(arr[sel, 1].sum()), int(arr[sel, 2].sum())
        bins_out[label] = btp / (btp + bfp + bfn) if (btp + bfp + bfn) else 1.0
    return {
        "micro_iou": tp / (tp + fp + fn) if (tp + fp + fn) else 1.0,
        "mean_img_iou": float(img_ious.mean()),
        "micro_dice": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0,
        "mean_img_dice": float(img_dices.mean()),
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "pred_fg": (tp + fp) / pix,
        "pix_acc": (tp + tn) / pix,
        "empty_imgs": int(((tp_i + fp_i) == 0).sum()),
        "zero_tp_imgs": int((tp_i == 0).sum()),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "pixels": pix,
        "area_bins": bins_out,
    }


def evaluate_native(ckpt_path: Path, archive: Path, split_path: Path, out_dir: Path) -> None:
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    arch = ckpt["arch"]
    size = int(ckpt["image_size"])
    model = build_architecture(arch, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    split = json.loads(split_path.read_text(encoding="utf-8"))
    val_ids = split["validation"]
    n = len(val_ids)
    assert n == 500

    stats = {(mode, thr): np.zeros((n, 4), dtype=np.int64) for mode in ("base", "tta") for thr in THRESHOLDS}
    gt_fracs = np.zeros(n, dtype=np.float64)
    n_bins = 4096
    hists = {mode: [np.zeros(n_bins, dtype=np.int64), np.zeros(n_bins, dtype=np.int64)] for mode in ("base", "tta")}

    infer_time_base = 0.0
    infer_time_tta = 0.0

    with zipfile.ZipFile(archive) as zf:
        train_df = pd.read_csv(
            io.BytesIO(zf.read("AIGC Detection Dataset 2026/train.csv")),
            dtype={"ID": str, "annotation": str},
            keep_default_na=False,
        ).set_index("ID")

        with torch.inference_mode():
            for idx, img_id in enumerate(val_ids):
                raw = zf.read(f"AIGC Detection Dataset 2026/train/images/{img_id}.webp")
                with Image.open(io.BytesIO(raw)) as im:
                    rgb = im.convert("RGB")
                h, w = rgb.height, rgb.width
                truth = decode_rle(train_df.loc[img_id, "annotation"], h, w).astype(bool)
                gt_px = int(truth.sum())
                tot_px = truth.size
                gt_fracs[idx] = gt_px / tot_px
                truth_t = torch.from_numpy(truth).to(device)
                tr_s = np.array(
                    Image.fromarray(truth.astype(np.uint8)).resize((size, size), Image.Resampling.NEAREST),
                    dtype=bool,
                )

                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                tb0 = time.perf_counter()
                x = image_to_tensor(rgb, size).unsqueeze(0).to(device)
                prob_s = model(x)[0, 0].sigmoid()
                prob_full = F.interpolate(
                    prob_s[None, None], size=(h, w), mode="bilinear", align_corners=False
                )[0, 0]
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                infer_time_base += time.perf_counter() - tb0

                tt0 = time.perf_counter()
                x_flip = torch.flip(x, dims=[-1])
                prob_flip_s = torch.flip(model(x_flip)[0, 0].sigmoid(), dims=[-1])
                prob_tta_s = 0.5 * (prob_s + prob_flip_s)
                prob_tta_full = F.interpolate(
                    prob_tta_s[None, None], size=(h, w), mode="bilinear", align_corners=False
                )[0, 0]
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                infer_time_tta += time.perf_counter() - tt0

                for mode, ps in (("base", prob_s), ("tta", prob_tta_s)):
                    p_np = ps.cpu().numpy()
                    b_idx = np.clip((p_np * (n_bins - 1)).astype(np.int32), 0, n_bins - 1)
                    hists[mode][0] += np.bincount(b_idx[tr_s], minlength=n_bins)
                    hists[mode][1] += np.bincount(b_idx[~tr_s], minlength=n_bins)

                for thr in THRESHOLDS:
                    mb = prob_full >= thr
                    tp_b = int((mb & truth_t).sum().item())
                    fp_b = int((mb & ~truth_t).sum().item())
                    stats[("base", thr)][idx] = (tp_b, fp_b, gt_px - tp_b, tot_px)

                    mt = prob_tta_full >= thr
                    tp_t = int((mt & truth_t).sum().item())
                    fp_t = int((mt & ~truth_t).sum().item())
                    stats[("tta", thr)][idx] = (tp_t, fp_t, gt_px - tp_t, tot_px)

                if (idx + 1) % 100 == 0:
                    print(f"[{arch}] Native eval {idx + 1}/{n} ({time.perf_counter() - t0:.1f}s)", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_rows = []
    for mode in ("base", "tta"):
        for thr in THRESHOLDS:
            s = summarize_array(stats[(mode, thr)], gt_fracs)
            row = {"mode": mode, "threshold": thr, **{k: v for k, v in s.items() if k != "area_bins"}}
            for bname, bval in s["area_bins"].items():
                row[f"bin_{bname}"] = bval
            sweep_rows.append(row)
    pd.DataFrame(sweep_rows).to_csv(out_dir / "threshold_sweep.csv", index=False)

    # 5-fold OOF CV threshold check on validation set
    rng = np.random.default_rng(5489)
    folds = np.array_split(rng.permutation(n), 5)
    oof = {}
    for mode in ("base", "tta"):
        oof_arr = np.zeros((n, 4), dtype=np.int64)
        chosen = []
        for k in range(5):
            te_idx = folds[k]
            tr_idx = np.concatenate([folds[j] for j in range(5) if j != k])
            best_t, best_v = 0.5, -1.0
            for thr in THRESHOLDS:
                a = stats[(mode, thr)][tr_idx]
                v = a[:, 0].sum() / (a[:, 0].sum() + a[:, 1].sum() + a[:, 2].sum())
                if v > best_v:
                    best_v, best_t = v, thr
            chosen.append(best_t)
            oof_arr[te_idx] = stats[(mode, best_t)][te_idx]
        oof[mode] = {"chosen_thrs": chosen, **summarize_array(oof_arr, gt_fracs)}

    # Save per-image CSV for protocol threshold 0.50 (TTA) and base 0.50
    for mode, fname in (("base", "validation_by_image_base050.csv"), ("tta", "validation_by_image.csv")):
        arr = stats[(mode, 0.50)]
        by_img = pd.DataFrame({
            "ID": val_ids,
            "TP": arr[:, 0],
            "FP": arr[:, 1],
            "FN": arr[:, 2],
            "pixels": arr[:, 3],
            "target_fraction": gt_fracs,
            "predicted_fraction": (arr[:, 0] + arr[:, 1]) / arr[:, 3],
            "iou": arr[:, 0] / np.maximum(arr[:, 0] + arr[:, 1] + arr[:, 2], 1),
            "dice": (2 * arr[:, 0]) / np.maximum(2 * arr[:, 0] + arr[:, 1] + arr[:, 2], 1),
        })
        by_img.to_csv(out_dir / fname, index=False)

    summary = {
        "arch": arch,
        "image_size": size,
        "pr_auc_grid_base": pr_auc_from_hist(*hists["base"]),
        "pr_auc_grid_tta": pr_auc_from_hist(*hists["tta"]),
        "base_050": summarize_array(stats[("base", 0.50)], gt_fracs),
        "tta_050": summarize_array(stats[("tta", 0.50)], gt_fracs),
        "oof_cv": oof,
        "infer_seconds_500_base": round(infer_time_base, 2),
        "infer_seconds_500_tta": round(infer_time_base + infer_time_tta, 2),
    }
    (out_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    b = summary["base_050"]
    t = summary["tta_050"]
    print(f"\n=== [{arch} @ {size}x{size}] NATIVE 500-IMG SUMMARY ===", flush=True)
    print(
        f"Base@0.50: mIoU={b['micro_iou']:.2%}, imgIoU={b['mean_img_iou']:.2%}, "
        f"mDice={b['micro_dice']:.2%}, imgDice={b['mean_img_dice']:.2%}, "
        f"P={b['precision']:.2%}, R={b['recall']:.2%}, FG={b['pred_fg']:.2%}, "
        f"Acc={b['pix_acc']:.2%}, PR-AUC={summary['pr_auc_grid_base']:.2%}",
        flush=True,
    )
    print(
        f"TTA @0.50: mIoU={t['micro_iou']:.2%}, imgIoU={t['mean_img_iou']:.2%}, "
        f"mDice={t['micro_dice']:.2%}, imgDice={t['mean_img_dice']:.2%}, "
        f"P={t['precision']:.2%}, R={t['recall']:.2%}, FG={t['pred_fg']:.2%}, "
        f"Acc={t['pix_acc']:.2%}, PR-AUC={summary['pr_auc_grid_tta']:.2%}",
        flush=True,
    )
    print(
        f"OOF CV (TTA): chosen={oof['tta']['chosen_thrs']}, "
        f"mIoU={oof['tta']['micro_iou']:.2%}, imgIoU={oof['tta']['mean_img_iou']:.2%}",
        flush=True,
    )
    print(f"Area bins (TTA@0.50): { {k: f'{v:.2%}' for k, v in t['area_bins'].items()} }", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    evaluate_native(args.ckpt, args.archive, args.split, args.out)
