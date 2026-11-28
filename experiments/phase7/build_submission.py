"""Assemble Phase 7 submission_candidate, build report & test notebooks, execute & verify end-to-end."""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import io
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
P7 = REPO / "experiments" / "phase7"
CAND = P7 / "submission_candidate"
DEFAULT_EXPORT_ROOT = REPO.parent / "Course_Project_export"


def execute_notebook_inproc(nb: nbf.NotebookNode, cwd: Path) -> nbf.NotebookNode:
    """Execute every code cell sequentially in-process and capture stdout, display(), and plt.show()."""
    prev_cwd = Path.cwd()
    prev_sys_path = list(sys.path)
    os.chdir(cwd)
    if str(cwd) not in sys.path:
        sys.path.insert(0, str(cwd))

    ns: dict = {"__name__": "__main__"}
    exec_count = 0
    try:
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            exec_count += 1
            cell.execution_count = exec_count
            outputs = []
            stdout_buf = io.StringIO()

            def _display(*objs):
                for obj in objs:
                    data = {"text/plain": repr(obj)}
                    if hasattr(obj, "_repr_html_"):
                        data["text/html"] = obj._repr_html_()
                    outputs.append(nbf.v4.new_output("display_data", data=data, metadata={}))

            orig_show = plt.show

            def _show(*args, **kwargs):
                for fignum in plt.get_fignums():
                    fig = plt.figure(fignum)
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
                    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                    outputs.append(
                        nbf.v4.new_output(
                            "display_data",
                            data={"image/png": b64, "text/plain": "<Figure>"},
                            metadata={},
                        )
                    )
                plt.close("all")

            # Patch IPython.display.display if imported inside cell
            import IPython.display as ipydisp
            orig_ipydisp = ipydisp.display
            ipydisp.display = _display
            plt.show = _show
            ns["display"] = _display
            try:
                with contextlib.redirect_stdout(stdout_buf):
                    exec(compile(cell.source, f"<cell {exec_count}>", "exec"), ns)
            finally:
                plt.show = orig_show
                ipydisp.display = orig_ipydisp

            text = stdout_buf.getvalue()
            if text:
                outputs.insert(0, nbf.v4.new_output("stream", name="stdout", text=text))
            cell.outputs = outputs
    finally:
        os.chdir(prev_cwd)
        sys.path[:] = prev_sys_path
    return nb


def make_test_notebook(test_dir_expr: str = 'Path("test_images")', model_path_expr: str = 'Path("model.pth")', output_csv_expr: str = 'Path("submission.csv")'):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python (cs5489)", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "version": "3.11.16"}

    nb.cells = [
        nbf.v4.new_markdown_cell(
            "**Group Name:** Course Project Group 5  \n"
            "**Members:** [Fill Canvas group member names before final upload]\n\n"
            "# CS5489 — Test Code Template (2026)\n\n"
            "**Purpose:** Verify that the submitted `model.pth` can be loaded and used to predict RLE masks "
            "from RGB test images. **No ground-truth labels are required.** TAs only need to change the paths below and Run All.\n\n"
            "**Model:** Phase 7B `ConvNeXt-Small Detail-UNet` (512×512, horizontal-flip TTA, threshold 0.50, pure PyTorch + torchvision, 100% offline)."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "from PIL import Image\n"
            "import torch\n"
            "from models import build_architecture, predict_mask\n\n"
            "# TA should only need to change these paths.\n"
            f"test_dir = {test_dir_expr}\n"
            f"model_path = {model_path_expr}\n"
            f"output_csv = {output_csv_expr}\n"
            "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n"
            "print(\"Device:\", device)"
        ),
        nbf.v4.new_markdown_cell(
            "## Student implementation\n\n"
            "Instantiates the architecture offline (`pretrained=False`), loads `model.pth`, and outputs a full-resolution "
            "2D binary array of shape `(original_height, original_width)` with values strictly in `{0, 1}`."
        ),
        nbf.v4.new_code_cell(
            "def build_model(arch=\"convnext_small_unet\"):\n"
            "    return build_architecture(arch, pretrained=False)\n\n\n"
            "def load_model(model_path, device):\n"
            "    checkpoint = torch.load(model_path, map_location=device, weights_only=True)\n"
            "    arch = checkpoint.get(\"arch\", \"convnext_small_unet\")\n"
            "    model = build_model(arch).to(device)\n"
            "    model.load_state_dict(checkpoint[\"model_state\"])\n"
            "    model.image_size = int(checkpoint.get(\"image_size\", 512))\n"
            "    model.threshold = float(checkpoint.get(\"threshold\", 0.5))\n"
            "    model.use_tta = bool(checkpoint.get(\"tta\", True))\n"
            "    model.eval()\n"
            "    return model\n\n\n"
            "def predict_binary_mask(model, image, device):\n"
            "    return predict_mask(\n"
            "        model, image, device,\n"
            "        size=model.image_size,\n"
            "        threshold=model.threshold,\n"
            "        tta=model.use_tta,\n"
            "    )"
        ),
        nbf.v4.new_markdown_cell(
            "## Shared RLE encoder (do not change)\n\n"
            "Uses **column-major** flattening, **1-based** starting positions, and `start length` pairs. Empty mask → empty string."
        ),
        nbf.v4.new_code_cell(
            "def mask_to_rle(mask):\n"
            "    mask = np.asarray(mask)\n"
            "    if mask.ndim != 2 or not np.isin(mask, [0, 1]).all():\n"
            "        raise ValueError(\"Predicted mask must be 2D and contain only 0/1\")\n"
            "    pixels = mask.astype(np.uint8).ravel(order=\"F\")\n"
            "    padded = np.pad(pixels, (1, 1), constant_values=0)\n"
            "    changes = np.flatnonzero(padded[1:] != padded[:-1])\n"
            "    starts = changes[::2] + 1\n"
            "    lengths = changes[1::2] - changes[::2]\n"
            "    return \" \".join(str(v) for pair in zip(starts, lengths) for v in pair)\n\n"
            "assert mask_to_rle(np.array([[0, 0, 1, 1, 1], [0, 0, 0, 0, 1]], dtype=np.uint8)) == \"5 1 7 1 9 2\"\n"
            "print(\"RLE encoder check passed\")"
        ),
        nbf.v4.new_markdown_cell(
            "## Inference verification and CSV export\n\n"
            "This cell must run **without access to any test ground-truth labels**. It checks every predicted mask shape "
            "and creates a Kaggle-compatible CSV with `ID,Predicted`. IDs are image filename stems and retain leading zeros."
        ),
        nbf.v4.new_code_cell(
            "def run_test(test_dir, model_path, output_csv, device):\n"
            "    if not test_dir.is_dir():\n"
            "        raise FileNotFoundError(f\"Test image directory not found: {test_dir}\")\n"
            "    if not model_path.is_file():\n"
            "        raise FileNotFoundError(f\"Checkpoint not found: {model_path}\")\n\n"
            "    extensions = {\".png\", \".jpg\", \".jpeg\", \".webp\", \".bmp\", \".tif\", \".tiff\"}\n"
            "    images = sorted(p for p in test_dir.iterdir()\n"
            "                    if p.is_file() and p.suffix.lower() in extensions)\n"
            "    if not images:\n"
            "        raise ValueError(f\"No images found in {test_dir}\")\n"
            "    ids = [p.stem for p in images]\n"
            "    if len(ids) != len(set(ids)):\n"
            "        raise ValueError(\"Duplicate image IDs across file extensions\")\n\n"
            "    model = load_model(model_path, device)\n"
            "    rows = []\n"
            "    for index, path in enumerate(images, 1):\n"
            "        with Image.open(path) as im:\n"
            "            rgb = im.convert(\"RGB\")\n"
            "            height, width = rgb.height, rgb.width\n"
            "            with torch.inference_mode():\n"
            "                mask = np.asarray(predict_binary_mask(model, rgb, device))\n"
            "        if mask.shape != (height, width):\n"
            "            raise ValueError(f\"{path.name}: expected {(height, width)}, got {mask.shape}\")\n"
            "        rle = mask_to_rle(mask)\n"
            "        rows.append({\"ID\": path.stem, \"Predicted\": rle})\n"
            "        if index == 1 or index % 100 == 0 or index == len(images):\n"
            "            print(f\"Processed {index}/{len(images)} images\")\n\n"
            "    df = pd.DataFrame(rows, columns=[\"ID\", \"Predicted\"])\n"
            "    df.to_csv(output_csv, index=False)\n"
            "    print(f\"PASS: checkpoint loaded and {len(df)} masks predicted\")\n"
            "    print(f\"Saved: {output_csv}\")\n"
            "    print(df.assign(Predicted=df.Predicted.str.slice(0, 70)).head())\n"
            "    return df\n\n"
            "submission = run_test(test_dir, model_path, output_csv, device)"
        ),
        nbf.v4.new_markdown_cell(
            "## Submission checklist\n\n"
            "- [x] Run all cells successfully with the submitted `model.pth`.\n"
            "- [x] Model loads offline in evaluation mode without training data or hidden labels.\n"
            "- [x] Every test image produces a full-resolution 0/1 mask.\n"
            "- [x] `submission.csv` contains one row per test image, with columns `ID,Predicted`.\n"
            "- [x] `models.py` and `python_environment.txt` accompany `test.ipynb` and `model.pth`.\n\n"
            "**For TAs:** Change `test_dir` and `model_path` in the configuration cell, then Run All."
        ),
    ]
    return nb


def make_report_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python (cs5489)", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "version": "3.11.16"}

    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# CS5489 Course Project — AI-Generated Content (AIGC) Localization\n\n"
            "**Group Name:** Course Project Group 5  \n"
            "**Members:** [Fill Canvas group member names before final upload]\n\n"
            "## 1. Problem Statement & Experimental Protocol\n\n"
            "Given an RGB photograph containing local AI-generated edits (object insertion, object removal, or inpainting), "
            "our goal is dense binary prediction: output a full-resolution 0/1 mask (`0`: authentic photographic pixel, "
            "`1`: AI-edited pixel) encoded as 1-based column-major RLE.\n\n"
            "- **Strict Data Hygiene:** All experiments across Phases 1–7 use the **exact same fixed 4,500 train / 500 validation split** "
            "(`split.json`, seed `5489`). Public test images are strictly excluded from training, model selection, and threshold analysis.\n"
            "- **Native-Resolution Evaluation:** While models are selected during training by grid-level validation `micro-IoU` at threshold `0.50`, "
            "all final comparisons are computed on the **500 original-resolution validation masks** across multiple metrics: "
            "`micro-IoU`, `mean image IoU`, `micro-Dice`, `mean image Dice`, `Precision`, `Recall`, `Pixel Accuracy`, `PR-AUC`, and five edited-area bins (`<1%` to `30–100%`).\n"
            "- **Offline TA Compatibility:** The winning Phase 7 model uses only `torch` and `torchvision` (matching TA's reference environment), "
            "requires `< 450 MB` GPU VRAM during inference (compatible with RTX A1000 8 GB and CPU fallback), and loads completely offline (`pretrained=False`)."
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Full Experimental Progression (Phases 1–7 & Teammate Tier-0 Audit)\n\n"
            "Phases 1–6 systematically diagnosed a randomly initialized 118K-parameter `SmallUNet` across loss weighting, input resolution (384), "
            "SRM high-pass filtering, patch-based cropping, and deeper 3-scale skip connections—all plateauing at `10.5%–14.1%` native micro-IoU "
            "(barely above the `13.07%` all-foreground baseline). A teammate's Tier-0 post-processing study pushed a CPU-reproduced Weighted BCE checkpoint "
            "from `12.92%` to `17.90%` by lowering the threshold to `0.35`, but at the cost of severe over-segmentation (`39.65%` predicted foreground vs `13.07%` ground truth) "
            "and collapsing `<5%` small-edit IoU by half.\n\n"
            "**Phase 7 Hypothesis:** Replacing scratch-initialized tiny CNNs with ImageNet-1K pretrained hierarchical encoders at `512×512`, "
            "paired with matched geometric/photometric augmentation, a balanced `BCE + per-image Dice + micro-Dice` objective, and a parallel high-resolution "
            "spatial detail branch, should simultaneously unlock semantic consistency cues and sharp pixel-level boundaries."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from IPython.display import display\n\n"
            "comp = pd.read_csv(\"phase_comparison.csv\")\n"
            "pct_cols = [\"micro_iou\", \"mean_img_iou\", \"micro_dice\", \"precision\", \"recall\", \"pred_fg\", \"pix_acc\", \"pr_auc\", \"bin_<1%\", \"bin_1-5%\", \"bin_30-100%\"]\n"
            "styled = comp[[\"stage\"] + pct_cols].copy()\n"
            "for c in pct_cols:\n"
            "    styled[c] = styled[c].map(lambda v: f\"{v:.2%}\" if pd.notna(v) else \"—\")\n"
            "display(styled)"
        ),
        nbf.v4.new_code_cell(
            "fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), layout=\"constrained\")\n"
            "pick = [\n"
            "    (\"Phase 1\\nV1 256\", 0.1336, 0.1430, 0.2358),\n"
            "    (\"Phase 2\\nWeighted\", 0.1405, 0.1439, 0.2464),\n"
            "    (\"Teammate\\nTier-0\", 0.1790, 0.1548, 0.3037),\n"
            "    (\"Phase 7A\\nResNet34\", 0.4523, 0.4209, 0.6229),\n"
            "    (\"Phase 7C\\nCNX+SRM\", 0.8051, 0.7092, 0.8920),\n"
            "    (\"Phase 7B\\nConvNeXt-S\", 0.8101, 0.7137, 0.8951),\n"
            "]\n"
            "x = np.arange(len(pick))\n"
            "w = 0.26\n"
            "axes[0].bar(x - w, [p[1] * 100 for p in pick], w, label=\"Native micro-IoU\")\n"
            "axes[0].bar(x, [p[2] * 100 for p in pick], w, label=\"Mean Image IoU\")\n"
            "axes[0].bar(x + w, [p[3] * 100 for p in pick], w, label=\"Native micro-Dice\")\n"
            "axes[0].axhline(13.07, color=\"gray\", linestyle=\"--\", linewidth=1, label=\"All-FG prior (13.07%)\")\n"
            "axes[0].set_xticks(x)\n"
            "axes[0].set_xticklabels([p[0] for p in pick], fontsize=8.5)\n"
            "axes[0].set(ylabel=\"Score (%)\", ylim=(0, 100), title=\"Validation Progression (500 Native-Res Images)\")\n"
            "axes[0].legend(fontsize=8)\n"
            "axes[0].grid(axis=\"y\", alpha=0.25)\n\n"
            "bins = [\"<1%\", \"1-5%\", \"5-15%\", \"15-30%\", \"30-100%\"]\n"
            "bx = np.arange(len(bins))\n"
            "for label, row_name, style in [\n"
            "    (\"Phase 2 Weighted (0.50)\", \"Phase 2: Weighted BCE 256 (0.50)\", \"o--\"),\n"
            "    (\"Teammate Tier-0 (0.35)\", \"Teammate Tier-0 (CPU Weighted, 0.35+TTA+fill)\", \"s--\"),\n"
            "    (\"Phase 7A ResNet34 (TTA)\", \"Phase 7A: ResNet34-UNet 512 (TTA@0.50)\", \"^-\"),\n"
            "    (\"Phase 7C ConvNeXt-S+SRM\", \"Phase 7C: ConvNeXt-Small+SRM Detail-UNet 512 (TTA@0.50)\", \"d-\"),\n"
            "    (\"Phase 7B ConvNeXt-S (Winner)\", \"Phase 7B: ConvNeXt-Small Detail-UNet 512 (TTA@0.50)\", \"o-\"),\n"
            "]:\n"
            "    r = comp.loc[comp[\"stage\"] == row_name].iloc[0]\n"
            "    axes[1].plot(bx, [r[f\"bin_{b}\"] * 100 for b in bins], style, linewidth=2, markersize=5, label=label)\n"
            "axes[1].set_xticks(bx)\n"
            "axes[1].set_xticklabels(bins)\n"
            "axes[1].set(xlabel=\"Ground-Truth Edited Area Fraction\", ylabel=\"Native micro-IoU (%)\", ylim=(0, 100), title=\"Performance by Edited Region Size\")\n"
            "axes[1].legend(fontsize=8)\n"
            "axes[1].grid(alpha=0.25)\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. Winning Candidate Training Dynamics & Threshold Calibration (Phase 7B)\n\n"
            "**Architecture (`ConvNeXt-Small Detail-UNet`):**\n"
            "- Pretrained `torchvision.models.convnext_small` (`IMAGENET1K_V1`, 7×7 depthwise convs + LayerNorm, strides 4/8/16/32).\n"
            "- **High-Resolution Spatial Detail Stem:** Two parallel convolutional stages at stride 1 (24 ch) and stride 2 (48 ch) fused directly into the final U-Net decoder stages so small-edit boundaries are preserved despite ConvNeXt's stride-4 patchify stem.\n"
            "- **Training Recipe:** `512×512` input, AdamW (`backbone_lr=1e-4`, `decoder_lr=5e-4`, `weight_decay=2e-2`), 2-epoch warmup + cosine schedule for 25 epochs, `bfloat16` AMP, horizontal flip + random resized crop + mild color jitter.\n\n"
            "**Controlled SRM Ablation (Exp 7B vs 7C):** Adding fixed 5×5 SRM high-pass residual filters to the detail stem slightly decreased native micro-IoU (`81.01% → 80.51%`) and hurt `<1%` small edits (`30.58% → 22.25%`), confirming that at `512×512` resized input, learned multi-scale RGB features outperform fixed high-pass filters."
        ),
        nbf.v4.new_code_cell(
            "config = json.loads(Path(\"config.json\").read_text(encoding=\"utf-8\"))\n"
            "history = pd.read_csv(\"training_history.csv\")\n"
            "train_sum = json.loads(Path(\"training_summary.json\").read_text(encoding=\"utf-8\"))\n"
            "val_sum = json.loads(Path(\"validation_summary.json\").read_text(encoding=\"utf-8\"))\n"
            "ta_probe = json.loads(Path(\"ta_compatibility_probe.json\").read_text(encoding=\"utf-8\"))\n\n"
            "print(f\"Architecture: {config['arch']} ({config['parameters']:,} params), input={config['input_size']}x{config['input_size']}\")\n"
            "print(f\"Best epoch: {train_sum['best_epoch']}/{config['epochs']} | 512-grid micro-IoU: {train_sum['best_val_micro_iou_512']:.2%} | Train time: {train_sum['training_seconds']:.1f}s\")\n"
            "t = val_sum['tta_050']\n"
            "print(f\"Native 500-img (TTA@0.50): micro-IoU={t['micro_iou']:.2%}, mean-img IoU={t['mean_img_iou']:.2%}, micro-Dice={t['micro_dice']:.2%}, Precision={t['precision']:.2%}, Recall={t['recall']:.2%}, PR-AUC={val_sum['pr_auc_grid_tta']:.2%}\")\n"
            "print(f\"5-fold OOF CV (TTA): chosen={val_sum['oof_cv']['tta']['chosen_thrs']}, OOF micro-IoU={val_sum['oof_cv']['tta']['micro_iou']:.2%} (vs fixed 0.50: {t['micro_iou']:.2%})\")\n"
            "print(f\"TA hardware probe: ckpt={ta_probe['checkpoint_mb']} MB, peak inference VRAM={ta_probe['inference_peak_vram_mb']} MB, CPU TTA={ta_probe['cpu_tta_ms_per_image']} ms/img\")\n\n"
            "fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), layout=\"constrained\")\n"
            "axes[0].plot(history.epoch, history.train_loss, \"o-\", ms=3, label=\"Train\")\n"
            "axes[0].plot(history.epoch, history.val_loss, \"o-\", ms=3, label=\"Validation\")\n"
            "axes[0].set(xlabel=\"Epoch\", ylabel=\"Loss\", title=\"BCE + Dice Loss\")\n"
            "axes[0].legend(); axes[0].grid(alpha=0.25)\n\n"
            "axes[1].plot(history.epoch, history.val_micro_iou * 100, \"o-\", ms=3, label=\"Micro-IoU\")\n"
            "axes[1].plot(history.epoch, history.val_mean_img_iou * 100, \"s--\", ms=3, label=\"Mean Image IoU\")\n"
            "axes[1].set(xlabel=\"Epoch\", ylabel=\"IoU (%)\", title=\"Validation IoU (512 Grid)\")\n"
            "axes[1].legend(); axes[1].grid(alpha=0.25)\n\n"
            "sweep = pd.read_csv(\"threshold_sweep.csv\")\n"
            "sw_tta = sweep[sweep[\"mode\"] == \"tta\"]\n"
            "axes[2].plot(sw_tta.threshold, sw_tta.micro_iou * 100, \"o-\", label=\"micro-IoU\")\n"
            "axes[2].plot(sw_tta.threshold, sw_tta.mean_img_iou * 100, \"s--\", label=\"Mean Img IoU\")\n"
            "axes[2].axvline(0.50, color=\"red\", linestyle=\":\", label=\"Protocol thr=0.50\")\n"
            "axes[2].set(xlabel=\"Probability Threshold\", ylabel=\"Native IoU (%)\", title=\"Threshold Robustness (TTA)\")\n"
            "axes[2].legend(fontsize=8); axes[2].grid(alpha=0.25)\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## 4. Per-Image Distribution & Failure Analysis\n\n"
            "Understanding *where* the model succeeds and fails is essential:\n"
            "- **High precision & sharp localization:** Across the 500 validation images, **median per-image IoU is 87.32%**, "
            "**316/500 images (63.2%) exceed 0.80 IoU**, and **388/500 (77.6%) exceed 0.50 IoU**. Even on `<1%` ultra-small edits, "
            "clearly visible edits such as `004048` (`0.57%` area) achieve `96.22%` IoU.\n"
            "- **Primary failure mode (70/500 images with IoU < 0.20):** The failure subset has a median edited area of **`1.29%`** "
            "(4× smaller than the overall validation median of `5.22%`), with `28/70` below `1%` area and `54/70` (`77%`) being misses/under-predictions "
            "(`31` empty predictions, `18` wrong-region false alarms). Representative failures include ultra-small edits (`004657`, `0.32%`; `003703`, `0.60%`) "
            "and subtle seamless background removal (`002578`, `6.42%`), where `512×512` downsampling attenuates fine-grained boundary/texture cues."
        ),
        nbf.v4.new_code_cell(
            "by_img = pd.read_csv(\"validation_by_image.csv\", dtype={\"ID\": str})\n"
            "fail = json.loads(Path(\"failure_analysis.json\").read_text(encoding=\"utf-8\"))\n"
            "print(\"Failure analysis summary:\", json.dumps(fail, indent=2))\n\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.8), layout=\"constrained\")\n"
            "axes[0].hist(by_img[\"iou\"] * 100, bins=25, edgecolor=\"black\", alpha=0.8)\n"
            "axes[0].axvline(by_img[\"iou\"].median() * 100, color=\"red\", linestyle=\"--\", label=f\"Median: {by_img['iou'].median():.2%}\")\n"
            "axes[0].axvline(by_img[\"iou\"].mean() * 100, color=\"orange\", linestyle=\":\", label=f\"Mean: {by_img['iou'].mean():.2%}\")\n"
            "axes[0].set(xlabel=\"Per-Image Native IoU (%)\", ylabel=\"Image Count\", title=\"Per-Image IoU Distribution (500 Val Images)\")\n"
            "axes[0].legend(); axes[0].grid(alpha=0.25)\n\n"
            "axes[1].scatter(by_img[\"target_fraction\"] * 100, by_img[\"iou\"] * 100, s=14, alpha=0.6)\n"
            "axes[1].set_xscale(\"log\")\n"
            "axes[1].set(xlabel=\"Ground-Truth Edited Area (%, log scale)\", ylabel=\"Per-Image Native IoU (%)\", title=\"IoU vs. Edited Region Area\")\n"
            "axes[1].grid(alpha=0.25)\n"
            "plt.show()\n\n"
            "print(\"Worst 10 validation failure cases:\")\n"
            "display(by_img.sort_values(\"iou\").head(10)[[\"ID\", \"target_fraction\", \"predicted_fraction\", \"iou\", \"dice\", \"TP\", \"FP\", \"FN\"]])"
        ),
        nbf.v4.new_markdown_cell(
            "## 5. Reproducible Training Pipeline\n\n"
            "The full model definition (`models.py`), cache builder (`prepare_cache.py`), training script (`train_phase7.py`), "
            "and native-resolution evaluator (`evaluate_phase7.py`) are bundled beside this notebook. "
            "Set `RETRAIN = True` and point `ARCHIVE` to `AIGC Detection Dataset 2026.zip` to reproduce training end-to-end "
            "into `reproduction/model.pth` without overwriting the frozen submission artifacts."
        ),
        nbf.v4.new_code_cell(
            "import subprocess\n"
            "import sys\n\n"
            "RETRAIN = False\n"
            "ARCHIVE = Path(\"AIGC Detection Dataset 2026.zip\")\n"
            "if RETRAIN:\n"
            "    subprocess.run([sys.executable, \"prepare_cache.py\", \"--archive\", str(ARCHIVE), \"--out\", \"cache512\", \"--size\", \"512\"], check=True)\n"
            "    subprocess.run([\n"
            "        sys.executable, \"train_phase7.py\",\n"
            "        \"--arch\", \"convnext_small_unet\",\n"
            "        \"--cache\", \"cache512\",\n"
            "        \"--split\", \"split.json\",\n"
            "        \"--out\", \"reproduction\",\n"
            "        \"--ckpt-dir\", \"reproduction\",\n"
            "        \"--epochs\", \"25\", \"--batch-size\", \"16\", \"--lr\", \"1e-4\", \"--weight-decay\", \"2e-2\",\n"
            "    ], check=True)\n"
            "    subprocess.run([\n"
            "        sys.executable, \"evaluate_phase7.py\",\n"
            "        \"--ckpt\", \"reproduction/model.pth\",\n"
            "        \"--archive\", str(ARCHIVE),\n"
            "        \"--split\", \"split.json\",\n"
            "        \"--out\", \"reproduction\",\n"
            "    ], check=True)\n"
            "else:\n"
            "    print(\"RETRAIN is False; displaying verified archived outputs from the frozen Phase 7B run.\")"
        ),
    ]
    return nb


def main(export_root: Path = DEFAULT_EXPORT_ROOT):
    t0 = time.perf_counter()
    EXPORT_P7 = export_root / "phase7"
    BUNDLE = EXPORT_P7 / "canvas_submission_bundle"
    ZIP_PATH = export_root / "AIGC Detection Dataset 2026.zip"
    PUBLIC_TEST_DIR = EXPORT_P7 / "public_test_images"
    WIN_CKPT = EXPORT_P7 / "convnext_small_512" / "model.pth"
    CAND.mkdir(parents=True, exist_ok=True)
    BUNDLE.mkdir(parents=True, exist_ok=True)

    # 1. Populate submission_candidate (Git-tracked lightweight files)
    for src in [
        P7 / "models.py",
        P7 / "prepare_cache.py",
        P7 / "train_phase7.py",
        P7 / "evaluate_phase7.py",
        REPO / "experiments" / "baseline" / "split.json",
        P7 / "phase_comparison.csv",
        P7 / "phase7_comparison.png",
        P7 / "failure_analysis.json",
        P7 / "ta_compatibility_probe.json",
        P7 / "convnext_small_512" / "config.json",
        P7 / "convnext_small_512" / "training_history.csv",
        P7 / "convnext_small_512" / "training_summary.json",
        P7 / "convnext_small_512" / "validation_summary.json",
        P7 / "convnext_small_512" / "validation_by_image.csv",
        P7 / "convnext_small_512" / "threshold_sweep.csv",
        P7 / "convnext_small_512" / "training_curves.png",
    ]:
        shutil.copy2(src, CAND / src.name)

    py_env_txt = (
        "CS5489 Phase 7 frozen ConvNeXt-Small Detail-UNet candidate\n"
        "Verified environment: Windows x64, Conda cs5489, Python 3.11.16\n"
        "torch==2.9.1+cu128 (compatible with TA torch>=2.9.0, CUDA or CPU)\n"
        "torchvision==0.24.1+cu128 (compatible with TA torchvision>=0.24.0)\n"
        "numpy==2.4.6\n"
        "pandas==2.2.3\n"
        "Pillow==12.3.0\n"
        "matplotlib==3.11.1\n"
        "ipykernel==7.3.0\n"
        "Notebook execution: nbformat==5.11.1, nbclient==0.11.0\n\n"
        "Notes for TAs:\n"
        "- Inference uses ONLY torch, torchvision, numpy, pandas, Pillow (no scipy/timm/smp required).\n"
        "- test.ipynb instantiates ConvNeXt-Small with pretrained=False and loads model.pth 100% offline.\n"
        "- Peak GPU inference VRAM is ~428 MB (fits comfortably on RTX A1000 8 GB and runs on CPU fallback).\n"
    )
    (CAND / "python_environment.txt").write_text(py_env_txt, encoding="utf-8")

    cand_readme = (
        "# CS5489 Phase 7 Frozen Submission Candidate (`ConvNeXt-Small Detail-UNet`)\n\n"
        "- **Native 500-image validation micro-IoU:** **81.01%** (mean image IoU **71.37%**, micro-Dice **89.51%**, Precision **94.66%**, Recall **84.89%**, PR-AUC **94.60%**).\n"
        "- **Offline TA inference:** Place `model.pth` (SHA-256 recorded in `docs/provenance/local_artifact_SHA256SUMS.txt`) beside `test.ipynb` and `models.py`. "
        "Update `test_dir` in cell 1 of `test.ipynb` and Run All to export `submission.csv` (`ID,Predicted`, 1-based column-major RLE).\n"
        "- **Project report:** `project_report.ipynb` contains full executed outputs, Phase 1–7 comparisons, ablations, failure analysis, and an optional `RETRAIN` entrypoint.\n"
        "- **Before Canvas upload:** Fill in your Canvas group member names in the header cells of `project_report.ipynb` and `test.ipynb`.\n"
    )
    (CAND / "README.md").write_text(cand_readme, encoding="utf-8")

    # Write clean source notebooks into CAND
    nbf.write(make_test_notebook(), CAND / "test.ipynb")
    nbf.write(make_report_notebook(), CAND / "project_report.ipynb")

    # Execute project_report.ipynb in CAND so it carries verified execution outputs!
    print("Executing project_report.ipynb...", flush=True)
    rep_nb = nbf.read(CAND / "project_report.ipynb", as_version=4)
    execute_notebook_inproc(rep_nb, CAND)
    nbf.write(rep_nb, CAND / "project_report.ipynb")
    print("project_report.ipynb executed and saved with outputs!", flush=True)

    # 2. Extract 500 public test images locally (outside Git) for real end-to-end test.ipynb verification
    if not PUBLIC_TEST_DIR.exists() or len(list(PUBLIC_TEST_DIR.glob("*.webp"))) < 500:
        PUBLIC_TEST_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ZIP_PATH) as zf:
            members = [m for m in zf.namelist() if m.startswith("AIGC Detection Dataset 2026/test/images/") and m.endswith(".webp")]
            for m in members:
                (PUBLIC_TEST_DIR / Path(m).name).write_bytes(zf.read(m))
        print(f"Extracted {len(members)} public test images to {PUBLIC_TEST_DIR}", flush=True)

    # 3. Assemble complete ready-to-upload local Canvas bundle (including model.pth)
    for f in CAND.iterdir():
        if f.is_file():
            shutil.copy2(f, BUNDLE / f.name)
    shutil.copy2(WIN_CKPT, BUNDLE / "model.pth")

    # Execute test.ipynb against ALL 500 public test images in BUNDLE!
    print("Executing test.ipynb end-to-end on all 500 public test images...", flush=True)
    sub_out = EXPORT_P7 / "submission_phase7_convnext_small.csv"
    test_nb_exec = make_test_notebook(
        test_dir_expr=f'Path(r"{PUBLIC_TEST_DIR}")',
        model_path_expr='Path("model.pth")',
        output_csv_expr=f'Path(r"{sub_out}")',
    )
    execute_notebook_inproc(test_nb_exec, BUNDLE)
    nbf.write(test_nb_exec, BUNDLE / "test_executed_verified.ipynb")
    print("test.ipynb executed cleanly on all 500 public test images!", flush=True)

    # 4. Validate the generated Kaggle submission.csv thoroughly
    with zipfile.ZipFile(ZIP_PATH) as zf:
        test_df = pd.read_csv(io.BytesIO(zf.read("AIGC Detection Dataset 2026/test.csv")), dtype=str, keep_default_na=False)
    sub_df = pd.read_csv(sub_out, dtype=str, keep_default_na=False)
    assert list(sub_df.columns) == ["ID", "Predicted"], sub_df.columns
    assert len(sub_df) == 500, len(sub_df)
    assert list(sub_df["ID"]) == list(test_df["ID"]), "IDs or ordering mismatch!"

    empty_count = int((sub_df["Predicted"] == "").sum())
    fg_pixels = 0
    total_pixels = 0
    for (_, srow), (_, trow) in zip(sub_df.iterrows(), test_df.iterrows()):
        h, w = int(trow["height"]), int(trow["width"])
        total_pixels += h * w
        rle = srow["Predicted"]
        if rle:
            toks = [int(x) for x in rle.split()]
            assert len(toks) % 2 == 0
            fg_pixels += sum(toks[1::2])
            # Validate monotonic bounds
            prev_end = 0
            for st, ln in zip(toks[::2], toks[1::2]):
                assert st - 1 >= prev_end and ln > 0 and st - 1 + ln <= h * w
                prev_end = st - 1 + ln

    verif = {
        "rows": len(sub_df),
        "ids_match_test_csv": True,
        "empty_masks": empty_count,
        "predicted_fg_fraction_public_test": fg_pixels / total_pixels,
        "submission_csv_bytes": sub_out.stat().st_size,
        "submission_csv_sha256": hashlib.sha256(sub_out.read_bytes()).hexdigest(),
        "model_pth_sha256": hashlib.sha256((BUNDLE / "model.pth").read_bytes()).hexdigest(),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }
    (P7 / "public_test_verification.json").write_text(json.dumps(verif, indent=2), encoding="utf-8")
    (CAND / "public_test_verification.json").write_text(json.dumps(verif, indent=2), encoding="utf-8")
    print("\n=== PUBLIC TEST SUBMISSION VERIFICATION ===")
    print(json.dumps(verif, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    main(parser.parse_args().export_root)
