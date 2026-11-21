"""Freeze Weighted BCE and verify two notebooks in a relocated, clean copy."""
import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import nbformat
from nbclient import NotebookClient
from PIL import Image


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PACKAGE = HERE / "submission_candidate"
BASE = ROOT / "baseline"
WEIGHTED = ROOT / "phase2/weighted"
ARCHIVE = ROOT / "AIGC Detection Dataset 2026.zip"


def prepare_package():
    for name in ("project.py", "prepare_data.py", "split.json"):
        shutil.copyfile(BASE / name, PACKAGE / name)
    for name in ("model.pth", "config.json", "training_history.csv", "training_summary.json",
                 "validation_by_image.csv"):
        shutil.copyfile(WEIGHTED / name, PACKAGE / name)
    shutil.copyfile(ROOT / "phase3/resolution384/native_comparison.csv", PACKAGE / "historical_comparison.csv")
    config_path = PACKAGE / "config.json"
    config = json.loads(config_path.read_text())
    config.update(model="project.py:SmallUNet", split="split.json")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    test = nbformat.read(BASE / "test.ipynb", as_version=4)
    test.cells[1].source = test.cells[1].source.replace('Path("data/test_images")', 'Path("test_images")')
    for cell in test.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    nbformat.write(test, PACKAGE / "test.ipynb")

    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    notebook = nbformat.v4.new_notebook(
        metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
        cells=[
            md("""# CS5489 Course Project — AIGC Localization

**Group name:** [Canvas group name]  
**Members:** [Member names]

We predict a binary mask of edited pixels from each RGB image. This notebook documents the frozen Weighted BCE candidate. Run all cells to inspect the measured results. Enable the last cell's training toggle to reproduce the model with the official labelled archive."""),
            md("""## Data, model, and selection

The fixed split contains 4,500 training and 500 validation images. We resize full RGB images to 256 × 256 with bilinear interpolation and scale them to [0, 1]. Binary training masks use nearest-neighbor resizing. A randomly initialized U-Net uses two max-pool stages and two skip connections. The experiment uses course data and random initialization.

The training objective combines positive-weighted binary cross-entropy with per-image soft Dice loss. The positive weight is the square root of the ratio of background to foreground training pixels. We use Adam at a learning rate of 0.001, batch size 8, seed 5489 and 10 epochs. The checkpoint with the highest 256 × 256 validation micro-IoU at threshold 0.5 is retained. At inference, logits are resized to the original image shape before thresholding."""),
            code("""from pathlib import Path
import json
import pandas as pd
from IPython.display import display

config = json.loads(Path("config.json").read_text())
history = pd.read_csv("training_history.csv")
summary = json.loads(Path("training_summary.json").read_text())
assert history.loc[history.val_iou.idxmax(), "epoch"] == summary["best_epoch"]
print(f'Positive BCE weight: {config["bce_pos_weight"]:.6f}')
print(f'Best checkpoint: epoch {summary["best_epoch"]}, 256-grid IoU {summary["best_val_iou_256"]:.2%}')
display(history)"""),
            code("""import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
for ax, column, label in zip(
    axes, ("train_weighted_loss", "val_loss", "val_iou"),
    ("Weighted training loss", "Unweighted validation loss", "Validation micro-IoU"),
):
    ax.plot(history.epoch, history[column], "o-")
    ax.set(xlabel="Epoch", ylabel=label)
    ax.grid(alpha=0.25)
plt.show()"""),
            md("""## Original-resolution validation

Micro-IoU pools the pixel counts over all 500 images. Mean image IoU weights images equally. Both-empty prediction/target pairs would score 1 in our local protocol. All validation masks contain foreground."""),
            code("""counts = pd.read_csv("validation_by_image.csv", dtype={"ID": str})
tp, fp, fn = counts[["TP", "FP", "FN"]].sum()
comparison = pd.read_csv("historical_comparison.csv")
assert len(counts) == counts.ID.nunique() == 500
assert ((counts.TP + counts.FN) > 0).all()
assert abs(tp / (tp + fp + fn) - comparison.loc[comparison.model.eq("Weighted"), "iou"].item()) < 1e-12
print(f"Uniform foreground micro-IoU: {(tp + fn) / counts.pixels.sum():.2%}")
small = counts.loc[counts.target_fraction.lt(.01)]
print(f"Weighted BCE IoU for edits below 1% ({len(small)} images): {small.TP.sum() / small[['TP', 'FP', 'FN']].to_numpy().sum():.2%}")
display(comparison[["model", "iou", "mean_image_iou", "dice", "precision", "recall", "FP", "FN"]])"""),
            md("""Weighted BCE increased recall and false positives relative to the unweighted 256 × 256 model. It led the three archived checkpoints on micro-IoU and Dice, while the 384 × 384 candidate led on mean image IoU. Localization of small edits remains weak. Model selection uses the held-out training split. The public images serve to verify the inference pipeline."""),
            md("""## Reproduce training

Place the official ZIP beside this notebook or edit its path below. The training command creates a cache with the same preprocessing and writes a new checkpoint to reproduction/model.pth. The frozen model.pth serves test.ipynb."""),
            code("""import subprocess
import sys

RETRAIN = False
ARCHIVE = Path("AIGC Detection Dataset 2026.zip")
if RETRAIN:
    subprocess.run([sys.executable, "-B", "train_candidate.py", "--archive", str(ARCHIVE)], check=True)"""),
        ],
    )
    nbformat.write(notebook, PACKAGE / "project_report.ipynb")

    (PACKAGE / "python_environment.txt").write_text(
        """CS5489 frozen Weighted BCE candidate
Verified environment: Windows x64, Conda cs5489, Python 3.11.16
torch==2.9.1+cu128 (CUDA 12.8 build, verified on RTX 5090)
numpy==2.4.6
pandas==2.2.3
Pillow==12.3.0
matplotlib==3.11.1
ipykernel==7.3.0
Notebook verification: nbformat==5.11.1, nbclient==0.11.0

Use a compatible official PyTorch build for the target CUDA or CPU environment:
https://pytorch.org/get-started/previous-versions/#v291
Inference selects CUDA if available and CPU otherwise. Training reproduction
was configured for the verified GPU stack. Inference works offline.
""", encoding="utf-8")
    (PACKAGE / "README.md").write_text(
        """# CS5489 frozen Weighted BCE candidate

Open the notebooks with this directory as the working directory. Put test images in
test_images/ or set test_dir in test.ipynb. Leave model_path at model.pth when the
checkpoint is in this directory. Run all cells to write submission.csv in the
TA template format (ID,Predicted). Predictions use native-size, 1-based
column-major RLE. The notebook runs on RGB images without test labels.

project_report.ipynb displays the archived training and validation results and has an
optional training reproduction cell. Add the Canvas group name and members to
the submission copies. The course archive and weights are supplied separately.

model.pth is the Phase 2 epoch 7 checkpoint copied byte for byte. Keep project.py
beside the notebooks. Training reproduction also uses prepare_data.py,
train_candidate.py and split.json. python_environment.txt lists the versions.
SHA256SUMS.txt records the package file checksums.
""", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(filename, relocated, output):
    book = nbformat.read(relocated / filename, as_version=4)
    client = NotebookClient(book, kernel_name="python3", timeout=180,
                            resources={"metadata": {"path": str(relocated)}})
    client.execute()
    assert all(cell.get("execution_count") is not None for cell in book.cells if cell.cell_type == "code")
    nbformat.write(book, output)


def verify_package():
    verification = HERE / "verification"
    verification.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="cs5489_relocated_", dir=HERE) as temporary:
        relocated = Path(temporary)
        for path in PACKAGE.iterdir():
            if path.is_file():
                shutil.copyfile(path, relocated / path.name)
        (relocated / "runtime").mkdir()
        (relocated / "test_images").mkdir()
        os.environ["JUPYTER_RUNTIME_DIR"] = str(relocated / "runtime")
        os.environ["MPLCONFIGDIR"] = str(relocated / "mpl")
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        os.environ.pop("PYTHONPATH", None)
        with ZipFile(ARCHIVE) as zipped:
            manifest, = [n for n in zipped.namelist() if n.endswith("/test.csv")]
            prefix = manifest.removesuffix("test.csv")
            meta = zipped.read(manifest)
            sample = zipped.read(prefix + "sample_submission.csv")
            public = list(csv.DictReader(meta.decode("utf-8-sig").splitlines()))
            assert len(public) == 500
            samples = [row for row in public if row["Usage"] == "Public"][:3]
            for row in samples:
                name = row["ID"] + ".webp"
                (relocated / "test_images" / name).write_bytes(zipped.read(prefix + "test/images/" + name))
        assert len(samples) == 3
        inference = relocated / "inference_only"
        inference.mkdir()
        for name in ("test.ipynb", "model.pth", "project.py"):
            shutil.copyfile(relocated / name, inference / name)
        shutil.move(str(relocated / "test_images"), str(inference / "test_images"))
        execute("test.ipynb", inference, verification / "test_executed_3_public.ipynb")
        print("Clean inference kernel passed", flush=True)
        execute("project_report.ipynb", relocated, verification / "project_report_executed.ipynb")
        print("Clean report kernel passed", flush=True)
        submission = inference / "submission.csv"
        with submission.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            assert reader.fieldnames == ["ID", "Predicted"]
        assert [row["ID"] for row in rows] == [row["ID"] for row in samples]
        details = []
        for row, expected in zip(rows, samples):
            identifier = row["ID"]
            with Image.open(inference / "test_images" / (identifier + ".webp")) as image:
                width, height = image.size
            assert (height, width) == (int(expected["height"]), int(expected["width"]))
            tokens = [int(token) for token in row["Predicted"].split()]
            assert len(tokens) % 2 == 0
            last = 0
            for start, length in zip(tokens[::2], tokens[1::2]):
                assert start > last and length > 0 and start + length - 1 <= height * width
                last = start + length - 1
            details.append({"ID": identifier, "original_shape": [height, width],
                            "foreground_pixels": sum(tokens[1::2]), "runs": len(tokens) // 2})
        shutil.copyfile(submission, verification / "submission_3_public_TA_format.csv")
        evidence = {
            "status": "PASS", "environment": sys.executable, "isolated_kernels": 2,
            "inference_inputs": ["test.ipynb", "project.py", "model.pth", "three public images"],
            "report_retrain_toggle": False, "shape_and_binary_checked_by_test_notebook": True,
            "headers_TA_format": reader.fieldnames, "rows": len(rows), "images": details,
            "course_ZIP_sample_header": sample.decode("utf-8-sig").splitlines()[0].split(","),
            "sample_equals_test_metadata": sample == meta,
            "checkpoint_matches_weighted": sha(PACKAGE / "model.pth") == sha(WEIGHTED / "model.pth"),
            "checkpoint_sha256": sha(PACKAGE / "model.pth"), "csv_sha256": sha(submission),
        }
        assert evidence["checkpoint_matches_weighted"]
        (verification / "verification.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    shutil.copyfile(verification / "project_report_executed.ipynb", PACKAGE / "project_report.ipynb")
    files = sorted(path for path in PACKAGE.iterdir() if path.is_file() and path.name != "SHA256SUMS.txt")
    manifest = PACKAGE / "SHA256SUMS.txt"
    manifest.write_text("".join(f"{sha(path)}  {path.name}\n" for path in files), encoding="utf-8")
    with ZipFile(HERE / "submission_candidate.zip", "w", compression=ZIP_DEFLATED) as zipped:
        for path in (*files, manifest):
            zipped.write(path, "submission_candidate/" + path.name)
    with ZipFile(HERE / "submission_candidate.zip") as zipped:
        assert zipped.testzip() is None
        for path in files:
            assert hashlib.sha256(zipped.read("submission_candidate/" + path.name)).hexdigest() == sha(path)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    prepare_package()
    verify_package()
