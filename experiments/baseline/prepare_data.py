import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from project import decode_rle


def prepare_data(archive, output, size=256, workers=6):
    """Validate the course archive and cache the resized RGB images and masks."""
    archive, output = Path(archive), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as zipped:
        names = set(zipped.namelist())
        train_csv, = [name for name in names if name.endswith("/train.csv")]
        root = train_csv.removesuffix("train.csv")
        labels = pd.read_csv(zipped.open(train_csv), dtype={"ID": str, "annotation": str}, keep_default_na=False)
        public = pd.read_csv(zipped.open(root + "test.csv"), dtype={"ID": str}, keep_default_na=False)
        sample = pd.read_csv(zipped.open(root + "sample_submission.csv"), dtype={"ID": str}, keep_default_na=False)
    if not labels["ID"].is_unique or not public["ID"].is_unique:
        raise ValueError("Duplicate IDs in the course CSVs")
    if set(labels.ID) & set(public.ID):
        raise ValueError("Training and public IDs overlap")
    if not (len(labels) == 5000 and len(public) == 500 and sample.equals(public)):
        raise ValueError("Unexpected dataset counts or public sample table")
    if not public["annotation"].eq("").all() or not public["Usage"].eq("Public").all():
        raise ValueError("Unexpected public test labels or usage")
    for split, frame in (("train", labels), ("test", public)):
        directory = f"{root}{split}/images/"
        expected = {f"{directory}{identifier}.webp" for identifier in frame.ID}
        actual = {name for name in names if name.startswith(directory) and name.endswith(".webp")}
        if expected != actual:
            raise ValueError(f"{split} CSV and archive image IDs differ")

    labels = labels.sort_values("ID").reset_index(drop=True)
    records = list(labels.itertuples(index=False))
    # Select non-square images across several ranges of foreground area.
    ratios = np.array([
        sum(int(length) for length in row.annotation.split()[1::2]) / (row.height * row.width)
        for row in records
    ])
    candidates = np.flatnonzero((labels.height != labels.width) & (ratios > 0.01) & (ratios < 0.85))
    ordered = candidates[np.argsort(ratios[candidates])]
    selected = ordered[np.linspace(0, len(ordered) - 1, 12, dtype=int)].tolist()

    def load(zipped, row):
        with Image.open(io.BytesIO(zipped.read(f"{root}train/images/{row.ID}.webp"))) as image:
            if image.size != (row.width, row.height):
                raise ValueError(f"{row.ID}: image size {image.size} differs from CSV dimensions")
            rgb = image.convert("RGB")
        try:
            mask = decode_rle(row.annotation, row.height, row.width)
        except ValueError as error:
            raise ValueError(f"{row.ID}: {error}") from error
        return rgb, mask

    fig, axes = plt.subplots(6, 4, figsize=(13, 16), layout="constrained")
    with ZipFile(archive) as zipped:
        for i, index in enumerate(selected):
            row = records[index]
            image, mask = load(zipped, row)
            # Round-trip the source-resolution mask to catch axis/index errors.
            flat = mask.ravel(order="F")
            changes = np.flatnonzero(np.diff(np.pad(flat, 1)))
            encoded = " ".join(str(x) for pair in zip(changes[::2] + 1, changes[1::2] - changes[::2]) for x in pair)
            if encoded != " ".join(row.annotation.split()):
                raise ValueError(f"{row.ID}: full-resolution RLE round-trip mismatch")
            image.thumbnail((480, 360), Image.Resampling.BILINEAR)
            mask_preview = np.asarray(Image.fromarray(mask).resize(image.size, Image.Resampling.NEAREST))
            lhs, rhs = axes[i // 2, (i % 2) * 2:(i % 2) * 2 + 2]
            lhs.imshow(image)
            lhs.set_title(row.ID)
            rhs.imshow(image)
            rhs.imshow(np.ma.masked_where(mask_preview == 0, mask_preview), cmap="autumn", vmin=0, vmax=1, alpha=0.45)
            rhs.set_title(f"Mask {ratios[index]:.1%}")
            lhs.axis("off")
            rhs.axis("off")
    fig.savefig(output / "annotation_check.png", dpi=125)
    plt.close(fig)
    print(f"Verified train/test IDs and 12 native RLE round-trips; preview saved to {output}.", flush=True)

    images = np.lib.format.open_memmap(
        output / "images.npy", mode="w+", dtype=np.uint8, shape=(len(labels), size, size, 3)
    )
    masks = np.lib.format.open_memmap(
        output / "masks.npy", mode="w+", dtype=np.uint8, shape=(len(labels), size, size)
    )
    with ZipFile(archive) as zipped, ThreadPoolExecutor(max_workers=workers) as executor:
        def resize_example(index):
            image, mask = load(zipped, records[index])
            images[index] = image.resize((size, size), Image.Resampling.BILINEAR)
            masks[index] = Image.fromarray(mask).resize((size, size), Image.Resampling.NEAREST)
            return index

        for count, _ in enumerate(executor.map(resize_example, range(len(labels))), 1):
            if count % 500 == 0:
                print(f"Verified and resized {count}/{len(labels)} labeled images.", flush=True)
    images.flush()
    masks.flush()
    summary = {
        "train_images": len(labels),
        "public_images": len(public),
        "mask_order": "column-major, 1-based",
        "preview_ids": [records[i].ID for i in selected],
        "empty_labels": int(np.count_nonzero(ratios == 0)),
        "fully_labeled": int(np.count_nonzero(ratios == 1)),
        "resized_empty_from_nonempty": int(np.count_nonzero((ratios > 0) & ~masks.any(axis=(1, 2)))),
        "image_size": size,
    }
    labels[["ID", "height", "width"]].to_csv(output / "images.csv", index=False)
    (output / "annotation_check.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    prepare_data(here.parent / "AIGC Detection Dataset 2026.zip", here / "cache256")
