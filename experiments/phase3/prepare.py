import io
import json
import math
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "baseline"
PHASE2 = HERE.parent / "phase2"
ARCHIVE = HERE.parent / "AIGC Detection Dataset 2026.zip"
CACHE = HERE / "cache384"
sys.path.insert(0, str(BASE))
from prepare_data import prepare_data
from project import decode_rle


def archive_index(zipped):
    manifest, = [name for name in zipped.namelist() if name.endswith("/train.csv")]
    table = pd.read_csv(zipped.open(manifest), dtype={"ID": str, "annotation": str}, keep_default_na=False)
    return manifest.removesuffix("train.csv"), table.set_index("ID", verify_integrity=True)


def read_pair(zipped, root, identifier, row):
    with Image.open(io.BytesIO(zipped.read(f"{root}train/images/{identifier}.webp"))) as image:
        if image.size != (row.width, row.height):
            raise ValueError(f"Image and CSV dimensions differ for {identifier}")
        rgb = image.convert("RGB")
    return rgb, decode_rle(row.annotation, row.height, row.width)


def cache_areas(folder, size, ids):
    index = pd.read_csv(folder / "images.csv", dtype={"ID": str})
    assert index.ID.is_unique and set(index.ID) == ids
    masks = np.load(folder / "masks.npy", mmap_mode="r")
    assert masks.shape == (len(index), size, size) and masks.dtype == np.uint8
    # uint8 max <= 1 proves the cache has no values outside the binary mask domain.
    maxima = masks.max(axis=(1, 2))
    assert maxima.max() <= 1, f"Non-binary labels in {folder}"
    return pd.Series(masks.sum(axis=(1, 2), dtype=np.int64), index=index.ID)


def summarize(pixels, positives):
    fractions = positives / pixels
    q = np.quantile(fractions, [0, .05, .25, .5, .75, .95, 1])
    return {
        "images": len(fractions), "pixels": int(pixels.sum()), "foreground": int(positives.sum()),
        "background": int(pixels.sum() - positives.sum()),
        "micro": float(positives.sum() / pixels.sum()), "mean_image": float(fractions.mean()),
        **dict(zip(("min", "p05", "p25", "p50", "p75", "p95", "max"), q.tolist())),
        "empty": int(np.count_nonzero(positives == 0)),
        "below_1_percent": int(np.count_nonzero(fractions < .01)),
    }


def prepare():
    split = json.loads((BASE / "split.json").read_text())
    training, validation = split["train"], split["validation"]
    ids = set(training) | set(validation)
    assert len(training) == 4500 and len(validation) == 500 and len(ids) == 5000
    with ZipFile(ARCHIVE) as zipped:
        prefix, table = archive_index(zipped)
        assert set(table.index) == ids
    native_area = table.height * table.width
    assert (native_area > 0).all()
    counts = []
    for row in table.itertuples():
        runs = np.asarray(row.annotation.split(), dtype=np.int64)
        assert runs.size % 2 == 0, row.Index
        starts, lengths = runs[::2] - 1, runs[1::2]
        ends = starts + lengths
        assert (starts >= 0).all() and (lengths > 0).all() and (ends <= row.height * row.width).all(), row.Index
        assert (starts[1:] >= ends[:-1]).all(), row.Index
        counts.append(int(lengths.sum()))
    native = pd.Series(counts, index=table.index)
    old = BASE / "cache256"
    low = cache_areas(old, 256, ids)

    v1_train_fg = int(low.loc[training].sum())
    v1_train_bg = len(training) * 256**2 - v1_train_fg
    ratio = v1_train_bg / v1_train_fg
    stored = json.loads((PHASE2 / "weighted" / "config.json").read_text())["bce_pos_weight"]
    assert math.isclose(math.sqrt(ratio), stored, rel_tol=1e-12)
    print(f"Phase 2 verified: train FG {v1_train_fg:,}; BG {v1_train_bg:,}; "
          f"BG/FG {ratio:.9f}; sqrt {stored:.9f}.", flush=True)
    for name, subset in split.items():
        print(name, "native", summarize(native_area.loc[subset], native.loc[subset]),
              "256", summarize(pd.Series(256**2, index=subset), low.loc[subset]), flush=True)

    if not (CACHE / "annotation_check.json").exists():
        prepare_data(ARCHIVE, CACHE, size=384)
    high = cache_areas(CACHE, 384, ids)
    # Compare both caches against direct preprocessing on fixed IDs from both subsets.
    samples = training[:8] + validation[:8]
    with ZipFile(ARCHIVE) as zipped:
        for size, folder in ((256, old), (384, CACHE)):
            index = pd.read_csv(folder / "images.csv", dtype={"ID": str})
            image_map = np.load(folder / "images.npy", mmap_mode="r")
            mask_map = np.load(folder / "masks.npy", mmap_mode="r")
            lookup = {identifier: i for i, identifier in enumerate(index.ID)}
            for identifier in samples:
                image, mask = read_pair(zipped, prefix, identifier, table.loc[identifier])
                rgb = np.asarray(image.resize((size, size), Image.Resampling.BILINEAR))
                resized = np.asarray(Image.fromarray(mask).resize((size, size), Image.Resampling.NEAREST))
                i = lookup[identifier]
                assert np.array_equal(image_map[i], rgb) and np.array_equal(mask_map[i], resized), identifier

    records = []
    details = pd.DataFrame({
        "native_pixels": native_area, "native_foreground": native,
        "256_foreground": low, "384_foreground": high,
    })
    details.index.name = "ID"
    details["split"] = np.where(details.index.isin(validation), "validation", "train")
    details.to_csv(HERE / "foreground_by_image.csv")
    for name, subset in split.items():
        for size, pixels, positives in (
            ("native", native_area.loc[subset], native.loc[subset]),
            ("256", pd.Series(256**2, index=subset), low.loc[subset]),
            ("384", pd.Series(384**2, index=subset), high.loc[subset]),
        ):
            records.append({"split": name, "resolution": size, **summarize(pixels, positives)})
    pd.DataFrame(records).to_csv(HERE / "foreground_summary.csv", index=False)
    audit = dict(
        verified_train_positive_pixels=v1_train_fg, verified_train_negative_pixels=v1_train_bg,
        negative_positive_ratio=ratio, expected_pos_weight=math.sqrt(ratio), saved_pos_weight=stored,
        binary_labels_verified=5000, matching_preprocess_pairs_per_resolution=len(samples),
        preprocess_check_ids=samples,
    )
    (HERE / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("PRECHECK PASSED", flush=True)


if __name__ == "__main__":
    prepare()
