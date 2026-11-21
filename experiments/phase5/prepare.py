import io
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
from PIL import Image
import torch

from sliding import SIZE, pad_image, sliding_probability

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT / "baseline"
sys.path.insert(0, str(BASE))
from project import decode_rle

ARCHIVE = ROOT / "AIGC Detection Dataset 2026.zip"
CACHE = HERE / "patch_cache"
SEED = 5489
EPOCHS = 10


def archive_index(zipped):
    manifest, = [name for name in zipped.namelist() if name.endswith("/train.csv")]
    table = pd.read_csv(zipped.open(manifest), dtype={"ID": str, "annotation": str}, keep_default_na=False)
    return manifest.removesuffix("train.csv"), table.set_index("ID", verify_integrity=True)


def read_pair(zipped, prefix, identifier, row):
    with Image.open(io.BytesIO(zipped.read(f"{prefix}train/images/{identifier}.webp"))) as image:
        rgb = image.convert("RGB")
    if rgb.size != (row.width, row.height):
        raise ValueError(f"Unexpected image size for {identifier}")
    return rgb, decode_rle(row.annotation, rgb.height, rgb.width)


def uniform(rng, h, w):
    return int(rng.integers(h - SIZE + 1)), int(rng.integers(w - SIZE + 1))


def crop(arr, coord):
    y, x = coord
    return arr[y:y + SIZE, x:x + SIZE]


def texture(crop_rgb):
    """Mean absolute horizontal and vertical green-channel change, stride two."""
    g = crop_rgb[::2, ::2, 1].astype(np.int16)
    return float((np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean()) / 2)


def foreground_sites(mask):
    coarse = mask[::8, ::8].astype(bool)
    edge = np.zeros_like(coarse)
    edge[1:] |= coarse[1:] != coarse[:-1]
    edge[:, 1:] |= coarse[:, 1:] != coarse[:, :-1]
    points = np.argwhere(coarse & edge) * 8
    if not len(points):
        points = np.argwhere(coarse) * 8
    if not len(points) and mask.any():
        points = np.array([np.unravel_index(mask.argmax(), mask.shape)])
    return points


def tamper(rng, mask, points, fallback):
    if not len(points):
        return fallback, "uniform_fallback"
    h, w = mask.shape
    first_positive = None
    for _ in range(12):
        py, px = points[rng.integers(len(points))] + rng.integers(-64, 65, size=2)
        coord = (int(np.clip(py - SIZE // 2, 0, h - SIZE)),
                 int(np.clip(px - SIZE // 2, 0, w - SIZE)))
        ratio = crop(mask, coord).mean()
        if .01 <= ratio <= .95:
            return coord, "tamper_boundary"
        if ratio > 0 and first_positive is None:
            first_positive = coord
    if first_positive is not None:
        return first_positive, "tamper_any_positive"
    return fallback, "uniform_fallback"


def background(rng, rgb, mask, fallback):
    h, w = mask.shape
    best, best_score = None, -1
    for _ in range(12):
        coord = uniform(rng, h, w)
        if not crop(mask, coord).any():
            strength = texture(crop(rgb, coord))
            if strength > best_score:
                best, best_score = coord, strength
    return (fallback, "uniform_fallback") if best is None else (best, "background_texture")


def preflight():
    device = torch.device("cuda")
    assert torch.cuda.is_available()
    rows = []
    rng = np.random.default_rng(SEED)
    for h, w in ((256, 256), (319, 517), (111, 768), (17, 13), (513, 257)):
        rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        identity, tiles = sliding_probability(rgb, lambda x: x[:, 0], device)
        constant, _ = sliding_probability(rgb, lambda x: torch.full_like(x[:, 0], .375), device)
        err = float(np.abs(identity - rgb[:, :, 0] / 255).max())
        assert identity.shape == (h, w) and err < 3e-7 and np.allclose(constant, .375, atol=3e-7)
        # Identical coordinate mapping through padding and cropping for both modalities.
        mask = (rgb[:, :, 0] > 127).astype(np.uint8)
        padded, padded_mask, (y, x) = pad_image(rgb, mask)
        assert np.array_equal(padded[y:y + h, x:x + w], rgb)
        assert np.array_equal(padded_mask[y:y + h, x:x + w], mask)
        rows.append({"height": h, "width": w, "tiles": tiles, "max_coordinate_error": err})
    (HERE / "sliding_preflight.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("Sliding-window coverage, constant fusion and coordinate round-trip passed.", flush=True)


def prepare():
    preflight()
    start = time.perf_counter()
    CACHE.mkdir(exist_ok=True)
    split = json.loads((BASE / "split.json").read_text())
    train, val = split["train"], split["validation"]
    assert len(train) == 4500 and len(val) == 500 and not set(train) & set(val)
    schedule_rng = np.random.default_rng(SEED)
    categories = np.stack([
        schedule_rng.permutation(np.repeat(["uniform", "tamper", "background"], [2250, 1350, 900]))
        for _ in range(EPOCHS)
    ])
    a = np.lib.format.open_memmap(CACHE / "A.npy", mode="w+", dtype="uint8", shape=(len(train), EPOCHS, SIZE, SIZE, 4))
    b = np.lib.format.open_memmap(CACHE / "B.npy", mode="w+", dtype="uint8", shape=a.shape)
    v = np.lib.format.open_memmap(CACHE / "val.npy", mode="w+", dtype="uint8", shape=(len(val), 5, SIZE, SIZE, 4))
    with ZipFile(ARCHIVE) as zipped, ThreadPoolExecutor(max_workers=6) as pool:
        prefix, labels = archive_index(zipped)
        train_labels = labels.loc[train]
        shape = train_labels[["height", "width"]].to_numpy()
        native = pd.read_csv(ROOT / "phase3/foreground_by_image.csv", dtype={"ID": str}).set_index("ID").loc[train]
        dims = {"shorter_side": shape.min(1), "longer_side": shape.max(1),
                "equivalent_edit_square_side": np.sqrt(native.native_foreground.to_numpy()),
                "foreground_fraction": native.native_foreground.to_numpy() / native.native_pixels.to_numpy()}
        summary = {name: dict(zip(("min", "p05", "median", "p95", "max"), np.quantile(values, [0,.05,.5,.95,1]).tolist()))
                   for name, values in dims.items()}
        summary["training_images_with_padding"] = int(np.count_nonzero(shape.min(1) < SIZE))
        summary["training_edits_below_1_percent"] = int(np.count_nonzero(dims["foreground_fraction"] < .01))
        print("Training data distribution", summary, flush=True)

        def load(identifier):
            rgb, target = read_pair(zipped, prefix, identifier, labels.loc[identifier])
            return pad_image(np.asarray(rgb), target)[:2]

        def write_training(i):
            identifier = train[i]
            rgb, mask = load(identifier)
            rng_a = np.random.default_rng([SEED, i, 0])
            rng_b = np.random.default_rng([SEED, i, 1])
            coords = [uniform(rng_a, *mask.shape) for _ in range(EPOCHS)]
            points = foreground_sites(mask) if "tamper" in categories[:, i] else []
            record = []
            for epoch, coord in enumerate(coords):
                category = categories[epoch, i]
                final, actual = coord, "uniform"
                if category == "tamper":
                    final, actual = tamper(rng_b, mask, points, coord)
                if category == "background":
                    final, actual = background(rng_b, rgb, mask, coord)
                for group, store, selected, requested, realized in (
                    ("A", a, coord, "uniform", "uniform"), ("B", b, final, category, actual)
                ):
                    x, m = crop(rgb, selected), crop(mask, selected)
                    store[i, epoch, :, :, :3] = x
                    store[i, epoch, :, :, 3] = m
                    fraction = float(m.mean())
                    if realized == "background_texture":
                        assert fraction == 0
                    record.append((identifier, epoch + 1, group, requested, realized, *selected, fraction, texture(x)))
            return record

        def write_validation(i):
            rgb, mask = load(val[i])
            h, w = mask.shape
            for j, (fy, fx) in enumerate(((.25, .25), (.25, .75), (.75, .25), (.75, .75), (.5, .5))):
                coord = (round((h - SIZE) * fy), round((w - SIZE) * fx))
                v[i, j, :, :, :3] = crop(rgb, coord)
                v[i, j, :, :, 3] = crop(mask, coord)

        rows = []
        for count, record in enumerate(pool.map(write_training, range(len(train))), 1):
            rows.extend(record)
            if count % 500 == 0:
                print(f"Cached paired crops for {count}/{len(train)} training images.", flush=True)
        for _ in pool.map(write_validation, range(len(val))):
            pass
    for array in (a, b, v):
        array.flush()
    df = pd.DataFrame(rows, columns=["ID","epoch","group","requested","realized","y","x","foreground_fraction","texture"])
    assert len(df) == 90000
    assert df.groupby(["group", "epoch"]).size().eq(4500).all()
    assert df[df.group.eq("B")].groupby(["epoch", "requested"]).size().unstack().eq(
        pd.Series(dict(background=900, tamper=1350, uniform=2250))).all().all()
    df.to_csv(HERE / "sampling_by_crop.csv", index=False)
    df.assign(nonempty=df.foreground_fraction.gt(0)).groupby(["group","requested","realized"]).agg(
        patches=("ID","size"), nonempty=("nonempty","sum"), mean_foreground=("foreground_fraction","mean"),
        mean_texture=("texture","mean")).to_csv(HERE / "sampling_summary.csv")
    fig, axes = plt.subplots(4, 4, figsize=(10, 10), layout="constrained")
    for panels, request in zip(axes, ("uniform", "tamper", "background", "background_fallback")):
        wanted = request.removesuffix("_fallback")
        selection = df[df.group.eq("B") & df.epoch.eq(1) & df.requested.eq(wanted)]
        if request.endswith("fallback"):
            selection = selection[selection.realized.eq("uniform_fallback")]
        else:
            selection = selection[~selection.realized.eq("uniform_fallback")]
        if selection.empty:
            for panel in panels:
                panel.axis("off")
            continue
        sample = selection.iloc[len(selection) // 2]
        i = train.index(sample.ID)
        for panel, patch, title in zip(panels, (a[i,0], a[i,0], b[i,0], b[i,0]),
                                       (f"{sample.ID} A", "A mask", f"B {sample.realized}", "B mask")):
            if title.endswith("mask"):
                panel.imshow(patch[:,:,3], cmap="gray", vmin=0, vmax=1)
            else:
                panel.imshow(patch[:,:,:3])
            panel.set_title(title)
            panel.axis("off")
    fig.savefig(HERE / "sampling_examples.png", dpi=135)
    plt.close(fig)
    summary["preprocessing_seconds"] = time.perf_counter() - start
    (HERE / "data_preflight.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Paired caches and sampling statistics ready.", flush=True)


if __name__ == "__main__":
    prepare()
