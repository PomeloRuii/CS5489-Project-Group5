"""Build a verified uint8 image/mask cache at a target square size from the course ZIP."""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from models import decode_rle


def _decode_one(args):
    idx, img_id, raw_bytes, rle, size = args
    with Image.open(io.BytesIO(raw_bytes)) as im:
        rgb = im.convert("RGB")
        h, w = rgb.height, rgb.width
        mask = decode_rle(rle, h, w)
        rgb_rs = np.asarray(rgb.resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8)
        mask_rs = np.asarray(
            Image.fromarray(mask).resize((size, size), Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
    return idx, img_id, h, w, rgb_rs, mask_rs


def build_cache(archive: Path, out_dir: Path, size: int = 512, workers: int = 8) -> None:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    if all((out_dir / f).is_file() for f in ("images.csv", "images.npy", "masks.npy")):
        print(f"Cache already exists at {out_dir}, skipping.", flush=True)
        return

    with zipfile.ZipFile(archive) as zf:
        train_df = pd.read_csv(
            io.BytesIO(zf.read("AIGC Detection Dataset 2026/train.csv")),
            dtype={"ID": str, "annotation": str},
            keep_default_na=False,
        )
        train_df = train_df.sort_values("ID").reset_index(drop=True)
        n = len(train_df)
        print(f"Preparing {size}x{size} cache for {n} images from {archive.name}...", flush=True)

        images_mm = np.lib.format.open_memmap(
            out_dir / "images.npy", mode="w+", dtype=np.uint8, shape=(n, size, size, 3)
        )
        masks_mm = np.lib.format.open_memmap(
            out_dir / "masks.npy", mode="w+", dtype=np.uint8, shape=(n, size, size)
        )
        meta_rows = [None] * n

        # Process in chunks to keep RAM bounded
        chunk = 250
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            tasks = []
            for idx in range(start, end):
                row = train_df.iloc[idx]
                img_id = row["ID"]
                raw = zf.read(f"AIGC Detection Dataset 2026/train/images/{img_id}.webp")
                tasks.append((idx, img_id, raw, row["annotation"], size))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for idx, img_id, h, w, rgb_rs, mask_rs in ex.map(_decode_one, tasks):
                    images_mm[idx] = rgb_rs
                    masks_mm[idx] = mask_rs
                    meta_rows[idx] = {"ID": img_id, "height": h, "width": w}
            print(f"Cached {end}/{n} ({time.perf_counter() - t0:.1f}s)", flush=True)

        images_mm.flush()
        masks_mm.flush()

    pd.DataFrame(meta_rows).to_csv(out_dir / "images.csv", index=False)
    fg_ratio = float(np.mean(np.load(out_dir / "masks.npy", mmap_mode="r")))
    summary = {
        "archive": archive.name,
        "size": size,
        "count": n,
        "foreground_ratio": fg_ratio,
        "seconds": round(time.perf_counter() - t0, 2),
    }
    (out_dir / "cache_info.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Done: {summary}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    build_cache(args.archive, args.out, size=args.size, workers=args.workers)
