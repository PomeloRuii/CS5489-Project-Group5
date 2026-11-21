import io
import json
from zipfile import ZipFile

import numpy as np
from PIL import Image
import torch

from prepare import ARCHIVE, BASE, HERE, archive_index
from project import SmallUNet, decode_rle
from sliding import sliding_probability


def check():
    torch.set_num_threads(4)
    device = torch.device("cuda")
    model = SmallUNet().to(device)
    saved = torch.load(BASE / "model.pth", map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state"])
    model.eval()
    ids = json.loads((BASE / "split.json").read_text())["validation"][:3]
    report = []
    with ZipFile(ARCHIVE) as zipped:
        prefix, _ = archive_index(zipped)
        for identifier in ids:
            with Image.open(io.BytesIO(zipped.read(f"{prefix}train/images/{identifier}.webp"))) as image:
                rgb = np.asarray(image.convert("RGB"))
            prob, tiles = sliding_probability(rgb, lambda x: model(x)[:, 0].sigmoid(), device)
            assert prob.shape == rgb.shape[:2] and np.isfinite(prob).all()
            mask = (prob >= .5).astype(np.uint8)
            changes = np.flatnonzero(np.diff(np.pad(mask.ravel(order="F"), 1)))
            rle = " ".join(str(v) for pair in zip(changes[::2] + 1, changes[1::2] - changes[::2]) for v in pair)
            assert np.array_equal(decode_rle(rle, *mask.shape), mask)
            report.append({"ID": identifier, "height": mask.shape[0], "width": mask.shape[1], "tiles": tiles})
    (HERE / "real_inference_preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    check()
