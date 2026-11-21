import json
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from srm_model import BASE, HERE, FixedSRM, SRMUNet
from project import CachedMaskDataset, SmallUNet, image_tensor, run_epoch, segmentation_loss

ARCHIVE = HERE.parent / "AIGC Detection Dataset 2026.zip"
CACHE = BASE / "cache256"
SEED = 5489


def loaders():
    split = json.loads((BASE / "split.json").read_text())
    index = {identifier: i for i, identifier in enumerate(pd.read_csv(CACHE / "images.csv", dtype={"ID": str}).ID)}
    datasets = {
        key: CachedMaskDataset(CACHE, [index[i] for i in ids])
        for key, ids in split.items()
    }
    return split, datasets


@torch.no_grad()
def residual_preview(device, train_ids):
    srm = FixedSRM().to(device)
    assert srm(torch.full((1, 3, 8, 9), 0.5, device=device)).abs().max().item() < 1e-5
    area = pd.read_csv(HERE.parent / "phase3" / "foreground_by_image.csv", dtype={"ID": str}).set_index("ID")
    fractions = (area.native_foreground / area.native_pixels).loc[train_ids].sort_index()
    rng = np.random.default_rng(SEED)
    ids = [
        key for lo, hi in ((.002, .01), (.01, .05), (.05, .15), (.15, .5))
        for key in rng.choice(fractions[fractions.between(lo, hi, inclusive="left")].index, 3, replace=False)
    ]
    with ZipFile(ARCHIVE) as zipped:
        path, = [name for name in zipped.namelist() if name.endswith("/train.csv")]
        root = path.removesuffix("train.csv")
        labels = pd.read_csv(zipped.open(path), dtype={"ID": str, "annotation": str}, keep_default_na=False).set_index("ID")
        from project import decode_rle
        import io

        fig, axes = plt.subplots(6, 4, figsize=(11, 15), layout="constrained")
        rows = []
        for i, key in enumerate(ids):
            row = labels.loc[key]
            with Image.open(io.BytesIO(zipped.read(f"{root}train/images/{key}.webp"))) as file:
                rgb = file.convert("RGB")
            mask = decode_rle(row.annotation, rgb.height, rgb.width)
            small_mask = np.asarray(Image.fromarray(mask).resize((256, 256), Image.Resampling.NEAREST)).astype(bool)
            small = image_tensor(rgb, 256).to(device).unsqueeze(0)
            native = torch.from_numpy(np.array(rgb)).permute(2, 0, 1).to(device, dtype=torch.float32).div_(255)
            after = srm.raw(small)
            before = F.interpolate(srm.raw(native[None]), (256, 256), mode="bilinear", align_corners=False, antialias=True)
            maps = [abs(srm.normalize(r))[0].mean(0).cpu().numpy() for r in (before, after)]
            result = {"ID": key, "target_fraction_256": float(small_mask.mean())}
            for label, m in zip(("native_then_resize", "resize_then_srm"), maps):
                result[f"{label}_inside_mean"] = float(m[small_mask].mean())
                result[f"{label}_outside_mean"] = float(m[~small_mask].mean())
                result[f"{label}_all_mean"] = float(m.mean())
            rows.append(result)
            if i < 6:
                axes[i, 0].imshow(small[0].permute(1, 2, 0).cpu())
                axes[i, 1].imshow(small_mask, cmap="gray")
                for j, m in enumerate(maps, 2):
                    axes[i, j].imshow(m, cmap="magma", vmin=0, vmax=1)
                    axes[i, j].contour(small_mask, levels=[.5], colors=["cyan"], linewidths=.6)
                for ax, name in zip(axes[i], (key, "Reference", "Native SRM → resize", "Resize → SRM")):
                    ax.set_title(name)
                    ax.axis("off")
        fig.savefig(HERE / "training_residual_examples.png", dpi=135)
        plt.close(fig)
    return pd.DataFrame(rows)


def check():
    torch.set_num_threads(4)
    device = torch.device("cuda")
    split, ds = loaders()
    torch.manual_seed(SEED)
    initial_v1 = SmallUNet()
    v1_state = initial_v1.state_dict()
    v1_rng_after_init = torch.random.get_rng_state()
    torch.manual_seed(SEED)
    initial_srm = SRMUNet()
    assert torch.equal(torch.random.get_rng_state(), v1_rng_after_init)
    for name, p in initial_srm.state_dict().items():
        assert torch.equal(p[:, :3] if name == "encoder1.0.weight" else p, v1_state[name]), name
    parameter_counts = [sum(p.numel() for p in net.parameters()) for net in (initial_v1, initial_srm)]
    print("Parameter counts", parameter_counts, flush=True)
    fresh = SmallUNet().to(device)
    fresh.load_state_dict(v1_state)
    checkpoint = torch.load(BASE / "model.pth", map_location=device, weights_only=True)
    trained = SmallUNet().to(device)
    trained.load_state_dict(checkpoint["model_state"])
    v1 = {key: dict(zip(("loss", "iou"), run_epoch(trained, DataLoader(data, batch_size=8), device)))
          for key, data in ds.items()}
    print("V1 checkpoint train/validation", v1, flush=True)

    training = ds["train"]
    subset = [j for j, i in enumerate(training.indices) if .03 <= training.masks[i].mean() <= .35][:8]
    assert len(subset) == 8
    x, y = (torch.stack(t).to(device) for t in zip(*(training[i] for i in subset)))
    optimizer = torch.optim.Adam(fresh.parameters(), lr=.001)
    log = []
    for step in range(301):
        logits = fresh(x)
        loss = segmentation_loss(logits, y)
        if step % 25 == 0:
            pred, actual = logits.detach() >= 0, y.bool()
            iou = (pred & actual).sum().item() / (pred | actual).sum().item()
            log.append({"steps": step, "loss": loss.item(), "iou": iou})
        if step != 300:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    pd.DataFrame(log).to_csv(HERE / "overfit_trace.csv", index=False)
    overfit = {"IDs": [split["train"][i] for i in subset], "initial": log[0], "final": log[-1]}
    print("Fixed subset", overfit, flush=True)
    if log[-1]["iou"] < .5 or log[-1]["iou"] - log[0]["iou"] < .3:
        raise RuntimeError("The V1 small-sample learning check did not pass")
    residual = residual_preview(device, split["train"])
    residual.to_csv(HERE / "training_residual_stats.csv", index=False)
    stats = {col: float(residual[col].median()) for col in residual.columns if col != "ID"}
    result = {"v1": v1, "overfit": overfit, "parameters_v1": parameter_counts[0],
              "parameters_srm": parameter_counts[1], "preview_medians": stats,
              "stable_initial_common_weights": True, "stable_rng_after_init": True}
    (HERE / "preflight.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    check()
