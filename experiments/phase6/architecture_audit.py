"""Audit network support, checkpoint compatibility, and archived validation counts."""
import hashlib
import json

import pandas as pd
import torch

from deep_unet import BASE, HERE, DeepUNet, SmallUNet


ROOT = HERE.parent
PATHS = {
    "V1": BASE / "model.pth",
    "Weighted": ROOT / "phase2/weighted/model.pth",
    "384": ROOT / "phase3/resolution384/model.pth",
}
TABLES = {
    "V1": ROOT / "phase2/v1/validation_by_image.csv",
    "Weighted": ROOT / "phase2/weighted/validation_by_image.csv",
    "384": ROOT / "phase3/resolution384/validation_by_image.csv",
}


def merge(*bounds):
    return min(start for start, _ in bounds), max(end for _, end in bounds)


def receptive_field(levels, size=256):
    # One axis suffices because all spatial operators act symmetrically.
    def block(axis):
        # Two padded size-three convolutions on the same grid.
        return [merge(*axis[max(0, k - 2): k + 3]) for k in range(len(axis))]

    axis = [(k, k) for k in range(size)]
    skips = []
    for _ in range(levels):
        skip = block(axis)
        skips.append(skip)
        axis = [merge(a, b) for a, b in zip(skip[::2], skip[1::2])]
    axis = block(axis)
    for skip in reversed(skips):
        # align_corners=False, 2x: source coordinate is k/2 - 1/4.
        upsample = [merge(axis[max(0, (k - 1) // 2)], axis[min(len(axis) - 1, (k - 1) // 2 + 1)])
                    for k in range(len(skip))]
        axis = block([merge(a, b) for a, b in zip(skip, upsample)])
    widths = [end - start + 1 for start, end in axis]
    return {"center_index": size // 2, "center_bounds": axis[size // 2],
            "center_width": widths[size // 2], "interior_widths": sorted(set(widths[64:192])),
            "minimum_width_border": min(widths), "maximum_width": max(widths)}


def audit():
    torch.set_num_threads(4)
    torch.manual_seed(5489)
    v1 = SmallUNet()
    rng_v1 = torch.random.get_rng_state()
    torch.manual_seed(5489)
    deep = DeepUNet()
    same_shared_weights = all(torch.equal(tensor, deep.state_dict()[name])
                              for name, tensor in v1.state_dict().items())
    same_shuffle_rng = torch.equal(rng_v1, torch.random.get_rng_state())
    assert same_shared_weights and same_shuffle_rng
    val_ids = json.loads((BASE / "split.json").read_text())["validation"]
    existing, actual, pixels = {}, None, None
    for name, path in PATHS.items():
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        v1.load_state_dict(checkpoint["model_state"], strict=True)
        assert all(torch.isfinite(tensor).all() for tensor in v1.state_dict().values())
        table = pd.read_csv(TABLES[name], dtype={"ID": str})
        assert table.ID.tolist() == val_ids
        target = table.TP + table.FN
        if actual is None:
            actual, pixels = target, table.pixels
        assert target.equals(actual) and table.pixels.equals(pixels)
        tp, fp, fn = table[["TP", "FP", "FN"]].sum()
        existing[name] = {
            "checkpoint": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "strict_state_dict_load": True, "input_size": checkpoint["image_size"],
            "native_micro_iou": tp / (tp + fp + fn), "native_macro_iou": table.iou.mean(),
            "native_dice": 2 * tp / (2 * tp + fp + fn),
        }
    assert [row["input_size"] for row in existing.values()] == [256, 256, 384]
    assert torch.cuda.is_available()
    deep = deep.to("cuda")
    x = torch.zeros(1, 3, 256, 256, device="cuda")
    with torch.inference_mode():
        logits = deep(x)
    assert logits.shape == (1, 1, 256, 256) and torch.isfinite(logits).all()
    result = {
        "parameters": {"V1": sum(p.numel() for p in v1.parameters()),
                       "Deep": sum(p.numel() for p in deep.parameters())},
        "receptive_field_input_pixels_one_axis": {"V1": receptive_field(2), "Deep": receptive_field(3)},
        "matches_v1_shared_initial_weights": same_shared_weights, "matches_v1_shuffle_rng": same_shuffle_rng,
        "gpu_forward": {"device": torch.cuda.get_device_name(), "shape": list(logits.shape), "finite": True},
        "validation": {"same_ids_and_truth": True, "images": len(val_ids), "empty_targets": int(actual.eq(0).sum())},
        "checkpoints": existing,
    }
    (HERE / "architecture_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    audit()
