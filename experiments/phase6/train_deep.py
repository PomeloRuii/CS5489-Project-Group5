"""Run the single three-downsample full-image comparison."""
import argparse
import sys
from pathlib import Path

from deep_unet import BASE, HERE, DeepUNet

sys.path.append(str(HERE / "submission_candidate"))
from train_candidate import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "deep256")
    args = parser.parse_args()
    if not (HERE / "architecture_audit.json").is_file():
        raise FileNotFoundError("Run architecture_audit.py first")
    train(BASE / "cache256", BASE / "split.json", args.out, model_factory=DeepUNet,
          weighted=False, model_details={
              "change": "one additional downsample, 128-channel bridge, upsample and skip",
              "initialization": "PyTorch defaults, V1 common tensors and post-init shuffle RNG matched",
              "encoder_channels": [16, 32, 64], "decoder_channels": [64, 32, 16],
          })
