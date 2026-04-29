"""
Sanity checks for mask format, palette mapping, and RGB conditioning.
Run from repo root:
  python -m src.testing.test_masks --split train --num-samples 5
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from src.data.dataset import ControlNetDataset, PALETTE
from data_prep.class_mapping import CLASS_NAMES, COLOR_PALETTE
from src import config


def _save_mask_rgb(mask_rgb: torch.Tensor, out_path: Path) -> None:
    """Save a (3,H,W) float mask RGB in [0,1] to PNG."""
    out = (mask_rgb.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(out, mode="RGB").save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mask and palette sanity checks")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--resolution", type=int, default=config.RESOLUTION)
    parser.add_argument("--out-dir", type=str, default="outputs/logs/testing_masks")
    args = parser.parse_args()

    if args.split == "train":
        images_dir = config.TRAIN_IMAGES_DIR
        masks_dir = config.TRAIN_MASKS_DIR
        prompts_file = config.TRAIN_PROMPTS_FILE
    else:
        images_dir = config.VAL_IMAGES_DIR
        masks_dir = config.VAL_MASKS_DIR
        prompts_file = config.VAL_PROMPTS_FILE

    dataset = ControlNetDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        prompts_file=prompts_file,
        resolution=args.resolution,
        augment_masks=False,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MASK FORMAT CHECK")
    print("=" * 80)
    print(f"Split: {args.split}")
    print(f"Samples: {min(args.num_samples, len(dataset))}")
    print(f"Resolution: {args.resolution}")
    print(f"Out dir: {out_dir}")
    print("=" * 80)

    all_classes = set()
    for i in range(min(args.num_samples, len(dataset))):
        sample = dataset[i]
        mask = sample["mask"].cpu().numpy()
        mask_rgb = sample["mask_rgb"]
        item = dataset.data[i]

        classes = np.unique(mask).tolist()
        all_classes.update(classes)

        print(f"Sample {i}: {item['image_name']}")
        print(f"  Mask shape: {mask.shape}, dtype: {mask.dtype}")
        print(f"  Classes: {classes}")

        # Save a quick RGB visualization
        _save_mask_rgb(mask_rgb, out_dir / f"mask_rgb_{i}_{item['mask_name']}")

        # Round-trip check: PALETTE[mask] should match mask_rgb
        manual_rgb = PALETTE[torch.from_numpy(mask).long()].permute(2, 0, 1)
        max_diff = (manual_rgb - mask_rgb).abs().max().item()
        print(f"  Palette mapping max diff: {max_diff:.6f}")

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total unique classes seen: {len(all_classes)}")
    for cid in sorted(all_classes):
        name = CLASS_NAMES.get(cid, "UNKNOWN")
        color = COLOR_PALETTE.get(cid, (0, 0, 0))
        print(f"  {cid:2d}: {name:20s} RGB{color}")


if __name__ == "__main__":
    main()
