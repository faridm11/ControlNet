"""
Generate prompts.csv for train and test splits using prompt_templates.py.

Usage:
    python data_prep/generate_prompts.py          # regenerate both splits
    python data_prep/generate_prompts.py --split train
    python data_prep/generate_prompts.py --split test

Output:
    data/train/prompts/prompts.csv
    data/test/prompts/prompts.csv

CSV format (matches dataset.py expectation):
    Image_Name, Mask_Name, Text_Prompt
"""

import csv
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

# Allow running from repo root or from data_prep/
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from prompt_templates import generate_prompt_from_mask

DATA_ROOT = _HERE.parent / "data"


def generate_split(split: str, seed_offset: int = 0) -> None:
    """
    Generate prompts.csv for one split (train or test).

    Args:
        split: "train" or "test"
        seed_offset: added to per-image seed so train/test prompts differ
    """
    labels_dir = DATA_ROOT / split / "labels"
    prompts_dir = DATA_ROOT / split / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    out_csv = prompts_dir / "prompts.csv"

    # Collect mask files, sort for determinism
    mask_files = sorted(labels_dir.glob("*.png"))
    if not mask_files:
        print(f"[WARN] No .png files found in {labels_dir} — skipping {split}")
        return

    print(f"Generating {len(mask_files)} prompts for [{split}] ...")

    rows = []
    for i, mask_path in enumerate(mask_files):
        mask_np = np.array(Image.open(mask_path))
        # Use per-image seed so prompts are reproducible but varied
        prompt = generate_prompt_from_mask(mask_np, seed=seed_offset + i)
        rows.append({
            "Image_Name": mask_path.name,
            "Mask_Name":  mask_path.name,
            "Text_Prompt": prompt,
        })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Image_Name", "Mask_Name", "Text_Prompt"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved {len(rows)} rows → {out_csv}")
    # Show a few examples
    for row in rows[:3]:
        print(f"  [{row['Image_Name']}] {row['Text_Prompt']}")


def main():
    parser = argparse.ArgumentParser(description="Generate prompt CSVs from segmentation masks")
    parser.add_argument(
        "--split",
        choices=["train", "test", "both"],
        default="both",
        help="Which split to generate (default: both)",
    )
    args = parser.parse_args()

    splits = ["train", "test"] if args.split == "both" else [args.split]
    # Different seed offsets so train and test get different prompts for the same mask
    offsets = {"train": 0, "test": 100_000}

    for split in splits:
        generate_split(split, seed_offset=offsets[split])

    print("\nDone.")


if __name__ == "__main__":
    main()
