"""
Generate prompts.csv for train and test splits using prompt_templates.py.

Usage:
    python data_prep/generate_prompts.py          # regenerate both splits
    python data_prep/generate_prompts.py --split train
    python data_prep/generate_prompts.py --split test

    # Prompts that match what training actually sees (512 centre crop, no tiny objects),
    # written to a separate file so prompts.csv stays untouched:
    python data_prep/generate_prompts.py --crop 512 --min-area 0.002 --out-name prompts_crop.csv
    # Same, but without the random lighting/weather/camera words:
    python data_prep/generate_prompts.py --crop 512 --min-area 0.002 --variant objects --out-name prompts_objects.csv

    # Dataset 2 (SANPO): prompts from the session's real weather/environment labels.
    # The label space (SANPO ids) is read from data_sanpo/LABEL_SPACE automatically.
    python data_prep/generate_prompts.py --data-root data_sanpo --crop 512 --min-area 0.002 --variant meta --out-name prompts_meta.csv

Select the file for training/eval with DDPM_PROMPTS_FILE_NAME=<name>.

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

from prompt_templates import generate_prompt_from_mask, get_present_objects, format_object_list

DATA_ROOT = _HERE.parent / "data"
TO_V1 = None  # set when labels are not in the project's 35-class ids


def crop_like_training(mask: Image.Image, resolution: int) -> Image.Image:
    """Resize the short side to `resolution` and centre-crop — same geometry as dataset.py."""
    w, h = mask.size
    if w <= h:
        new_w, new_h = resolution, int(resolution * h / w)
    else:
        new_w, new_h = int(resolution * w / h), resolution
    mask = mask.resize((new_w, new_h), Image.NEAREST)
    left = int(round((new_w - resolution) / 2.0))
    top = int(round((new_h - resolution) / 2.0))
    return mask.crop((left, top, left + resolution, top + resolution))


def objects_only_prompt(mask_np: np.ndarray, min_area_frac: float) -> str:
    """Prompt with no random style words: a fixed scene phrase plus the objects in the mask."""
    objects = get_present_objects(mask_np, min_area_frac=min_area_frac)
    base = "street-level photo of a city sidewalk"
    return f"{base}, {format_object_list(objects)} visible" if objects else base


WEATHER_WORDS = {"sunny": "sunny day", "cloudy": "overcast sky", "snowy": "snow on the ground",
                 "rainy": "rainy weather", "foggy": "fog"}
ENV_WORDS = {"urban": "urban street", "suburban": "suburban street", "park": "park path",
             "road_junction": "road junction", "open_space": "open plaza", "open_terrain": "open terrain",
             "rural": "rural road", "indoor": "indoor walkway"}


def meta_prompt(mask_np: np.ndarray, min_area_frac: float, meta: dict) -> str:
    """Prompt grounded in the recorded session attributes (weather, environment) plus objects in the mask."""
    env = [ENV_WORDS[e] for e in meta.get("environment", "").split("+") if e in ENV_WORDS][:2]
    weather = [WEATHER_WORDS[w] for w in meta.get("weather", "").split("+") if w in WEATHER_WORDS]
    parts = ["street-level photo of a sidewalk"] + env + weather
    objects = get_present_objects(mask_np, min_area_frac=min_area_frac)
    if objects:
        parts.append(format_object_list(objects) + " visible")
    return ", ".join(parts)


def generate_split(split: str, seed_offset: int = 0, crop: int = 0, min_area: float = 0.0,
                   variant: str = "full", out_name: str = "prompts.csv") -> None:
    """
    Generate a prompt CSV for one split (train or test).

    Args:
        split: "train" or "test"
        seed_offset: added to per-image seed so train/test prompts differ
        crop: if > 0, derive objects from the training crop at this resolution instead of the full frame
        min_area: minimum mask fraction for a class to be named in the prompt
        variant: "full" (scene + lighting + weather + sensor + objects) or "objects" (objects only)
        out_name: output file name inside <split>/prompts/
    """
    labels_dir = DATA_ROOT / split / "labels"
    prompts_dir = DATA_ROOT / split / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    out_csv = prompts_dir / out_name

    # Collect mask files, sort for determinism
    mask_files = sorted(labels_dir.glob("*.png"))
    meta = {}
    if variant == "meta":
        import csv as _csv
        meta = {r["name"]: r for r in _csv.DictReader(open(DATA_ROOT / "metadata.csv"))}
    if not mask_files:
        print(f"[WARN] No .png files found in {labels_dir} — skipping {split}")
        return

    print(f"Generating {len(mask_files)} prompts for [{split}] ...")

    rows = []
    for i, mask_path in enumerate(mask_files):
        mask = Image.open(mask_path)
        if crop:
            mask = crop_like_training(mask, crop)
        mask_np = np.array(mask)
        if TO_V1 is not None:
            mask_np = TO_V1[mask_np]
        if variant == "objects":
            prompt = objects_only_prompt(mask_np, min_area)
        elif variant == "meta":
            prompt = meta_prompt(mask_np, min_area, meta[mask_path.name])
        else:
            # Use per-image seed so prompts are reproducible but varied
            prompt = generate_prompt_from_mask(mask_np, seed=seed_offset + i, min_area_frac=min_area)
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
    parser.add_argument("--crop", type=int, default=0,
                        help="Derive objects from the training centre crop at this resolution (0 = full frame)")
    parser.add_argument("--min-area", type=float, default=0.0,
                        help="Minimum mask fraction for a class to be named (0 = any pixel)")
    parser.add_argument("--variant", choices=["full", "objects", "meta"], default="full",
                        help="full = random scene/lighting/weather/sensor + objects; objects = objects only; "
                             "meta = recorded weather/environment (needs <data-root>/metadata.csv) + objects")
    parser.add_argument("--label-space", choices=["v1", "sanpo"], default=None,
                        help="ids stored in the label files; default: read <data-root>/LABEL_SPACE (absent = v1). "
                             "Given explicitly, it must agree with that file.")
    parser.add_argument("--data-root", default=None,
                        help="dataset folder with train/ and test/ (default: DDPM/data = dataset 1)")
    parser.add_argument("--out-name", default="prompts.csv",
                        help="Output file name inside <split>/prompts/ (default: prompts.csv)")
    args = parser.parse_args()
    global DATA_ROOT, TO_V1
    if args.data_root:
        DATA_ROOT = Path(args.data_root).resolve()
    marker = DATA_ROOT / "LABEL_SPACE"
    data_space = marker.read_text().strip() if marker.exists() else "v1"
    if args.label_space and args.label_space != data_space:
        raise SystemExit(f"--label-space {args.label_space} disagrees with {marker if marker.exists() else DATA_ROOT} ({data_space})")
    print(f"Label space: {data_space} ({DATA_ROOT})")
    if data_space == "sanpo":
        sys.path.insert(0, str(_HERE / "sanpo"))
        from class_mapping_sanpo import SANPO_TO_V1
        TO_V1 = np.array([SANPO_TO_V1[i] for i in range(31)], dtype=np.uint8)

    splits = ["train", "test"] if args.split == "both" else [args.split]
    # Different seed offsets so train and test get different prompts for the same mask
    offsets = {"train": 0, "test": 100_000}

    for split in splits:
        generate_split(split, seed_offset=offsets[split], crop=args.crop, min_area=args.min_area,
                       variant=args.variant, out_name=args.out_name)

    print("\nDone.")


if __name__ == "__main__":
    main()
