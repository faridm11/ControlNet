"""
Step 1 — CPU checks of data, palette, crop alignment and prompt length.

Run from DDPM/ (HF_HOME must point at the model cache):
    python controlnet_testing/step1_cpu_checks/check_data_and_prompts.py

Writes overlay images (photo blended with its conditioning mask, after the
training crop) to controlnet_testing/step1_cpu_checks/overlays/ for a visual check.
"""

import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data_prep"))
os.chdir(ROOT)

import numpy as np
from PIL import Image
from transformers import CLIPTokenizer

from src import config
from src.data.dataset import ControlNetDataset, PALETTE, _REMAP
import class_mapping_v2 as v2

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


# 1. v2 palette: every non-void class has its own colour; remap covers all 35 V1 ids
colours = [v2.COLOR_PALETTE[i] for i in range(v2.NUM_CLASSES)]
check("v2 palette colours are unique", len(set(colours)) == v2.NUM_CLASSES)
check("v2 remap covers V1 ids 0-34 and lands in 0-20",
      sorted(v2.REMAP_V1_TO_V2) == list(range(35)) and set(v2.REMAP_V1_TO_V2.values()) == set(range(21)))

# 2. dataset output ranges + every conditioning pixel is an exact palette colour
ds = ControlNetDataset(config.VAL_IMAGES_DIR, config.VAL_MASKS_DIR, config.VAL_PROMPTS_FILE,
                       resolution=512, augment_masks=False)
out_dir = Path(__file__).parent / "overlays"
out_dir.mkdir(exist_ok=True)
ok_range, ok_palette = True, True
pal255 = {tuple(int(round(c * 255)) for c in row) for row in PALETTE.tolist()}
for idx in range(0, len(ds), 5):
    s = ds[idx]
    img, cond = s["image"], s["mask_rgb"]
    ok_range &= img.shape == (3, 512, 512) and -1 <= img.min() and img.max() <= 1 and 0 <= cond.min() and cond.max() <= 1
    cond255 = (cond.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    ok_palette &= {tuple(c) for c in np.unique(cond255.reshape(-1, 3), axis=0).tolist()} <= pal255
    if idx < 30:
        photo = ((img.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
        blend = (0.5 * photo + 0.5 * cond255).astype(np.uint8)
        Image.fromarray(np.concatenate([photo, cond255, blend], axis=1)).save(out_dir / f"overlay_{s['image_name']}")
check("dataset tensors: shape 3x512x512, image in [-1,1], conditioning in [0,1]", bool(ok_range))
check("conditioning image contains only exact palette colours", ok_palette)
print(f"       overlays for visual alignment check: {out_dir}")

# 3. prompt length vs CLIP's 77-token window, for every prompt file
tok = CLIPTokenizer.from_pretrained(config.SD_MODEL_ID, subfolder="tokenizer")
for name in ("prompts.csv", "prompts_crop.csv", "prompts_objects.csv"):
    lens = []
    for split in ("train", "test"):
        f = config.DATA_DIR / split / "prompts" / name
        if f.exists():
            lens += [len(tok(r["Text_Prompt"]).input_ids) for r in csv.DictReader(open(f))]
    if lens:
        over = sum(l > 77 for l in lens)
        check(f"{name}: no prompt exceeds 77 tokens", over == 0,
              f"mean {np.mean(lens):.0f}, max {max(lens)} tokens, {over} truncated of {len(lens)}")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
