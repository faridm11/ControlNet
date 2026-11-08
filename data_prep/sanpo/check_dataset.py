"""
SANPO step 5 — CPU checks of the converted dataset 2 (run from DDPM/).

    DDPM_DATA_ROOT=$PWD/data_sanpo HF_HOME=... python data_prep/sanpo/check_dataset.py

Checks file counts, sizes, label range (SANPO ids 0-30), the LABEL_SPACE marker and that
the project's dataset class loads it; prints class frequencies next to dataset 1,
and writes a few photo/mask overlays to data_sanpo/check_overlays/.
"""

import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data_prep"))
assert os.environ.get("DDPM_DATA_ROOT", "").endswith("data_sanpo"), "set DDPM_DATA_ROOT to .../data_sanpo"

import numpy as np
from PIL import Image

sys.path.insert(0, str(ROOT / "data_prep" / "sanpo"))
from class_mapping import CLASS_NAMES
from class_mapping_sanpo import CLASS_NAMES as SANPO_NAMES, SANPO_TO_V1
from src import config
from src.data.dataset import ControlNetDataset
from src.segmentor import V1_TO_SEG, SEG_CLASS_NAMES, SEG_IGNORE

D2 = Path(config.DATA_DIR)
D1 = ROOT / "data"
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


TO_V1 = np.array([SANPO_TO_V1[i] for i in range(31)])


def freq(root, split, n=None, sanpo=False):
    """Pixel share per project (V1) class; SANPO ids are converted to V1 for the comparison."""
    files = sorted((root / split / "labels").glob("*.png"))
    if n:
        files = files[:: max(1, len(files) // n)]
    c = np.zeros(35, dtype=np.int64)
    for f in files:
        m = np.array(Image.open(f))
        c += np.bincount((TO_V1[m] if sanpo else m).ravel(), minlength=256)[:35]
    return c / c.sum()


meta = list(csv.DictReader(open(D2 / "metadata.csv")))
for split in ("train", "test"):
    expected = {r["name"] for r in meta if r["split"] == split}
    imgs = {p.name for p in (D2 / split / "images").glob("*.png")}
    lbls = {p.name for p in (D2 / split / "labels").glob("*.png")}
    check(f"{split}: every selected frame has image and label", expected == imgs == lbls,
          f"{len(imgs)} images, {len(lbls)} labels, {len(expected)} selected")
    sample = sorted(lbls)[:: max(1, len(lbls) // 20)]
    ok = all(Image.open(D2 / split / "images" / n).size == (1920, 1080) == Image.open(D2 / split / "labels" / n).size
             and np.array(Image.open(D2 / split / "labels" / n)).max() <= 30 for n in sample)
    check(f"{split}: 1920x1080 and SANPO label IDs 0-30 (sample of {len(sample)})", ok)
    prompts = D2 / split / "prompts"
    for f in sorted(prompts.glob("*.csv")):
        rows = list(csv.DictReader(open(f)))
        check(f"{split}: {f.name} covers every frame", {r["Image_Name"] for r in rows} == imgs, f"{len(rows)} rows")

check("LABEL_SPACE marker selects the sanpo class version", config.CLASS_VERSION == "sanpo" and config.NUM_CLASSES == 31)
ds = ControlNetDataset(config.VAL_IMAGES_DIR, config.VAL_MASKS_DIR, config.VAL_PROMPTS_FILE, resolution=512, augment_masks=False)
s = ds[0]
check("project dataset class loads dataset 2", s["image"].shape == (3, 512, 512) and s["mask_rgb"].shape == (3, 512, 512),
      f"{len(ds)} test samples")

f2, f1 = freq(D2, "train", 300, sanpo=True), freq(D1, "train", 300)
print("\nPixel share per project class (dataset 1 vs dataset 2, train; SANPO converted with SANPO_TO_V1):")
for c in np.argsort(-f2):
    if f2[c] > 0.001 or f1[c] > 0.01:
        print(f"  {c:2d} {CLASS_NAMES[c]:18s} d1 {100 * f1[c]:5.1f}%   d2 {100 * f2[c]:5.1f}%")
seg = np.zeros(11)
for c in range(35):
    k = V1_TO_SEG[c]
    seg[10 if k == SEG_IGNORE else k] += f2[c]
print("Segmentor target classes in dataset 2:", {(SEG_CLASS_NAMES + ["ignored"])[k]: f"{100 * v:.1f}%" for k, v in enumerate(seg)})

out = D2 / "check_overlays"
out.mkdir(exist_ok=True)
for i in range(0, len(ds), max(1, len(ds) // 4))[:4]:
    s = ds[i]
    photo = ((s["image"].permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
    cond = (s["mask_rgb"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(np.concatenate([photo, cond, (0.5 * photo + 0.5 * cond).astype(np.uint8)], 1)).save(out / s["image_name"])
print(f"\nOverlays: {out}")
print(f"{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
