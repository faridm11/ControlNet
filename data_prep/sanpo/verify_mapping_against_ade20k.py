"""
Check dataset 2's conditioning colours against the official ADE20K palette.

For every SANPO class: the ADE20K class chosen in class_mapping_sanpo.py, whether it is an
exact or closest match, the RGB colour fed to the ControlNet, and the share of dataset-2
pixels. Verifies that every non-void colour is an ADE20K colour and that it belongs to
the intended ADE20K class.

The ADE20K table comes from data_prep/color_coding_semantic_segmentation_classes.xlsx
(150 classes, the palette the seg ControlNet was trained on), exported to
data_prep/ade20k_150_palette.csv.

Run from DDPM/:  python data_prep/sanpo/verify_mapping_against_ade20k.py
"""

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from class_mapping_sanpo import ADE_NAMES, CLASS_NAMES, COLOR_PALETTE, SANPO_TO_ADE
from common import OUT_DIR

ade = {}
for r in csv.DictReader(open(HERE.parent / "ade20k_150_palette.csv")):
    ade.setdefault((int(r["r"]), int(r["g"]), int(r["b"])), []).append(int(r["ade_idx"]))

counts = np.zeros(31, dtype=np.int64)
for f in sorted((OUT_DIR / "train" / "labels").glob("*.png"))[::5]:
    counts += np.bincount(np.array(Image.open(f)).ravel(), minlength=31)[:31]
share = counts / counts.sum()

problems = 0
print(f"{'SANPO class':26s} {'kind':8s} {'ADE20K class':16s} {'colour':16s} pixels")
for s in range(31):
    idx, kind = SANPO_TO_ADE[s]
    rgb = COLOR_PALETTE[s]
    if idx is not None and idx not in ade.get(rgb, []):
        problems += 1
        kind += " MISMATCH"
    print(f"{s:2d} {CLASS_NAMES[s]:23s} {kind:8s} {ADE_NAMES[s]:16s} {str(rgb):16s} {100 * share[s]:5.2f}%")

for kind in ("exact", "closest", "void"):
    ids = [s for s in range(31) if SANPO_TO_ADE[s][1] == kind]
    print(f"{kind:8s}: {len(ids):2d} classes, {100 * share[ids].sum():5.1f}% of pixels")
print(f"{problems} mismatches")
sys.exit(1 if problems else 0)
