"""
SANPO step 3b — build the project labels from SANPO's original masks (no download).

Reads RAW_DIR/masks/<name> (SANPO class ids, 2208x1242, from fetch_raw_masks.py),
resizes with nearest neighbour to 1920x1080 and writes data_sanpo/<split>/labels/<name>.
Labels keep SANPO's own 31 ids; the colours come from class_mapping_sanpo.COLOR_PALETTE.
Also writes data_sanpo/LABEL_SPACE = "sanpo", which src/config.py reads so the right
class definitions are used automatically for this data root.

    python data_prep/sanpo/build_labels.py
"""

import csv

import numpy as np
from PIL import Image

from common import HERE, OUT_DIR, RAW_DIR

rows = list(csv.DictReader(open(HERE / "selection.csv")))
for r in rows:
    m = np.array(Image.open(RAW_DIR / "masks" / r["name"]))
    assert m.max() <= 30, r["name"]
    Image.fromarray(m.astype(np.uint8)).resize((1920, 1080), Image.NEAREST).save(OUT_DIR / r["split"] / "labels" / r["name"])
(OUT_DIR / "LABEL_SPACE").write_text("sanpo\n")
print(f"wrote {len(rows)} label files in SANPO ids to {OUT_DIR}")
