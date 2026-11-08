"""
SANPO helper — keep SANPO's ORIGINAL masks (class id per pixel, 31 classes) for the selected frames.

Saved as single-channel PNGs (SANPO class id) in RAW_DIR/masks/<name>, full 2208x1242
resolution. The conversion step builds the project labels from these, so the
mapping can be changed later without downloading again.

    https_proxy=http://proxy.nhr.fau.de:80 python data_prep/sanpo/fetch_raw_masks.py
"""

import csv
import io
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from common import BUCKET, CAMERA, HERE, RAW_DIR, fetch

OUT = RAW_DIR / "masks"


def one(row):
    out = OUT / row["name"]
    if out.exists():
        return "skipped"
    url = f"{BUCKET}/{row['session']}/{CAMERA}/segmentation_masks/{int(row['frame']):06d}.png"
    for attempt in range(3):
        try:
            m = np.array(Image.open(io.BytesIO(fetch(url))))
            break
        except Exception as e:
            if attempt == 2:
                return f"FAILED {row['name']}: {e}"
    semantic = m[..., 0] if m.ndim == 3 else m
    Image.fromarray(semantic.astype(np.uint8)).save(out)
    return "ok"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(HERE / "selection.csv")))
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(one, rows))
    print({k: res.count(k) for k in ("ok", "skipped")}, "failed:", [r for r in res if r.startswith("FAILED")][:5])


if __name__ == "__main__":
    main()
