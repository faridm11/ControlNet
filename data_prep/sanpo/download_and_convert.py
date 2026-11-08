"""
SANPO step 3 — download the selected frames and convert them to this project's format.

For each row of selection.csv:
  - frame  : camera_chest/left/video_frames/<frame>.png (2208x1242) -> resized to 1920x1080 RGB PNG
  - mask   : first version wrote labels in the project's 35-class ids. Labels are now built by
             build_labels.py from SANPO's original masks (fetch_raw_masks.py), keeping SANPO's
             31 ids; the label files written here are overwritten by that step.
1920x1080 is the same size and aspect ratio as dataset 1, so the training crop
(short side 512, centre crop) behaves identically.

Output (same layout as DDPM/data, so DDPM_DATA_ROOT=<...>/data_sanpo switches datasets):
    data_sanpo/{train,test}/images/<name>.png
    data_sanpo/{train,test}/labels/<name>.png
    data_sanpo/metadata.csv     (name, split, session, frame, weather, visibility, environment)

Already converted files are skipped, so the script can be re-run after an interruption.
    https_proxy=http://proxy.nhr.fau.de:80 python data_prep/sanpo/download_and_convert.py
"""

import csv
import io
import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from class_mapping_sanpo import SANPO_TO_V1
from common import BUCKET, CAMERA, HERE, OUT_DIR, fetch

SIZE = (1920, 1080)
LUT = np.full(256, 0, dtype=np.uint8)
for s, v in SANPO_TO_V1.items():
    LUT[s] = v


def convert(row):
    img_out = OUT_DIR / row["split"] / "images" / row["name"]
    lbl_out = OUT_DIR / row["split"] / "labels" / row["name"]
    if img_out.exists() and lbl_out.exists():
        return "skipped"
    base = f"{BUCKET}/{row['session']}/{CAMERA}"
    fname = f"{int(row['frame']):06d}.png"
    for attempt in range(3):
        try:
            frame = Image.open(io.BytesIO(fetch(f"{base}/video_frames/{fname}"))).convert("RGB")
            mask = np.array(Image.open(io.BytesIO(fetch(f"{base}/segmentation_masks/{fname}"))))
            break
        except Exception as e:
            if attempt == 2:
                return f"FAILED {row['name']}: {e}"
    semantic = mask[..., 0] if mask.ndim == 3 else mask
    if semantic.max() > 30:
        return f"FAILED {row['name']}: unexpected class id {semantic.max()}"
    labels = Image.fromarray(LUT[semantic]).resize(SIZE, Image.NEAREST)
    frame.resize(SIZE, Image.BILINEAR).save(img_out)
    labels.save(lbl_out)
    return "ok"


def main():
    rows = list(csv.DictReader(open(HERE / "selection.csv")))
    for split in ("train", "test"):
        for sub in ("images", "labels", "prompts"):
            (OUT_DIR / split / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy(HERE / "selection.csv", OUT_DIR / "metadata.csv")

    done = {"ok": 0, "skipped": 0}
    failed = []
    with ThreadPoolExecutor(12) as ex:
        for i, res in enumerate(ex.map(convert, rows), 1):
            if res in done:
                done[res] += 1
            else:
                failed.append(res)
            if i % 200 == 0:
                print(f"{i}/{len(rows)} ok={done['ok']} skipped={done['skipped']} failed={len(failed)}", flush=True)
    print(f"done: ok={done['ok']} skipped={done['skipped']} failed={len(failed)}")
    for f in failed[:20]:
        print(" ", f)


if __name__ == "__main__":
    main()
