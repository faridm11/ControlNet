"""
SANPO helper (2026-10-25) — how much white (void) could a different frame choice remove?

Dataset 2 uses every 3rd (train) / 5th (test) human-annotated chest frame. Here the masks of ALL
human-annotated chest frames are fetched (small single-channel PNGs, ~12k files, outside the repo in
RAW_DIR/masks_all/), the void share of each is measured (void = SANPO ids mapped to white:
unlabeled, inaccessible surface, obstacle, bus stop), and an alternative selection is simulated:
same sessions, same number of frames per session, but the lowest-void frames, at least MIN_GAP
frames apart so that near-duplicates are not picked. Writes void_per_frame.csv and prints a summary.
Nothing in data_sanpo/ is changed.

    https_proxy=http://proxy.nhr.fau.de:80 python data_prep/sanpo/analyze_void_reselection.py
"""

import csv
import io
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from class_mapping_sanpo import SANPO_TO_ADE
from common import BUCKET, CAMERA, HERE, RAW_DIR, fetch

OUT = RAW_DIR / "masks_all"
VOID = [s for s, (a, _) in SANPO_TO_ADE.items() if a is None]
STRIDE = {"train": 3, "test": 5}            # current selection (select_frames.py)
MIN_GAP = {"train": 12, "test": 18}          # annotated frames are 6 apart; ≥ 2-3 annotation steps apart


def stats(session, frame):
    """(void share, rider pixels share) of one frame; mask cached on disk."""
    path = OUT / session / f"{frame:06d}.png"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        data = None
        for _ in range(3):
            try:
                data = fetch(f"{BUCKET}/{session}/{CAMERA}/segmentation_masks/{frame:06d}.png")
                break
            except Exception:
                pass
        if data is None:
            return None
        path.write_bytes(data)
    m = np.array(Image.open(path))
    m = (m[..., 0] if m.ndim == 3 else m)[::4, ::4]  # channel 0 = semantic class id (as in fetch_raw_masks.py)
    return float(np.isin(m, VOID).mean()), float((m == 13).mean())


def pick(frames, void, n, gap):
    """n lowest-void frames, greedily, keeping at least `gap` frame numbers between picks."""
    chosen = []
    for f in sorted(frames, key=lambda f: void[f]):
        if all(abs(f - c) >= gap for c in chosen):
            chosen.append(f)
        if len(chosen) == n:
            break
    return chosen


def main():
    recs = [r for r in json.load(open(RAW_DIR / "sanpo_index.json")) if r["has_masks"]]
    jobs = [(r["session"], f) for r in recs for f in r["human_frames"]]
    print(f"{len(recs)} sessions, {len(jobs)} annotated chest frames")
    with ThreadPoolExecutor(16) as ex:
        res = dict(zip(jobs, ex.map(lambda j: stats(*j), jobs)))
    missing = sum(v is None for v in res.values())
    with open(HERE / "void_per_frame.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "session", "frame", "void_share", "rider_share"])
        for r in recs:
            for f in r["human_frames"]:
                if res[(r["session"], f)] is not None:
                    w.writerow([r["split"], r["session"], f, *(f"{x:.5f}" for x in res[(r["session"], f)])])
    print(f"missing masks: {missing}")

    for split in ("train", "test"):
        cur, alt, allv, riders_cur, riders_alt = [], [], [], 0, 0
        for r in (r for r in recs if r["split"] == split):
            fr = [f for f in r["human_frames"] if res[(r["session"], f)] is not None]
            void = {f: res[(r["session"], f)][0] for f in fr}
            rider = {f: res[(r["session"], f)][1] for f in fr}
            now = [f for f in r["human_frames"][::STRIDE[split]] if f in void]
            new = pick(fr, void, len(now), MIN_GAP[split])
            cur += [void[f] for f in now]; alt += [void[f] for f in new]; allv += list(void.values())
            riders_cur += sum(rider[f] > 0 for f in now); riders_alt += sum(rider[f] > 0 for f in new)
        cur, alt, allv = map(np.array, (cur, alt, allv))
        print(f"\n{split}: all annotated {len(allv)} | current selection {len(cur)} | low-void selection {len(alt)}")
        for name, v in (("all annotated", allv), ("current (every %dth)" % STRIDE[split], cur), ("low-void, same size", alt)):
            print(f"  {name:22s} void mean {100*v.mean():4.1f}%  median {100*np.median(v):4.1f}%  "
                  f">10%: {100*(v > .1).mean():3.0f}%  >20%: {100*(v > .2).mean():3.0f}%")
        print(f"  frames with riders: current {riders_cur}, low-void {riders_alt}")


if __name__ == "__main__":
    main()
