"""
SANPO step 1 — index all SANPO-Real sessions (metadata only, a few MB).

For every session in the official train/test split, records whether the chest
camera has segmentation masks, which frames are HUMAN-annotated (the others are
machine-propagated and noisier), and the session's weather / visibility /
environment attributes. Writes RAW_DIR/sanpo_index.json.

    https_proxy=http://proxy.nhr.fau.de:80 python data_prep/sanpo/index_sessions.py
"""

import json
import urllib.error
from concurrent.futures import ThreadPoolExecutor

from common import BUCKET, CAMERA, RAW_DIR, fetch, fetch_json


def index_one(session_split):
    sid, split = session_split
    rec = {"session": sid, "split": split, "has_masks": False, "human_frames": []}
    try:
        meta = fetch_json(f"{BUCKET}/{sid}/description.json")["session_video_metadata"]
        rec.update({k: meta.get(k) for k in ("weather_conditions", "visibility", "environment_types",
                                             "human_traffic", "vehicular_traffic", "motion_blur")})
    except Exception as e:
        rec["error"] = f"description: {e}"
    try:
        types = fetch_json(f"{BUCKET}/{sid}/{CAMERA}/frame_segmentation_annotation_type.json")
        rec["has_masks"] = True
        rec["human_frames"] = sorted(int(k) for k, v in types.items() if v == "HUMAN_ANNOTATED")
        rec["n_annotated"] = len(types)
    except urllib.error.HTTPError:
        pass  # session without segmentation for this camera
    return rec


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for split in ("train", "test"):
        f = RAW_DIR / f"{split}_session_ids.txt"
        if not f.exists():
            f.write_bytes(fetch(f"{BUCKET}/splits/{split}_session_ids.txt"))
        jobs += [(s.strip(), split) for s in f.read_text().splitlines() if s.strip()]
    with ThreadPoolExecutor(16) as ex:
        index = list(ex.map(index_one, jobs))
    (RAW_DIR / "sanpo_index.json").write_text(json.dumps(index, indent=1))
    for split in ("train", "test"):
        s = [r for r in index if r["split"] == split]
        m = [r for r in s if r["has_masks"]]
        print(f"{split}: {len(s)} sessions, {len(m)} with chest masks, "
              f"{sum(len(r['human_frames']) for r in m)} human-annotated frames")


if __name__ == "__main__":
    main()
