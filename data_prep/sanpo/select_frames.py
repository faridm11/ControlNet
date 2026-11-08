"""
SANPO step 2 — choose which frames become dataset 2 (deterministic, no download).

Only HUMAN-annotated frames of the chest camera are used (machine-propagated
labels are noisier). Consecutive annotated frames are 0.4 s apart and nearly
identical, so frames are thinned per session:

    train: every 3rd human-annotated frame (~1.2 s apart)  -> ~2,900 frames from 116 sessions
    test:  every 5th human-annotated frame (~2.0 s apart)  -> ~470 frames from 30 sessions

Train and test come from SANPO's official, disjoint session splits.
Writes data_prep/sanpo/selection.csv (committed, so the dataset is reproducible).
"""

import csv
import json

from common import HERE, RAW_DIR

STRIDE = {"train": 3, "test": 5}


def main():
    index = json.load(open(RAW_DIR / "sanpo_index.json"))
    rows = []
    for r in sorted(index, key=lambda r: (r["split"], r["session"])):
        if not r["has_masks"]:
            continue
        for frame in r["human_frames"][::STRIDE[r["split"]]]:
            rows.append({
                "split": r["split"], "session": r["session"], "frame": frame,
                "name": f"sanpo_{r['session'].lstrip('-')}_{frame:06d}.png",
                "weather": "+".join(w.replace("WEATHER_CONDITION_", "").lower() for w in (r.get("weather_conditions") or [])),
                "visibility": (r.get("visibility") or "").replace("VISIBILITY_", "").lower(),
                "environment": "+".join(e.replace("ENVIRONMENT_TYPE_", "").lower() for e in (r.get("environment_types") or [])),
            })
    with open(HERE / "selection.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for split in ("train", "test"):
        n = [x for x in rows if x["split"] == split]
        print(f"{split}: {len(n)} frames from {len({x['session'] for x in n})} sessions, "
              f"{sum('snowy' in x['weather'] for x in n)} snowy")


if __name__ == "__main__":
    main()
