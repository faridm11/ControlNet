"""
Step 23 — final flow: the 200 photos, and a data root that contains only them, so a ControlNet can be fine-tuned on them alone.

Adds limited_200 (200 evenly spaced frames of the 300 limited_real) to splits.json and builds data_sanpo_200/:
symlinks to the same images, labels, test split and label space; prompts_meta.csv lists the 200 photos 4 times, so one
epoch is 200 steps and the per-epoch validation does not run 60 times. Nothing in data_sanpo/ is changed.
    python controlnet_testing/step20_downstream_sanpo/make_data_200.py
"""
import csv, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DDPM = HERE.parents[1]
SRC, DST = DDPM / "data_sanpo", DDPM / "data_sanpo_200"
REPEAT = 4

s = json.load(open(HERE / "splits.json"))
L = s["limited_real"]
s["limited_200"] = [L[i] for i in np.linspace(0, len(L) - 1, 200).round().astype(int)]
assert len(set(s["limited_200"])) == 200 and not set(s["limited_200"]) & set(s["other_masks"])
json.dump(s, open(HERE / "splits.json", "w"), indent=1)

for sub in ("images", "labels", "prompts"):
    (DST / "train" / sub).mkdir(parents=True, exist_ok=True)
for name in ("test", "LABEL_SPACE", "metadata.csv"):
    if not (DST / name).exists():
        (DST / name).symlink_to(SRC / name)
with open(SRC / "train/prompts/prompts_meta.csv", newline="") as f:
    rows = {r["Image_Name"]: r for r in csv.DictReader(f)}
with open(DST / "train/prompts/prompts_meta.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["Image_Name", "Mask_Name", "Text_Prompt"])
    w.writeheader()
    for _ in range(REPEAT):
        w.writerows(rows[n] for n in s["limited_200"])
for n in s["limited_200"]:
    for sub, name in (("images", n), ("labels", rows[n]["Mask_Name"])):
        link = DST / "train" / sub / name
        if not link.exists():
            link.symlink_to(SRC / "train" / sub / name)
print(f"{len(s['limited_200'])} photos, {len({n.rsplit('_', 1)[0] for n in s['limited_200']})} sessions -> {DST}")
