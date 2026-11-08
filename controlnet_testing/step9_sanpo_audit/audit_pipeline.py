"""
Step 9 — end-to-end audit of the dataset-2 (SANPO) data and mask pipeline, CPU only.

Every stage is tested against an independent re-computation, not just read:

  A. raw SANPO mask -> stored label      (resize, ids, class shares)
  B. stored label   -> dataset sample    (crop geometry, image/mask alignment, palette colours)
  C. dataset sample -> what the pipeline sees at generation time (PIL round trip of the colours)
  D. segmentor target (mask_seg)         (independent mapping vs the dataset's)
  E. splits and prompts                  (session disjointness, prompt rows, object words vs mask)
  F. frozen segmentor on REAL SANPO test images (sanity of target mapping + ceiling), small sample

Run from DDPM/ (segmentor libs on PYTHONPATH):
    DDPM_DATA_ROOT=$PWD/data_sanpo HF_HOME=... PYTHONPATH=$PWD:<segdeps> python controlnet_testing/step9_sanpo_audit/audit_pipeline.py
"""

import csv
import os
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data_prep"))
sys.path.insert(0, str(ROOT / "data_prep" / "sanpo"))
assert os.environ.get("DDPM_DATA_ROOT", "").endswith("data_sanpo")

import numpy as np
import torch
from PIL import Image

from class_mapping_sanpo import COLOR_PALETTE, SANPO_TO_V1, SANPO_TO_ADE, CLASS_NAMES
from common import RAW_DIR
from prompt_templates import CLASS_PROMPT_NAMES
from src import config
from src.data.dataset import ControlNetDataset, PALETTE
from src.segmentor import V1_TO_SEG, SEG_IGNORE, load_segmentor, segmentor_logits, confusion_matrix, miou_from_confusion

D2 = Path(config.DATA_DIR)
random.seed(0)
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)


meta = list(csv.DictReader(open(D2 / "metadata.csv")))
by_split = {s: [r for r in meta if r["split"] == s] for s in ("train", "test")}

# ---------------------------------------------------------------- A
print("\nA. raw SANPO mask -> stored label")
worst, share_raw, share_lbl = 1.0, np.zeros(31), np.zeros(31)
for r in random.sample(meta, 40):
    raw = Image.open(RAW_DIR / "masks" / r["name"])
    expect = np.array(raw.resize((1920, 1080), Image.NEAREST))
    lbl = np.array(Image.open(D2 / r["split"] / "labels" / r["name"]))
    worst = min(worst, (expect == lbl).mean())
    share_raw += np.bincount(np.array(raw).ravel(), minlength=31)[:31] / np.array(raw).size
    share_lbl += np.bincount(lbl.ravel(), minlength=31)[:31] / lbl.size
check("stored label == nearest-resized raw SANPO mask (40 random frames)", worst == 1.0, f"min agreement {worst:.4f}")
check("class shares preserved by the resize (max abs diff < 0.2 % points)",
      np.abs(share_raw - share_lbl).max() / 40 < 0.002, f"{100 * np.abs(share_raw - share_lbl).max() / 40:.3f} %")
img_sizes = Counter(Image.open(D2 / r["split"] / "images" / r["name"]).size for r in random.sample(meta, 40))
check("images are 1920x1080 RGB", set(img_sizes) == {(1920, 1080)}, str(dict(img_sizes)))

# ---------------------------------------------------------------- B
print("\nB. stored label -> dataset sample")
ds_test = ControlNetDataset(config.VAL_IMAGES_DIR, config.VAL_MASKS_DIR, config.VAL_PROMPTS_FILE, resolution=512, augment_masks=False)
pal = np.array([COLOR_PALETTE[i] for i in range(31)])
geo_ok, col_ok, align_ok = True, True, True
for i in random.sample(range(len(ds_test)), 20):
    s = ds_test[i]
    name = s["image_name"]
    # independent geometry: short side 1080 -> 512 gives 910x512 (torchvision int()), centre crop left = round(199)
    lbl = Image.open(D2 / "test" / "labels" / name).resize((910, 512), Image.NEAREST).crop((199, 0, 711, 512))
    exp_ids = np.array(lbl)
    geo_ok &= np.array_equal(exp_ids, s["mask"].numpy())
    exp_rgb = pal[exp_ids]
    got_rgb = (s["mask_rgb"].permute(1, 2, 0).numpy() * 255)
    col_ok &= np.abs(got_rgb - exp_rgb).max() < 1e-3
    img = Image.open(D2 / "test" / "images" / name).convert("RGB").resize((910, 512), Image.BILINEAR).crop((199, 0, 711, 512))
    got_img = (s["image"].permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255
    align_ok &= np.abs(got_img - np.array(img).astype(float)).mean() < 3.0  # bilinear impl. differences only
check("dataset mask == independently resized + centre-cropped label (20 samples)", geo_ok)
check("conditioning colours == ADE20K palette of each SANPO id", col_ok)
check("dataset image == independently resized + centre-cropped photo (same crop as the mask)", align_ok)
check("PALETTE tensor has the 31 SANPO colours in id order",
      np.allclose(PALETTE.numpy() * 255, pal, atol=1e-3) and PALETTE.shape == (31, 3))

# ---------------------------------------------------------------- C
print("\nC. conditioning image as the generation pipeline receives it (float -> uint8 PIL)")
f = (torch.tensor(pal, dtype=torch.float32) / 255.0).numpy()
trunc = (f * 255).astype(np.uint8)              # what evaluation/sampling did before
rounded = np.round(f * 255).astype(np.uint8)
bad = [(CLASS_NAMES[i], tuple(pal[i]), tuple(trunc[i])) for i in range(31) if not np.array_equal(trunc[i], pal[i])]
check("float->uint8 by truncation keeps every colour (eval/sampling code path)", not bad, f"changed: {bad}")
check("float->uint8 with rounding keeps every colour", np.array_equal(rounded, pal))

# ---------------------------------------------------------------- D
print("\nD. segmentor target")
to_seg = np.array([V1_TO_SEG[SANPO_TO_V1[i]] for i in range(31)])
seg_ok = True
for i in random.sample(range(len(ds_test)), 10):
    s = ds_test[i]
    seg_ok &= np.array_equal(to_seg[s["mask"].numpy()], s["mask_seg"].numpy())
check("dataset mask_seg == SANPO -> V1 -> segmentor mapping applied to the mask", seg_ok)
print("     SANPO class -> segmentor target:",
      {CLASS_NAMES[i]: ("ignore" if to_seg[i] == SEG_IGNORE else int(to_seg[i])) for i in range(31)})

# ---------------------------------------------------------------- E
print("\nE. splits and prompts")
tr_sessions = {r["session"] for r in by_split["train"]}
te_sessions = {r["session"] for r in by_split["test"]}
check("train and test sessions are disjoint", not (tr_sessions & te_sessions), f"{len(tr_sessions)} / {len(te_sessions)} sessions")
check("no duplicate frame names", len({r["name"] for r in meta}) == len(meta))
for split in ("train", "test"):
    rows = list(csv.DictReader(open(D2 / split / "prompts" / "prompts_meta.csv")))
    names = {r["name"] for r in by_split[split]}
    check(f"{split}: prompts_meta.csv has exactly one row per frame", len(rows) == len(names) and {r["Image_Name"] for r in rows} == names)
# object words in prompts must be present (>= 0.2 % of the 512 crop) in the mask
to_v1 = np.array([SANPO_TO_V1[i] for i in range(31)])
name_to_v1 = {}
for k, v in CLASS_PROMPT_NAMES.items():
    if v:
        name_to_v1.setdefault(v, []).append(k)
import re
from prompt_templates import get_present_objects
rows = {r["Image_Name"]: r["Text_Prompt"] for r in csv.DictReader(open(D2 / "test" / "prompts" / "prompts_meta.csv"))}
wrong = differs = 0
for r in by_split["test"]:
    m = to_v1[np.array(Image.open(D2 / "test" / "labels" / r["name"]).resize((910, 512), Image.NEAREST).crop((199, 0, 711, 512)))]
    cnt = np.bincount(m.ravel(), minlength=35)
    p = rows[r["name"]]
    words = []
    if p.endswith(" visible"):  # object phrase, e.g. "car, street pole, and fence visible"
        words = [w for w in (re.sub(r"^and\s+", "", x.strip()) for x in re.split(r",\s*|\s+and\s+", p[:-8])) if w in name_to_v1]
    wrong += sum(not any(cnt[i] >= 0.002 * m.size for i in name_to_v1[w]) for w in words)
    differs += words != get_present_objects(m, min_area_frac=0.002)
check("every object named in a test prompt covers >= 0.2 % of the crop", wrong == 0, f"{wrong} unsupported mentions")
check("object lists equal an independent recomputation from the masks", differs == 0, f"{differs} of {len(rows)} differ")
meta_test = {r["name"]: r for r in by_split["test"]}
mismatch = sum(("snow" in rows[n]) != ("snowy" in meta_test[n]["weather"]) for n in rows)
check("weather words in prompts_meta match the recorded session weather", mismatch == 0, f"{mismatch} mismatches")

# ---------------------------------------------------------------- F
print("\nF. frozen segmentor on real SANPO test photos (CPU, 30 frames)")
seg = load_segmentor("cpu")
conf = torch.zeros(10, 10, dtype=torch.long)
with torch.no_grad():
    for i in random.sample(range(len(ds_test)), 30):
        s = ds_test[i]
        conf += confusion_matrix(segmentor_logits(seg, (s["image"] * 0.5 + 0.5).unsqueeze(0)).argmax(1), s["mask_seg"].unsqueeze(0))
miou, acc, per = miou_from_confusion(conf)
print(f"     mIoU {miou:.3f}, pixel acc {acc:.3f}, per class " + ", ".join(f"{k} {v:.2f}" for k, v in per.items()))
check("segmentor agrees with SANPO targets well enough to be a judge (pixel acc > 0.8)", acc > 0.8)

# ---------------------------------------------------------------- G
print("\nG. training-time options added for dataset 2")
ds_flip = ControlNetDataset(config.VAL_IMAGES_DIR, config.VAL_MASKS_DIR, config.VAL_PROMPTS_FILE, resolution=512,
                            augment_masks=False, hflip=True)
plain, flipped_seen, unflipped_seen, ok = ds_test[3], 0, 0, True
for k in range(12):
    torch.manual_seed(k)
    f = ds_flip[3]
    same = torch.equal(f["mask"], plain["mask"])
    mirrored = torch.equal(f["mask"], plain["mask"].flip(-1))
    if same and not mirrored:
        unflipped_seen += 1
        ok &= torch.equal(f["image"], plain["image"]) and torch.equal(f["mask_seg"], plain["mask_seg"])
    else:
        flipped_seen += 1
        ok &= mirrored and torch.equal(f["image"], plain["image"].flip(-1)) and torch.equal(f["mask_seg"], plain["mask_seg"].flip(-1)) \
              and torch.equal(f["mask_rgb"], plain["mask_rgb"].flip(-1))
check("joint flip: image, mask, colours and segmentor target flipped together (or not at all)",
      ok and flipped_seen > 0 and unflipped_seen > 0, f"{flipped_seen} flipped / {unflipped_seen} not, of 12 draws")
config.VAL_SUBSET_SIZE = 150
from src.data.dataset import create_val_dataset
sub = create_val_dataset(resolution=512)
names = [sub.dataset.data[i]["image_name"] for i in sub.indices]
sess = {n.rsplit("_", 1)[0] for n in names}
check("evaluation subset: 150 fixed samples covering all 30 test sessions", len(names) == 150 and len(sess) == 30,
      f"{len(names)} samples, {len(sess)} sessions")
check("full=True ignores the subset (downstream test uses all test images)", len(create_val_dataset(resolution=512, full=True)) == 522)

print(f"\n{sum(results)}/{len(results)} checks passed")
