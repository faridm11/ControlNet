"""Step 19 (CPU): which RAW judge classes does the judge predict for each of our target classes on the REAL
SANPO photos (150-image evaluation subset)? Shows where the real-photo mIoU is lost to definition mismatches
(e.g. ADE20K 'floor' on a paved plaza falls into 'other'). Judge from DDPM_SEG_JUDGE (ade20k or mapillary)."""
import os, sys, json
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, torch
from src import config, segmentor as S
from src.data.dataset import create_val_dataset

torch.set_num_threads(16)
judge_name = S.SEG_JUDGE
ds = create_val_dataset(config.RESOLUTION)
j = S.load_segmentor("cpu")
inner = j.model
if judge_name == "ade20k":
    id2label = inner.config.id2label; mapping = S.ADE20K_TO_JUDGE
else:
    id2label = inner.config.id2label; mapping = S.MAPILLARY_TO_JUDGE
mean = torch.tensor(S._MEAN).view(1, 3, 1, 1); std = torch.tensor(S._STD).view(1, 3, 1, 1)
counts = defaultdict(lambda: np.zeros(len(id2label), dtype=np.int64))
with torch.inference_mode():
    for i in range(0, len(ds), int(os.environ.get("DIAG_STEP", "1"))):
        b = ds[i]
        x = ((b["image"][None] * 0.5 + 0.5).clamp(0, 1) - mean) / std
        x = torch.nn.functional.interpolate(x, scale_factor=j.scale, mode="bilinear", align_corners=False) if j.scale != 1 else x
        if judge_name == "ade20k":
            lg = inner(pixel_values=x).logits
            raw = torch.nn.functional.interpolate(lg, size=(512, 512), mode="bilinear", align_corners=False).argmax(1)[0]
        else:
            out = inner(pixel_values=x)
            sem = torch.einsum("bqc,bqhw->bchw", out.class_queries_logits.softmax(-1)[..., :-1], out.masks_queries_logits.sigmoid())
            raw = torch.nn.functional.interpolate(sem, size=(512, 512), mode="bilinear", align_corners=False).argmax(1)[0]
        t = b["mask_seg"]
        for k in range(12):
            sel = t == k
            if sel.any():
                counts[k] += np.bincount(raw[sel].numpy(), minlength=len(id2label))
        if i % 30 == 0:
            print(f"{i}/{len(ds)}", flush=True)
report = {}
for k in range(12):
    c = counts[k]
    if c.sum() == 0:
        continue
    top = np.argsort(-c)[:6]
    report[S.SEG_CLASS_NAMES[k]] = [(id2label[int(r)], S.SEG_CLASS_NAMES[mapping[r]], round(100 * c[r] / c.sum(), 1)) for r in top]
    print(f"{S.SEG_CLASS_NAMES[k]:11s} ({c.sum():8d} px) -> " + ", ".join(f"{n} [{m}] {p}%" for n, m, p in report[S.SEG_CLASS_NAMES[k]]))
json.dump(report, open(Path(__file__).parent / f"raw_classes_{judge_name}.json", "w"), indent=1)
