"""Step 18 CPU check of a second, independent judge (SegFormer-B5, DDPM_SEG_JUDGE=cityscapes or ade20k):
mapping sanity, output is a valid log-distribution, and mIoU on real SANPO photos (every 5th image of the
150-image evaluation subset) at 512 and 1024 px. Real-photo mIoU is the ceiling for the generated images."""
import os, sys, time
from pathlib import Path
os.environ.setdefault("DDPM_SEG_JUDGE", "cityscapes")
os.environ.setdefault("DDPM_DATA_ROOT", "/home/woody/iwi5/iwi5388h/SENSATION/DDPM/data_sanpo")
os.environ.setdefault("DDPM_PROMPTS_FILE_NAME", "prompts_meta.csv")
os.environ.setdefault("DDPM_VAL_SUBSET_SIZE", "150")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from src import config, segmentor as S
from src.data.dataset import create_val_dataset

ok = True
def check(name, cond, info=""):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name, info)

check(f"judge {S.SEG_JUDGE} uses the 13 shared classes", S.SHARED_JUDGE_CLASSES and S.SEG_NUM_CLASSES == 13, S.SEG_CLASS_NAMES)
check("mapping sizes: Cityscapes 19, ADE20K 150", len(S.CITYSCAPES_TO_JUDGE) == 19 and len(S.ADE20K_TO_JUDGE) == 150)
ds = create_val_dataset(config.RESOLUTION)
idx = list(range(0, len(ds), 5))
print(f"dataset {config.DATASET_NAME}, {len(ds)} eval images, testing {len(idx)}")
torch.set_num_threads(16)
for scale in os.environ.get("SCALES", "2 1").split():
    os.environ["DDPM_SEG_INPUT_SCALE"] = scale
    judge = S.load_segmentor("cpu")
    conf = torch.zeros(13, 13, dtype=torch.long); t0 = time.time()
    with torch.inference_mode():
        for k, i in enumerate(idx):
            b = ds[i]
            x = (b["image"].unsqueeze(0) * 0.5 + 0.5).clamp(0, 1)
            lp = S.segmentor_logits(judge, x)
            if k == 0:
                check("log-probs sum to 1 per pixel", torch.allclose(lp.exp().sum(1), torch.ones(1), atol=1e-3))
            conf += S.confusion_matrix(lp.argmax(1), b["mask_seg"].unsqueeze(0))
    miou, acc, per = S.miou_from_confusion(conf)
    print(f"real SANPO photos, judge input {int(512*float(scale))} px: mIoU {miou:.3f}, pixel acc {acc:.3f} ({time.time()-t0:.0f} s) | "
          + ", ".join(f"{c} {v:.2f}" for c, v in per.items()))
    check(f"real-photo mIoU at {int(512*float(scale))} px is a usable ceiling (> 0.45)", miou > 0.45)
print("ALL PASS" if ok else "SOME FAILED")

n = S.SEG_CLASS_NAMES
for k in (0, 1):
    row = conf[k].double() / conf[k].sum()
    print(f"target {n[k]} predicted as:", ", ".join(f"{n[c]} {100*row[c]:.0f}%" for c in row.argsort(descending=True)[:4]))
