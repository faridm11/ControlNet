"""
Step 18b — are the generated images worse only where the mask is white (void)?

Uses the images saved by submit_second_judge.sh (images/<model>/, seed 42, 150-image SANPO subset).
The 150 test images are split at the median white share of their conditioning mask into two equal groups,
and everything is copied into two folders for viewing:

    images_by_white/low_white/{real,mask,<model>...}/<name>.png
    images_by_white/high_white/{real,mask,<model>...}/<name>.png
    images_by_white/{low,high}_white_sheet.jpg   (first 8 scenes: real | mask | models)

Per group and model: KID (subset 50, seeded), LPIPS, CLIP prompt / image, mIoU with the Mapillary judge
(512 px, as in training) and the ADE20K judge (640 px, the independent one), plus both judges on the real
photos (ceiling). Results: split_by_white.json and a printed table.

    DDPM_DATA_ROOT=.../data_sanpo DDPM_PROMPTS_FILE_NAME=prompts_meta.csv DDPM_VAL_SUBSET_SIZE=150 \
    DDPM_SEG_JUDGE=ade20k python controlnet_testing/step18_second_judge/split_by_white.py
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

import os
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
HERE = Path(os.environ.get("SPLIT_BASE", Path(__file__).resolve().parent))  # folder holding images/ (override for tests)
from src import config, segmentor as S                      # noqa: E402
from src.data.dataset import create_val_dataset              # noqa: E402
from src.evaluation import ControlNetEvaluator               # noqa: E402
from torchmetrics.image.kid import KernelInceptionDistance   # noqa: E402
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity  # noqa: E402

MODELS = ["pretrained", "B", "seg0.006", "seg0.02", "seg0.05-0.4"]   # folders in images/
LABELS = {"pretrained": "pretrained", "B": "B (no seg loss)", "seg0.006": "seg λ 0.006",
          "seg0.02": "seg λ 0.02", "seg0.05-0.4": "seg λ 0.05→0.4"}
JUDGES = {"mapillary": 1.0, "ade20k": 1.25}                 # judge -> input scale (512 px, 640 px)
OUT = HERE / "images_by_white"


def to_pil(t):  # (3, H, W) in [0, 1]
    return Image.fromarray((t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))


def load_judge(name, scale, device):
    os.environ["DDPM_SEG_INPUT_SCALE"] = str(scale)
    S.SEG_JUDGE = name  # both judges use the same 13 shared classes and targets
    return S.load_segmentor(device)


@torch.inference_mode()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = create_val_dataset(config.RESOLUTION)
    items = [ds[i] for i in range(len(ds))]
    names = [Path(it["image_name"]).with_suffix(".png").name for it in items]
    reals = [(it["image"] * 0.5 + 0.5).clamp(0, 1) for it in items]
    masks = [it["mask_rgb"] for it in items]
    white = np.array([float((m >= 0.99).all(0).float().mean()) for m in masks])
    models = [m for m in MODELS if (HERE / "images" / m).is_dir() and len(list((HERE / "images" / m).glob("*.png"))) == len(ds)]
    print(f"{len(ds)} images, models with all images: {models}")
    thr = float(np.median(white))
    groups = {"low_white": np.where(white <= thr)[0], "high_white": np.where(white > thr)[0]}
    print(f"white share: median {100*thr:.1f}% | low group {len(groups['low_white'])} images (mean {100*white[groups['low_white']].mean():.1f}%), "
          f"high group {len(groups['high_white'])} (mean {100*white[groups['high_white']].mean():.1f}%)")

    # --- folders for viewing
    if OUT.exists():
        shutil.rmtree(OUT)
    for g, idx in groups.items():
        for k in idx:
            for sub, img in (("real", to_pil(reals[k])), ("mask", to_pil(masks[k]))):
                (OUT / g / sub).mkdir(parents=True, exist_ok=True)
                img.save(OUT / g / sub / names[k])
            for m in models:
                (OUT / g / m).mkdir(parents=True, exist_ok=True)
                shutil.copy(HERE / "images" / m / names[k], OUT / g / m / names[k])
        S_ = 256
        cols = ["real", "mask"] + models
        sheet = Image.new("RGB", (S_ * len(cols), 22 + S_ * min(8, len(idx))), "white")
        d = ImageDraw.Draw(sheet)
        for j, c in enumerate(cols):
            d.text((j * S_ + 4, 4), LABELS.get(c, c + " photo" if c == "real" else c), fill="black")
        for r, k in enumerate(idx[:8]):
            for j, c in enumerate(cols):
                sheet.paste(Image.open(OUT / g / c / names[k]).convert("RGB").resize((S_, S_)), (j * S_, 22 + r * S_))
        sheet.save(OUT / f"{g}_sheet.jpg", quality=90)

    # --- metrics per group
    gens = {m: [torch.from_numpy(np.array(Image.open(HERE / "images" / m / n).convert("RGB"))).permute(2, 0, 1)
                for n in names] for m in models}
    targets = [it["mask_seg"] for it in items]
    prompts = [it["prompt"] for it in items]
    ev = ControlNetEvaluator(device)
    res = {g: {m: {} for m in models + ["real photos"]} for g in groups}

    for g, idx in groups.items():
        real_u8 = torch.stack([(reals[k] * 255).round().to(torch.uint8) for k in idx])
        for m in models:
            gen_u8 = torch.stack([gens[m][k] for k in idx])
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(config.EVAL_SEED)
                kid = KernelInceptionDistance(subset_size=min(50, len(idx)), normalize=False).to(device)
                kid.update(real_u8.to(device), real=True); kid.update(gen_u8.to(device), real=False)
                res[g][m]["kid"] = float(kid.compute()[0])
            lp = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(device)
            res[g][m]["lpips"] = float(np.mean([lp(gen_u8[i:i+1].float().to(device) / 255, real_u8[i:i+1].float().to(device) / 255).item()
                                                for i in range(len(idx))]))
            cps, cis = [], []
            for i in range(0, len(idx), 16):
                sl = list(range(i, min(i + 16, len(idx))))
                cp, ci = ev._clip_scores([to_pil(gen_u8[s].float() / 255) for s in sl], [to_pil(reals[idx[s]]) for s in sl],
                                         [prompts[idx[s]] for s in sl])
                cps.append(cp * len(sl)); cis.append(ci * len(sl))
            res[g][m]["clip_prompt"], res[g][m]["clip_img"] = sum(cps) / len(idx), sum(cis) / len(idx)
            print(f"  [{g}] {m}: KID {res[g][m]['kid']:.4f}, CLIP image {res[g][m]['clip_img']:.3f}", flush=True)

    for jname, scale in JUDGES.items():
        judge = load_judge(jname, scale, device)
        print(f"judge {jname} loaded ({device})", flush=True)
        for g, idx in groups.items():
            for m in models + ["real photos"]:
                conf = torch.zeros(S.SEG_NUM_CLASSES, S.SEG_NUM_CLASSES, dtype=torch.long)
                for k in idx:
                    x = reals[k] if m == "real photos" else gens[m][k].float() / 255
                    pred = S.segmentor_logits(judge, x[None].to(device)).argmax(1).cpu()
                    conf += S.confusion_matrix(pred, targets[k][None])
                miou, acc, per = S.miou_from_confusion(conf)
                res[g][m][f"miou_{jname}"] = miou
                res[g][m][f"iou_per_class_{jname}"] = per
                print(f"  [{g}] {m}: mIoU {jname} {miou:.3f}", flush=True)
        del judge
        if device == "cuda":
            torch.cuda.empty_cache()

    meta = {"threshold_white": thr, "groups": {g: [names[k] for k in idx] for g, idx in groups.items()},
            "white_share": dict(zip(names, white.tolist()))}
    json.dump({"meta": meta, "results": res}, open(HERE / "split_by_white.json", "w"), indent=2)
    for g in groups:
        print(f"\n{g} ({len(groups[g])} images, mean white {100*white[groups[g]].mean():.1f}%)")
        print(f"  {'model':16s} {'KID':>7s} {'LPIPS':>6s} {'CLIPimg':>7s} {'mIoU Mapillary':>15s} {'mIoU ADE20K':>12s}")
        for m in models + ["real photos"]:
            r = res[g][m]
            f = lambda k, p=3: f"{r[k]:.{p}f}" if k in r else "–"
            print(f"  {LABELS.get(m, m):16s} {f('kid', 4):>7s} {f('lpips'):>6s} {f('clip_img'):>7s} {f('miou_mapillary'):>15s} {f('miou_ade20k'):>12s}")


if __name__ == "__main__":
    main()
