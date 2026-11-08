"""
Side-by-side comparison images: for the same test masks, prompts and noise, show
    real photo | conditioning mask | pretrained | model A | model B | ...

Each model is a merged ControlNet directory (the "<run_id>_Full" folder every
training run writes), so models with different LoRA ranks can be compared.

Run from DDPM/ on a GPU node (see jobs/comparison_grid.sh):
    python controlnet_testing/reporting/make_comparison_grid.py \
        --models "old config=<dir>/<id>_Full" "LR 2e-4=<dir>/<id>_Full" \
        --indices 0 7 14 21 28 35 --out controlnet_testing/reporting/comparison/phase1
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from diffusers import ControlNetModel, DDIMScheduler, StableDiffusionControlNetPipeline
from PIL import Image, ImageDraw, ImageFont

from src import config
from src.data import create_val_dataset


def label_bar(labels, cell_w, height=36):
    bar = Image.new("RGB", (cell_w * len(labels), height), "white")
    draw = ImageDraw.Draw(bar)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    for i, text in enumerate(labels):
        draw.text((i * cell_w + 8, 6), text, fill="black", font=font)
    return bar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[], help='"label=path/to/<run_id>_Full" entries')
    ap.add_argument("--indices", nargs="+", type=int, default=[0, 7, 14, 21, 28, 35])
    ap.add_argument("--seed", type=int, default=config.EVAL_SEED)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    models = [("pretrained", config.CONTROLNET_MODEL_ID)] + [tuple(m.split("=", 1)) for m in args.models]

    ds = create_val_dataset(resolution=config.RESOLUTION)
    samples = [ds[i] for i in args.indices]

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        config.SD_MODEL_ID, controlnet=ControlNetModel.from_pretrained(config.CONTROLNET_MODEL_ID),
        scheduler=DDIMScheduler.from_pretrained(config.SD_MODEL_ID, subfolder="scheduler"),
        safety_checker=None, feature_extractor=None,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    generated = {}  # label -> list of PIL
    for label, path in models:
        pipe.controlnet = ControlNetModel.from_pretrained(path).to(device)
        imgs = []
        for k, s in enumerate(samples):
            control = Image.fromarray((s["mask_rgb"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
            g = torch.Generator(device=device).manual_seed(args.seed + k)  # same noise for every model
            imgs.append(pipe(prompt=s["prompt"], image=control, num_inference_steps=config.NUM_INFERENCE_STEPS,
                             guidance_scale=config.GUIDANCE_SCALE, negative_prompt=config.NEGATIVE_PROMPT or None,
                             controlnet_conditioning_scale=config.CONTROLNET_CONDITIONING_SCALE,
                             generator=g).images[0])
        generated[label] = imgs
        print(f"done: {label}")

    labels = ["real photo", "mask"] + [m[0] for m in models]
    res = config.RESOLUTION
    rows = []
    for k, s in enumerate(samples):
        real = Image.fromarray(((s["image"].permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255).round().astype(np.uint8))
        control = Image.fromarray((s["mask_rgb"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
        row = Image.new("RGB", (res * len(labels), res))
        for j, im in enumerate([real, control] + [generated[m[0]][k] for m in models]):
            row.paste(im, (j * res, 0))
        row.save(out / f"row_{s['image_name']}")
        rows.append(row)
        (out / f"row_{Path(s['image_name']).stem}_prompt.txt").write_text(s["prompt"] + "\n")

    bar = label_bar(labels, res)
    sheet = Image.new("RGB", (res * len(labels), bar.height + res * len(rows)), "white")
    sheet.paste(bar, (0, 0))
    for k, row in enumerate(rows):
        sheet.paste(row, (0, bar.height + k * res))
    sheet.save(out / "comparison_sheet.jpg", quality=90)
    print(f"Wrote {out / 'comparison_sheet.jpg'}")


if __name__ == "__main__":
    main()
