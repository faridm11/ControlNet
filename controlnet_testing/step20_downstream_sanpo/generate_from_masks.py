"""
Step 20b — generate one image per mask of a list in splits.json (--split, default other_masks) with one ControlNet.

Same seed for every model, so base and fine-tuned images share their noise. Output:
    <out>/images/<name>.jpg   generated image, 512x512
    <out>/masks/<name>.png    target of the shared judge classes (255 = not scored: white/void etc.)
Run from DDPM/ with DDPM_DATA_ROOT=.../data_sanpo, DDPM_SEG_JUDGE=ade20k, prompts_meta:
    python controlnet_testing/step20_downstream_sanpo/generate_from_masks.py --controlnet <pretrained | <run>_Full> --out <dir>
"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from diffusers import ControlNetModel, DDIMScheduler, StableDiffusionControlNetPipeline
from PIL import Image
from src import config
from src.data import create_train_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--controlnet", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--split", default="other_masks", help="list in splits.json")
ap.add_argument("--per-mask", type=int, default=1, help="images per mask, different noise; names get _0, _1, ... when > 1")
args = ap.parse_args()
out = Path(args.out)
(out / "images").mkdir(parents=True, exist_ok=True)
(out / "masks").mkdir(parents=True, exist_ok=True)
path = config.CONTROLNET_MODEL_ID if args.controlnet == "pretrained" else args.controlnet
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    config.SD_MODEL_ID, controlnet=ControlNetModel.from_pretrained(path, torch_dtype=torch.float16),
    scheduler=DDIMScheduler.from_pretrained(config.SD_MODEL_ID, subfolder="scheduler"),
    safety_checker=None, feature_extractor=None, torch_dtype=torch.float16).to("cuda")
pipe.set_progress_bar_config(disable=True)
ds = create_train_dataset(resolution=config.RESOLUTION)
pos = {d["image_name"]: i for i, d in enumerate(ds.data)}
names = json.load(open(Path(__file__).parent / "splits.json"))[args.split]
jobs = [(n, k) for k in range(args.per_mask) for n in names]
gen = torch.Generator(device="cuda").manual_seed(args.seed)
for s in range(0, len(jobs), args.batch):
    part = jobs[s:s + args.batch]
    items = [ds[pos[n]] for n, _ in part]
    controls = [Image.fromarray((it["mask_rgb"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)) for it in items]
    images = pipe(prompt=[it["prompt"] for it in items], image=controls, num_inference_steps=config.NUM_INFERENCE_STEPS,
                  guidance_scale=config.GUIDANCE_SCALE, controlnet_conditioning_scale=config.CONTROLNET_CONDITIONING_SCALE,
                  generator=gen).images
    for it, (_, k), im in zip(items, part, images):
        stem = Path(it["image_name"]).stem + (f"_{k}" if args.per_mask > 1 else "")
        im.save(out / "images" / f"{stem}.jpg", quality=95)
        Image.fromarray(it["mask_seg"].numpy().astype(np.uint8)).save(out / "masks" / f"{stem}.png")
    if (s // args.batch) % 10 == 0:
        print(f"{min(s + args.batch, len(jobs))}/{len(jobs)}", flush=True)
print(f"done: {len(jobs)} images -> {out}")
