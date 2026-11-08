"""
Generate images from a trained ControlNet model.

Usage — from a merged checkpoint (recommended after training):
    python inference.py \
        --controlnet outputs/checkpoints/1581867_Full \
        --mask data/test/labels/image_128.png \
        --prompt "urban sidewalk with trees and parked cars" \
        --output out.png

Usage — from a LoRA checkpoint:
    python inference.py \
        --lora outputs/checkpoints/1581867_LoRA.pt \
        --mask data/test/labels/image_128.png \
        --prompt "urban sidewalk with trees and parked cars" \
        --output out.png

Options:
    --num-images        Number of images to generate (default: 4)
    --steps             DDIM inference steps (default: 35)
    --guidance          CFG guidance scale (default: 6.0)
    --control-strength  ControlNet conditioning scale (default: 1.0)
    --seed              Random seed (default: 42)
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DDIMScheduler,
)
from src import config
from src.models.diffusion_controlnet import DiffusionControlNet, create_pipeline


def load_mask(mask_path: str, resolution: int = 512) -> Image.Image:
    """
    Load a segmentation mask and return the RGB conditioning image the ControlNet expects.

    - Class-ID mask (single channel, as stored in <data root>/*/labels; V1 ids for data/, SANPO ids for data_sanpo/):
      remapped for the active CLASS_VERSION and coloured with the training palette.
    - RGB mask: assumed to be already palette-coloured, used as is.

    Geometry matches training: resize the short side to `resolution`, then centre-crop.
    """
    from src.data.dataset import PALETTE, _REMAP, _RAW_NUM_CLASSES

    mask = Image.open(mask_path)
    if mask.mode in ("L", "P", "I", "I;16"):
        ids = np.array(mask).astype(np.int64)
        if ids.max() >= _RAW_NUM_CLASSES:
            raise ValueError(f"{mask_path}: class IDs up to {ids.max()}, expected 0-{_RAW_NUM_CLASSES - 1}")
        if _REMAP is not None:
            ids = _REMAP[ids]
        rgb = (PALETTE.numpy()[ids] * 255).round().astype(np.uint8)
        mask = Image.fromarray(rgb, mode="RGB")
    else:
        mask = mask.convert("RGB")

    w, h = mask.size
    scale = resolution / min(w, h)
    new_w, new_h = max(resolution, round(w * scale)), max(resolution, round(h * scale))
    mask = mask.resize((new_w, new_h), Image.NEAREST)
    left, top = (new_w - resolution) // 2, (new_h - resolution) // 2
    return mask.crop((left, top, left + resolution, top + resolution))


def build_pipeline_from_merged(controlnet_dir: str, device: str):
    """Load pipeline from a merged _Full/ checkpoint (no LoRA needed)."""
    print(f"Loading merged ControlNet from {controlnet_dir}...")
    controlnet = ControlNetModel.from_pretrained(controlnet_dir, torch_dtype=torch.float16)
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        config.SD_MODEL_ID,
        controlnet=controlnet,
        scheduler=DDIMScheduler.from_pretrained(config.SD_MODEL_ID, subfolder="scheduler"),
        safety_checker=None,
        feature_extractor=None,
        torch_dtype=torch.float16,
    )
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=False)
    print("Pipeline ready.")
    return pipeline


def build_pipeline_from_lora(lora_path: str, device: str):
    """Load pipeline from a LoRA checkpoint (merges on-the-fly)."""
    print(f"Loading LoRA checkpoint from {lora_path}...")
    model = DiffusionControlNet(pretrained=True, device=device)
    model.load_checkpoint(lora_path)
    model.eval()
    pipeline = create_pipeline(model, device=device)
    pipeline.set_progress_bar_config(disable=False)
    print("Pipeline ready.")
    return pipeline


def generate(
    pipeline,
    mask_pil: Image.Image,
    prompt: str,
    num_images: int,
    steps: int,
    guidance: float,
    control_strength: float,
    seed: int,
    device: str,
) -> list:
    generator = torch.Generator(device=device).manual_seed(seed)
    result = pipeline(
        prompt=prompt,
        image=mask_pil,
        num_images_per_prompt=num_images,
        num_inference_steps=steps,
        guidance_scale=guidance,
        controlnet_conditioning_scale=control_strength,
        generator=generator,
    )
    return result.images


def save_grid(mask_pil: Image.Image, images: list, output_path: str):
    """Save mask + all generated images as a side-by-side grid."""
    w, h = images[0].size
    grid = Image.new("RGB", (w * (1 + len(images)), h))
    grid.paste(mask_pil.resize((w, h), Image.NEAREST), (0, 0))
    for i, img in enumerate(images):
        grid.paste(img, ((i + 1) * w, 0))
    grid.save(output_path)
    print(f"Saved grid: {output_path}")
    # Also save individual images
    out = Path(output_path)
    for i, img in enumerate(images):
        img.save(out.parent / f"{out.stem}_{i}.png")
        print(f"  {out.stem}_{i}.png")


def main():
    parser = argparse.ArgumentParser(description="ControlNet inference")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--controlnet", help="Path to merged _Full/ checkpoint dir")
    source.add_argument("--lora",       help="Path to _LoRA.pt checkpoint file")

    parser.add_argument("--mask",             required=True,  help="Path to segmentation mask PNG")
    parser.add_argument("--prompt",           required=True,  help="Text prompt")
    parser.add_argument("--output",           default="output.png", help="Output image path")
    parser.add_argument("--num-images",       type=int,   default=4)
    parser.add_argument("--steps",            type=int,   default=35)
    parser.add_argument("--guidance",         type=float, default=6.0)
    parser.add_argument("--control-strength", type=float, default=1.0)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--resolution",       type=int,   default=512)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.controlnet:
        pipeline = build_pipeline_from_merged(args.controlnet, device)
    else:
        pipeline = build_pipeline_from_lora(args.lora, device)

    mask_pil = load_mask(args.mask, resolution=args.resolution)

    print(f"\nGenerating {args.num_images} image(s)...")
    print(f"  Prompt:           {args.prompt}")
    print(f"  Steps:            {args.steps}")
    print(f"  Guidance:         {args.guidance}")
    print(f"  Control strength: {args.control_strength}")
    print(f"  Seed:             {args.seed}")

    images = generate(
        pipeline=pipeline,
        mask_pil=mask_pil,
        prompt=args.prompt,
        num_images=args.num_images,
        steps=args.steps,
        guidance=args.guidance,
        control_strength=args.control_strength,
        seed=args.seed,
        device=device,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_grid(mask_pil, images, args.output)


if __name__ == "__main__":
    main()
