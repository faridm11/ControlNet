"""
Sample generation from a trained ControlNet model.

Can be called:
  - During training (via trainer.generate_samples()) for progress snapshots
  - Standalone after training, from the command line:

      python -m src.sampling \
          --checkpoint outputs/checkpoints/best_fid_model.pt \
          --output_dir outputs/final_samples \
          --num_samples 8 \
          --num_images_per_prompt 4
"""

import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

from .models.diffusion_controlnet import DiffusionControlNet, create_pipeline
from .data import create_val_dataset, create_dataloader
from . import config


# ---------------------------------------------------------------------------
# PIL helpers (used both by generate_samples and by Trainer._save_sample_grid)
# ---------------------------------------------------------------------------

def mask_rgb_to_pil(mask_rgb: torch.Tensor, resolution: int) -> Image.Image:
    """
    Convert a (3, H, W) float tensor in [0, 1] to a PIL RGB image.
    Resizes with NEAREST interpolation to preserve class boundaries.
    """
    assert mask_rgb.ndim == 3 and mask_rgb.shape[0] == 3, \
        f"Expected (3, H, W), got {mask_rgb.shape}"
    mask_np = (mask_rgb.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    pil_img = Image.fromarray(mask_np).convert("RGB")
    if pil_img.size != (resolution, resolution):
        pil_img = pil_img.resize((resolution, resolution), Image.NEAREST)
    return pil_img


def save_sample_grid(
    sample_dir: Path,
    idx: int,
    caption: str,
    orig_pil: Image.Image,
    mask_pil: Image.Image,
    generated_imgs: list,
) -> None:
    """
    Save a [input | control | output1 | ... | outputN] side-by-side grid image
    plus individual component files.

    Args:
        sample_dir: Directory to write files into
        idx: Sample index — used in output filenames
        caption: Caption text shown at the top of the grid (e.g. step + prompt)
        orig_pil: Original real image (PIL)
        mask_pil: Segmentation mask as RGB PIL image
        generated_imgs: List of generated PIL images
    """
    img_w, img_h = orig_pil.size
    text_height = 60
    num_gen = len(generated_imgs)
    grid = Image.new("RGB", (img_w * (2 + num_gen), img_h + text_height), color="white")
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    if len(caption) > 150:
        caption = caption[:147] + "..."
    draw.text((10, 10), caption, fill="black", font=font)

    grid.paste(orig_pil, (0, text_height))
    grid.paste(mask_pil, (img_w, text_height))
    for j, gen_img in enumerate(generated_imgs):
        grid.paste(gen_img, ((2 + j) * img_w, text_height))

    grid.save(sample_dir / f"sample_{idx}_concat.png")
    orig_pil.save(sample_dir / f"sample_{idx}_input.png")
    mask_pil.save(sample_dir / f"sample_{idx}_control.png")
    for j, gen_img in enumerate(generated_imgs):
        gen_img.save(sample_dir / f"sample_{idx}_output_{j}.png")


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_samples(
    model: DiffusionControlNet,
    pipeline,
    val_loader,
    output_dir: Path,
    num_samples: int = None,
    num_images_per_prompt: int = None,
    step: int = 0,
    epoch: int = 0,
    run_id: str = "",
) -> None:
    """
    Generate sample images from the current model weights and save grids.

    Pulls `num_samples` batches from val_loader, generates `num_images_per_prompt`
    variations per prompt, and saves side-by-side grids to output_dir/samples/.

    Args:
        model: DiffusionControlNet (used only for device — pipeline holds the weights)
        pipeline: StableDiffusionControlNetPipeline (already built with current weights)
        val_loader: Validation DataLoader to pull conditioning inputs from
        output_dir: Root output directory; grids go into output_dir/samples/<subdir>/
        num_samples: How many val images to visualize (default: config.NUM_SAMPLES_TO_GENERATE)
        num_images_per_prompt: Variations per image (default: config.NUM_IMAGES_PER_PROMPT)
        step: Current global step (for filename/caption)
        epoch: Current epoch (for filename)
        run_id: Run identifier (for filename)
    """
    num_samples = num_samples if num_samples is not None else config.NUM_SAMPLES_TO_GENERATE
    num_images_per_prompt = num_images_per_prompt if num_images_per_prompt is not None else config.NUM_IMAGES_PER_PROMPT

    if num_samples == 0:
        print("Sample generation disabled (NUM_SAMPLES_TO_GENERATE=0)")
        return

    device = next(model.parameters()).device

    # Collect num_samples batches from val loader
    all_batches = []
    for i, batch in enumerate(val_loader):
        all_batches.append(batch)
        if i + 1 >= num_samples:
            break

    prompts = [b["prompt"][0] for b in all_batches]
    mask_rgb = torch.cat([b["mask_rgb"][:1] for b in all_batches])
    original_images = torch.cat([b["image"][:1] for b in all_batches])

    generator = torch.Generator(device=device).manual_seed(42)

    all_generated = []
    for i in range(len(prompts)):
        control_pil = mask_rgb_to_pil(mask_rgb[i], config.RESOLUTION)
        generated = pipeline(
            prompt=prompts[i],
            image=control_pil,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            num_inference_steps=config.NUM_INFERENCE_STEPS,
            guidance_scale=config.GUIDANCE_SCALE,
            controlnet_conditioning_scale=config.CONTROLNET_CONDITIONING_SCALE,
        ).images
        all_generated.append(generated)

    # Save grids
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"job{run_id}_epoch{epoch}_step{step:06d}_{timestamp}" if run_id else f"epoch{epoch}_step{step:06d}_{timestamp}"
    sample_dir = Path(output_dir) / "samples" / tag
    sample_dir.mkdir(parents=True, exist_ok=True)

    for i, prompt in enumerate(prompts):
        orig_pil = Image.fromarray(
            ((original_images[i].cpu() * 0.5 + 0.5).clamp(0, 1)
             .numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        )
        mask_pil = mask_rgb_to_pil(mask_rgb[i], config.RESOLUTION)
        caption = f"Step {step} | {prompt}"
        save_sample_grid(sample_dir, i, caption, orig_pil, mask_pil, all_generated[i])

    print(f"Saved {len(prompts)} sample grids to {sample_dir}")


# ---------------------------------------------------------------------------
# Standalone CLI entry point
# ---------------------------------------------------------------------------

def _run_standalone():
    """
    Load a checkpoint and generate samples without starting a full training run.
    Usage: python -m src.sampling --checkpoint path/to/model.pt --output_dir path/to/out
    """
    import argparse
    parser = argparse.ArgumentParser(description="Generate samples from a trained checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--output_dir", default="outputs/standalone_samples")
    parser.add_argument("--num_samples", type=int, default=config.NUM_SAMPLES_TO_GENERATE)
    parser.add_argument("--num_images_per_prompt", type=int, default=config.NUM_IMAGES_PER_PROMPT)
    parser.add_argument("--num_inference_steps", type=int, default=config.NUM_INFERENCE_STEPS)
    parser.add_argument("--guidance_scale", type=float, default=config.GUIDANCE_SCALE)
    parser.add_argument("--control_strength", type=float, default=config.CONTROLNET_CONDITIONING_SCALE)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")

    model = DiffusionControlNet(pretrained=False, device=device)
    model.load_checkpoint(args.checkpoint)
    model.eval()

    pipeline = create_pipeline(model, device=device)
    pipeline.set_progress_bar_config(disable=False)

    val_dataset = create_val_dataset(resolution=config.RESOLUTION)
    val_loader = create_dataloader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # Temporarily override inference settings from CLI args
    orig_steps = config.NUM_INFERENCE_STEPS
    orig_guidance = config.GUIDANCE_SCALE
    orig_control = config.CONTROLNET_CONDITIONING_SCALE
    config.NUM_INFERENCE_STEPS = args.num_inference_steps
    config.GUIDANCE_SCALE = args.guidance_scale
    config.CONTROLNET_CONDITIONING_SCALE = args.control_strength

    generate_samples(
        model=model,
        pipeline=pipeline,
        val_loader=val_loader,
        output_dir=Path(args.output_dir),
        num_samples=args.num_samples,
        num_images_per_prompt=args.num_images_per_prompt,
    )

    # Restore
    config.NUM_INFERENCE_STEPS = orig_steps
    config.GUIDANCE_SCALE = orig_guidance
    config.CONTROLNET_CONDITIONING_SCALE = orig_control


if __name__ == "__main__":
    _run_standalone()
