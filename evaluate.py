"""
Standalone evaluation of a ControlNet LoRA checkpoint — or of the untouched
pretrained ControlNet when no checkpoint is given (the zero-shot baseline).

Computes FID, KID, LPIPS and CLIP scores on the validation set, once per
control strength, with the same protocol the trainer uses.

Usage:
    python evaluate.py                                        # pretrained baseline
    python evaluate.py --checkpoint outputs/checkpoints/<run_id>_LoRA.pt
    python evaluate.py --checkpoint <...>_LoRA.pt --control-strengths 0.5 1.0 1.5
    python evaluate.py --seed 43 --tag baseline_seed43
"""

import argparse
import json
import torch
from pathlib import Path
import pandas as pd
from datetime import datetime

from src.models.diffusion_controlnet import DiffusionControlNet, create_pipeline
from src.data import create_val_dataset, create_dataloader
from src.evaluation import ControlNetEvaluator
from src import config


def evaluate_checkpoint(
    checkpoint_path: str = None,
    control_strengths=(1.0,),
    num_inference_steps: int = config.NUM_INFERENCE_STEPS,
    device: str = 'cuda',
):
    """
    Evaluate a LoRA checkpoint (or the pretrained baseline if checkpoint_path is None).

    Returns:
        DataFrame with one row of metrics per control strength
    """
    model = DiffusionControlNet(pretrained=True, device=device)
    if checkpoint_path:
        model.load_checkpoint(checkpoint_path)
    else:
        print("No checkpoint given — evaluating the pretrained ControlNet (LoRA is a no-op at init)")
    model.eval()

    val_dataset = create_val_dataset(resolution=config.RESOLUTION)
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    print(f"Loaded {len(val_dataset)} validation samples")

    evaluator = ControlNetEvaluator(device=device)

    # merge_and_unload is destructive on the model, which is fine here — we only evaluate
    pipeline = create_pipeline(model, device=device)
    pipeline.set_progress_bar_config(disable=True)

    results = []
    for strength in control_strengths:
        print(f"\nControl strength: {strength}")
        metrics = evaluator.evaluate(
            model, val_loader, pipeline,
            control_strength=strength,
            num_inference_steps=num_inference_steps,
        )
        row = {'control_strength': strength, **metrics}
        results.append(row)
        print("  " + " | ".join(f"{k}={v:.5f}" for k, v in metrics.items() if isinstance(v, float)))

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a ControlNet LoRA checkpoint or the pretrained baseline")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to <run_id>_LoRA.pt (omit to evaluate the pretrained baseline)')
    parser.add_argument('--control-strengths', nargs='+', type=float,
                        default=[config.CONTROLNET_CONDITIONING_SCALE],
                        help='Control strength values to evaluate')
    parser.add_argument('--num-inference-steps', type=int, default=config.NUM_INFERENCE_STEPS,
                        help='Number of DDIM steps')
    parser.add_argument('--seed', type=int, default=config.EVAL_SEED,
                        help='Base generation seed (batch i uses seed + i)')
    parser.add_argument('--tag', type=str, default=None,
                        help='Name for the result files (default: checkpoint stem or "baseline")')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Where to write results (default: config.LOG_DIR)')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    config.EVAL_SEED = args.seed
    tag = args.tag or (Path(args.checkpoint).stem if args.checkpoint else "baseline")

    print("=" * 70)
    print("CONTROLNET EVALUATION")
    print("=" * 70)
    print(f"Checkpoint:        {args.checkpoint or 'none (pretrained baseline)'}")
    print(f"Class version:     {config.CLASS_VERSION}")
    print(f"Prompts:           {config.VAL_PROMPTS_FILE}")
    print(f"Control strengths: {args.control_strengths}")
    print(f"Inference steps:   {args.num_inference_steps}")
    print(f"Guidance scale:    {config.GUIDANCE_SCALE}")
    print(f"Seed:              {args.seed}")
    print("=" * 70)

    results_df = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        control_strengths=args.control_strengths,
        num_inference_steps=args.num_inference_steps,
        device=args.device,
    )

    out_dir = Path(args.output_dir) if args.output_dir else config.LOG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = out_dir / f"eval_{tag}_{timestamp}.json"
    with open(out_json, 'w') as f:
        json.dump({
            'tag': tag,
            'checkpoint': args.checkpoint,
            'class_version': config.CLASS_VERSION,
            'prompts_file': str(config.VAL_PROMPTS_FILE),
            'seed': args.seed,
            'num_inference_steps': args.num_inference_steps,
            'guidance_scale': config.GUIDANCE_SCALE,
            'negative_prompt': config.NEGATIVE_PROMPT,
            'results': results_df.to_dict(orient='records'),
        }, f, indent=2)

    print(f"\nResults saved to: {out_json}\n")
    print(results_df.to_string(index=False))


if __name__ == '__main__':
    main()
