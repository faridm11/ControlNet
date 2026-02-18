"""
Standalone evaluation script for ControlNet checkpoints.
Computes mIoU, FID, and optionally CLIP score across different control strengths.

Usage:
    python evaluate.py --checkpoint outputs/checkpoints/best_miou_model.pt
    python evaluate.py --checkpoint outputs/checkpoints/best_miou_model.pt --control-strengths 0.5 1.0 1.5 2.0
    python evaluate.py --checkpoint outputs/checkpoints/best_miou_model.pt --segmentor /path/to/seg_model.pt
"""

import argparse
import torch
from pathlib import Path
import pandas as pd
from datetime import datetime

from src.models.diffusion_controlnet import DiffusionControlNet, create_pipeline
from src.data import create_val_dataset, create_dataloader
from src.evaluation import ControlNetEvaluator
from src import config


def load_checkpoint(checkpoint_path: str, device='cuda'):
    """Load ControlNet model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Initialize model
    model = DiffusionControlNet(
        pretrained_model_name=config.PRETRAINED_MODEL_NAME,
        controlnet_conditioning_scale=config.CONTROLNET_CONDITIONING_SCALE,
        device=device
    )
    
    # Load state dict
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_unet' in checkpoint:
        model.unet.load_state_dict(checkpoint['model_unet'])
    if 'model_controlnet' in checkpoint:
        model.controlnet.load_state_dict(checkpoint['model_controlnet'])
    
    print(f"✓ Loaded checkpoint from step {checkpoint.get('global_step', 'unknown')}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Best val loss: {checkpoint.get('best_val_loss', 'unknown')}")
    print(f"  Best mIoU: {checkpoint.get('best_miou', 'unknown')}")
    print(f"  Best FID: {checkpoint.get('best_fid', 'unknown')}")
    
    return model


def evaluate_checkpoint(
    checkpoint_path: str,
    control_strengths=[0.5, 1.0, 1.5],
    num_inference_steps: int = 50,
    segmentor_checkpoint: str = None,
    device: str = 'cuda',
):
    """
    Evaluate a checkpoint with multiple control strengths.
    
    Args:
        checkpoint_path: Path to checkpoint file
        control_strengths: List of control strength values to sweep
        num_inference_steps: Number of DDIM steps for generation
        segmentor_checkpoint: Path to segmentation model for mIoU computation
        device: Device to use
        
    Returns:
        DataFrame with results for each control strength
    """
    # Load model
    model = load_checkpoint(checkpoint_path, device=device)
    model.eval()
    
    # Create validation dataset
    print("\nCreating validation dataset...")
    val_dataset = create_val_dataset(resolution=config.RESOLUTION)
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    print(f"✓ Loaded {len(val_dataset)} validation samples")
    
    # Create evaluator
    evaluator = ControlNetEvaluator(device=device)
    
    # Load segmentor if provided
    if segmentor_checkpoint:
        print(f"\nLoading segmentor from {segmentor_checkpoint}")
        evaluator.load_segmentor(segmentor_checkpoint)
    else:
        print("\n⚠️ No segmentor provided, mIoU will be 0.0")
        print("   Provide --segmentor argument to compute mIoU")
    
    # Create pipeline
    print("\nCreating inference pipeline...")
    pipeline = create_pipeline(model, device=device)
    pipeline.set_progress_bar_config(disable=False)
    
    # Evaluate across control strengths
    results = []
    
    for strength in control_strengths:
        print(f"\n{'='*70}")
        print(f"EVALUATING CONTROL STRENGTH: {strength}")
        print(f"{'='*70}")
        
        # Evaluate mIoU
        print("\nComputing mIoU...")
        miou_metrics = evaluator.evaluate_miou(
            model,
            val_loader,
            pipeline,
            control_strength=strength,
            num_inference_steps=num_inference_steps,
        )
        
        # Evaluate FID
        print("\nComputing FID...")
        fid = evaluator.evaluate_fid(
            model,
            val_loader,
            pipeline,
            control_strength=strength,
            num_inference_steps=num_inference_steps,
        )
        
        result = {
            'control_strength': strength,
            'miou': miou_metrics.get('miou', 0.0),
            'pixel_acc': miou_metrics.get('pixel_acc', 0.0),
            'fid': fid,
        }
        results.append(result)
        
        print(f"\nResults for control strength {strength}:")
        print(f"  mIoU: {result['miou']:.4f}")
        print(f"  Pixel Acc: {result['pixel_acc']:.4f}")
        print(f"  FID: {result['fid']:.2f}")
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ControlNet checkpoint with mIoU and FID metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation with default control strengths (0.5, 1.0, 1.5)
  python evaluate.py --checkpoint outputs/checkpoints/best_miou_model.pt
  
  # Custom control strengths
  python evaluate.py --checkpoint outputs/checkpoints/best_miou_model.pt --control-strengths 0.5 1.0 1.5 2.0
  
  # With segmentor for mIoU computation
  python evaluate.py --checkpoint outputs/checkpoints/best_miou_model.pt --segmentor /path/to/segmentation_model.pt
  
  # More inference steps for better quality
  python evaluate.py --checkpoint outputs/checkpoints/best_miou_model.pt --num-inference-steps 100
        """
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint file (required)'
    )
    parser.add_argument(
        '--control-strengths',
        nargs='+',
        type=float,
        default=[0.5, 1.0, 1.5],
        help='Control strength values to evaluate (default: 0.5 1.0 1.5)'
    )
    parser.add_argument(
        '--num-inference-steps',
        type=int,
        default=50,
        help='Number of DDIM steps (default: 50, higher = better quality but slower)'
    )
    parser.add_argument(
        '--segmentor',
        type=str,
        default=None,
        help='Path to segmentation model checkpoint for mIoU computation (optional)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (default: cuda)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV path (default: auto-generated in logs/)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("CONTROLNET CHECKPOINT EVALUATION")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Control strengths: {args.control_strengths}")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"Device: {args.device}")
    if args.segmentor:
        print(f"Segmentor: {args.segmentor}")
    print("=" * 70)
    
    # Run evaluation
    results_df = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        control_strengths=args.control_strengths,
        num_inference_steps=args.num_inference_steps,
        segmentor_checkpoint=args.segmentor,
        device=args.device,
    )
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = Path(args.checkpoint).stem
        output_path = config.LOG_DIR / f"eval_{checkpoint_name}_{timestamp}.csv"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    # Print summary
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_path}")
    print("\n📊 Summary Table:")
    print(results_df.to_string(index=False))
    
    # Find best configurations
    best_miou_row = results_df.loc[results_df['miou'].idxmax()]
    best_fid_row = results_df.loc[results_df['fid'].idxmin()]
    
    print(f"\n🏆 Best mIoU: {best_miou_row['miou']:.4f} at control strength {best_miou_row['control_strength']:.1f}")
    print(f"🏆 Best FID: {best_fid_row['fid']:.2f} at control strength {best_fid_row['control_strength']:.1f}")
    
    print(f"\n💡 Recommendation:")
    print(f"   - For structural adherence: Use control strength {best_miou_row['control_strength']:.1f}")
    print(f"   - For realism: Use control strength {best_fid_row['control_strength']:.1f}")
    
    print(f"\n{'='*70}")


if __name__ == '__main__':
    main()
