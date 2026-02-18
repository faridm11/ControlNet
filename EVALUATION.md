# Multi-Metric Evaluation System

## Overview

The training pipeline now tracks **three separate validation metrics**:

1. **mIoU (Mean Intersection over Union)** - PRIMARY metric for model selection
   - Measures structural adherence to control masks
   - Higher is better (0.0 to 1.0)
   - Best model saved as `best_miou_model.pt`

2. **FID (Fréchet Inception Distance)** - Secondary metric for realism
   - Measures photorealism of generated images
   - Lower is better
   - Best model saved as `best_fid_model.pt`

3. **Diffusion Loss** - For monitoring only
   - Traditional validation loss
   - Not used for model selection
   - Best saved as `best_model_overall.pt` for backward compatibility

## Evaluation Frequency

- **Quick validation** (diffusion loss only): Every `VALIDATION_STEPS` steps
- **Full evaluation** (mIoU + FID + Control Strength Sweep): Every 5 validation steps OR at end of each epoch
- **Control strength sweep**: Tests [0.5, 1.0, 1.5] to find optimal control-realism tradeoff
- Rationale: Full evaluation is expensive, so we run it less frequently while still monitoring training progress with diffusion loss

## Control Strength Sweep

During full evaluation, the system automatically tests multiple control strengths:
- **0.5**: Lower adherence, more creative/realistic
- **1.0**: Balanced (default training value)
- **1.5**: Higher adherence, more faithful to mask

The best mIoU and FID across all strengths are tracked separately, allowing you to see which strength works best for each metric.

## CSV Logging

All metrics are logged to `logs/metrics_{timestamp}.csv`:

```csv
step,epoch,train_loss,val_loss,miou,fid,control_strength,lr
1000,0,0.045231,0.043122,0.6234,52.34,0.5,0.00010000
1000,0,0.045231,0.043122,0.6543,45.23,1.0,0.00010000
1000,0,0.045231,0.043122,0.6789,42.15,1.5,0.00010000
2000,0,0.042156,0.041234,,,1.0,0.00010000
3000,0,0.040234,0.039876,,,1.0,0.00010000
4000,0,0.038912,0.038456,,,1.0,0.00010000
5000,1,0.037654,0.037123,0.6456,50.12,0.5,0.00010000
5000,1,0.037654,0.037123,0.6789,44.56,1.0,0.00010000
5000,1,0.037654,0.037123,0.6912,41.23,1.5,0.00010000
```

Note: 
- mIoU and FID columns are empty when only quick validation is run
- During full evaluation, multiple rows are logged (one per control strength tested)
- This allows you to track how different control strengths perform over training

## Model Selection Strategy

**Primary: mIoU** - Choose the model with highest mIoU for deployment
- This ensures the model follows the control masks accurately
- Critical for sidewalk segmentation where spatial correctness matters

**Secondary: FID** - Use for tie-breaking or when realism is prioritized
- Lower FID = more photorealistic images
- Useful when multiple models have similar mIoU

**Monitoring: Diffusion Loss** - Track for training health
- Helps identify overfitting or training instability
- Not used for final model selection

## Saved Checkpoints

The training system saves multiple checkpoints:

```
outputs/checkpoints/
├── best_miou_model.pt           # Best by mIoU (PRIMARY - use this!)
├── best_fid_model.pt            # Best by FID (secondary)
├── best_model_overall.pt        # Best by val_loss (legacy)
├── best_run_{timestamp}.pt      # Best for current training run
└── run_{timestamp}_step_*.pt    # Regular checkpoints (last N kept)
```

## Standalone Evaluation Script

For post-training evaluation with control strength sweep:

```bash
python evaluate.py \
    --checkpoint outputs/checkpoints/best_miou_model.pt \
    --control-strengths 0.5 1.0 1.5 \
    --num-inference-steps 50 \
    --segmentor /path/to/segmentation_model.pt
```

### Arguments:
- `--checkpoint`: Path to checkpoint file (required)
- `--control-strengths`: List of strengths to test (default: 0.5 1.0 1.5)
- `--num-inference-steps`: DDIM steps (default: 50)
- `--segmentor`: Path to segmentation model for mIoU (optional)
- `--compute-clip`: Compute CLIP score (not implemented yet)
- `--output`: CSV output path (default: auto-generated)

### Example Output:

```
control_strength    miou    pixel_acc    fid
0.5                0.6234  0.8456       52.34
1.0                0.6789  0.8623       42.15
1.5                0.6912  0.8701       38.92
```

## Integration with Segmentor

Currently, the evaluator expects a segmentation model to compute mIoU. To integrate your segmentation model:

1. Update `ControlNetEvaluator.load_segmentor()` in `src/evaluation.py`
2. Load your model from `sensation-sidewalk-segmentation` or `sensation-models`
3. The segmentor should:
   - Accept RGB images as input (B, 3, H, W) in [0, 1]
   - Return class predictions (B, H, W) with integer class IDs
   - Support 8-class segmentation matching the PALETTE

Example integration:

```python
def load_segmentor(self, checkpoint_path: str):
    """Load pretrained segmentation model."""
    from sensation.segmentation import load_model  # Your loader
    
    self.segmentor = load_model(checkpoint_path)
    self.segmentor.to(self.device)
    self.segmentor.eval()
    
    print(f"✓ Segmentor loaded from {checkpoint_path}")
```

## Requirements

For FID computation, install torchmetrics:

```bash
pip install torchmetrics
```

If not installed, FID computation will be skipped (returns inf).

## Migration from Old System

If you have existing checkpoints without `best_miou` and `best_fid` fields:

1. They will load fine but show "unknown" for these metrics
2. Run `evaluate.py` on old checkpoints to compute metrics retroactively
3. The evaluation script works with any checkpoint format

## Tips for Best Results

1. **Use mIoU model for production**: `best_miou_model.pt` gives best structural adherence
2. **Monitor both metrics**: Sometimes mIoU and FID improvements don't align
3. **Adjust control strength**: Try different values (0.5-1.5) at inference time
4. **Check CSV logs**: Plot metrics over time to understand training dynamics
5. **Validate on separate test set**: The evaluation script can run on any dataset

## Implementation Details

- **mIoU computation**: Generate images → Run segmentor → Compare with ground truth
- **FID computation**: Uses InceptionV3 features, compares generated vs real distributions
- **Control strength sweep**: Tests multiple conditioning scales to find optimal setting
- **Batched processing**: Efficient evaluation on full validation set
- **Deterministic sampling**: Uses fixed seed (42) for reproducible samples
