"""
Overfit a tiny subset (default 8 samples) to debug training quality.
Run from repo root:
  python -m src.testing.overfit_8_samples --num-samples 8 --epochs 200
"""

import argparse
from pathlib import Path
import torch
from torch.utils.data import Subset

from src.data import create_train_dataset, create_dataloader
from src.models.diffusion_controlnet import DiffusionControlNet
from src.train import Trainer
from src import config


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfit a tiny subset")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=config.RESOLUTION)
    args = parser.parse_args()

    # Override config for a focused overfit run
    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.GRADIENT_ACCUMULATION_STEPS = args.accum_steps
    config.LOGGING_STEPS = 5
    config.NUM_SAMPLES_TO_GENERATE = 4
    config.NUM_IMAGES_PER_PROMPT = 4
    config.USE_WANDB = False

    # UNet LoRA enabled for overfit — frozen UNet can't memorize images
    config.USE_UNET_LORA = True
    config.UNET_LORA_RANK = 16
    config.UNET_LORA_ALPHA = 16

    # Higher LR for aggressive memorization (10x main training LR)
    config.CONTROLNET_LORA_LR = 1e-4
    config.UNET_LR = 1e-6

    # No dropout — we want to memorize, not generalize
    config.CONTROLNET_LORA_DROPOUT = 0.0
    config.UNET_LORA_DROPOUT = 0.0

    # No text dropout — fix the conditioning, don't randomly null it
    config.TEXT_DROPOUT_PROB = 0.0

    # Do NOT fix seed per epoch — fixed timestep breaks DDIM inference chain
    config.OVERFIT_FIXED_SEED = False

    # Route outputs to test/outputs
    test_output_root = Path(config.PROJECT_ROOT) / "test" / "outputs"
    config.OUTPUT_DIR = test_output_root
    config.CHECKPOINT_DIR = test_output_root / "checkpoints"
    config.LOG_DIR = test_output_root / "logs"
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Fixed seed — deterministic noise/timesteps for reproducible analysis
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = create_train_dataset(resolution=args.resolution)
    num_samples = min(args.num_samples, len(dataset))
    subset = Subset(dataset, list(range(num_samples)))

    train_loader = create_dataloader(
        subset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    # Use the same subset as val to see if it can memorize
    val_loader = create_dataloader(
        subset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    model = DiffusionControlNet(pretrained=True, device=device)
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, device=device)
    trainer.train()


if __name__ == "__main__":
    main()
