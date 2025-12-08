"""
Basic training loop for ControlNet fine-tuning.
Phase 1: Diffusion loss only (no segmentor).
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
from datetime import datetime

from .models.diffusion_controlnet import DiffusionControlNet, create_pipeline
from .data import create_train_dataset, create_val_dataset, create_dataloader
from . import config


class Trainer:
    """
    ControlNet trainer with diffusion loss.
    """
    
    def __init__(
        self,
        model: DiffusionControlNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = None,
        num_epochs: int = None,
        gradient_accumulation_steps: int = None,
        mixed_precision: str = None,
        save_steps: int = None,
        validation_steps: int = None,
        output_dir: str = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: DiffusionControlNet model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            learning_rate: Learning rate (default: from config)
            num_epochs: Number of epochs (default: from config)
            gradient_accumulation_steps: Gradient accumulation (default: from config)
            mixed_precision: Mixed precision mode (default: from config)
            save_steps: Save checkpoint every N steps (default: from config)
            validation_steps: Run validation every N steps (default: from config)
            output_dir: Output directory (default: from config)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Hyperparameters (use config defaults if not provided)
        self.learning_rate = learning_rate or config.LEARNING_RATE
        self.num_epochs = num_epochs or config.NUM_EPOCHS
        self.gradient_accumulation_steps = gradient_accumulation_steps or config.GRADIENT_ACCUMULATION_STEPS
        self.mixed_precision = mixed_precision or config.MIXED_PRECISION
        self.save_steps = save_steps or config.SAVE_STEPS
        self.validation_steps = validation_steps or config.VALIDATION_STEPS
        self.output_dir = Path(output_dir) if output_dir else config.OUTPUT_DIR
        
        # Create output directories
        self.samples_dir = self.output_dir / "samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup learning rate scheduler
        self.lr_scheduler = self._create_scheduler()
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision == "fp16" else None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        print("=" * 70)
        print("TRAINER INITIALIZED")
        print("=" * 70)
        print(f"Learning rate: {self.learning_rate}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"Mixed precision: {self.mixed_precision}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 70)
    
    def _create_optimizer(self):
        """Create optimizer from config."""
        trainable_params = self.model.get_trainable_parameters()
        
        if config.OPTIMIZER.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.learning_rate,
                betas=(config.ADAM_BETA1, config.ADAM_BETA2),
                weight_decay=config.WEIGHT_DECAY,
                eps=config.ADAM_EPSILON
            )
        elif config.OPTIMIZER.lower() == "adam":
            optimizer = torch.optim.Adam(
                trainable_params,
                lr=self.learning_rate,
                betas=(config.ADAM_BETA1, config.ADAM_BETA2),
                eps=config.ADAM_EPSILON
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.OPTIMIZER}")
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        total_steps = len(self.train_loader) * self.num_epochs // self.gradient_accumulation_steps
        
        if config.LR_SCHEDULER == "constant_with_warmup":
            from transformers import get_constant_schedule_with_warmup
            scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=config.LR_WARMUP_STEPS
            )
        elif config.LR_SCHEDULER == "cosine":
            from transformers import get_cosine_schedule_with_warmup
            scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=config.LR_WARMUP_STEPS,
                num_training_steps=total_steps
            )
        elif config.LR_SCHEDULER == "constant":
            from transformers import get_constant_schedule
            scheduler = get_constant_schedule(self.optimizer)
        else:
            raise ValueError(f"Unknown scheduler: {config.LR_SCHEDULER}")
        
        return scheduler
    
    def train_step(self, batch):
        """
        Single training step.
        
        Args:
            batch: Batch from dataloader
            
        Returns:
            Loss value
        """
        images = batch['image'].to(self.model.device)
        masks = batch['mask'].to(self.model.device)
        prompts = batch['prompt']
        
        # Forward pass with mixed precision
        if self.mixed_precision == "fp16":
            with torch.cuda.amp.autocast():
                outputs = self.model(images, masks, prompts)
                loss = outputs['loss']
        else:
            outputs = self.model(images, masks, prompts)
            loss = outputs['loss']
        
        # Scale loss for gradient accumulation
        loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item() * self.gradient_accumulation_steps
    
    def optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        if self.scaler is not None:
            # Unscale gradients for clipping
            self.scaler.unscale_(self.optimizer)
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.get_trainable_parameters(),
                config.MAX_GRAD_NORM
            )
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.get_trainable_parameters(),
                config.MAX_GRAD_NORM
            )
            # Optimizer step
            self.optimizer.step()
        
        # Scheduler step
        self.lr_scheduler.step()
        # Zero gradients
        self.optimizer.zero_grad()
    
    @torch.no_grad()
    def validate(self):
        """Run validation."""
        self.model.eval()
        
        val_losses = []
        for batch in tqdm(self.val_loader, desc="Validation"):
            images = batch['image'].to(self.model.device)
            masks = batch['mask'].to(self.model.device)
            prompts = batch['prompt']
            
            outputs = self.model(images, masks, prompts)
            val_losses.append(outputs['loss'].item())
        
        avg_val_loss = np.mean(val_losses)
        
        self.model.train()
        return avg_val_loss
    
    @torch.no_grad()
    def generate_samples(self, num_samples: int = 4):
        """
        Generate sample images for visualization.
        
        Args:
            num_samples: Number of samples to generate
        """
        self.model.eval()
        
        # Get validation batch
        batch = next(iter(self.val_loader))
        masks = batch['mask'][:num_samples].to(self.model.device)
        prompts = batch['prompt'][:num_samples]
        
        # Create pipeline for generation
        pipeline = create_pipeline(self.model, device=self.model.device)
        pipeline.set_progress_bar_config(disable=True)
        
        # Prepare mask conditioning
        mask_rgb = batch['mask_rgb'][:num_samples]
        
        # Generate images
        generated = pipeline(
            prompt=prompts,
            image=mask_rgb,
            num_inference_steps=config.NUM_INFERENCE_STEPS,
            guidance_scale=config.GUIDANCE_SCALE,
            controlnet_conditioning_scale=config.CONTROLNET_CONDITIONING_SCALE,
        ).images
        
        # Save samples
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_dir = self.samples_dir / f"step_{self.global_step:06d}_{timestamp}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        for i, (img, prompt) in enumerate(zip(generated, prompts)):
            # Save generated image
            img.save(sample_dir / f"sample_{i}.png")
            
            # Save prompt
            with open(sample_dir / f"sample_{i}.txt", 'w') as f:
                f.write(prompt)
            
            # Save mask visualization
            mask_viz = (mask_rgb[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            Image.fromarray(mask_viz).save(sample_dir / f"mask_{i}.png")
        
        print(f"✓ Saved {num_samples} samples to {sample_dir}")
        
        self.model.train()
    
    def save_checkpoint(self, is_best: bool = False):
        """
        Save training checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint_path = config.CHECKPOINT_DIR / f"checkpoint_step_{self.global_step:06d}.pt"
        
        checkpoint = {
            'model_unet': self.model.unet.state_dict(),
            'model_controlnet': self.model.controlnet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'scaler': self.scaler.state_dict() if self.scaler else None,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = config.CHECKPOINT_DIR / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved: {best_path}")
        
        # Keep only last N checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = sorted(config.CHECKPOINT_DIR.glob("checkpoint_step_*.pt"))
        
        if len(checkpoints) > config.SAVE_TOTAL_LIMIT:
            for ckpt in checkpoints[:-config.SAVE_TOTAL_LIMIT]:
                ckpt.unlink()
                print(f"Removed old checkpoint: {ckpt.name}")
    
    def train(self):
        """Main training loop."""
        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        
        self.model.train()
        
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-" * 70)
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")
            
            for step, batch in enumerate(progress_bar):
                # Training step
                loss = self.train_step(batch)
                
                # Gradient accumulation
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer_step()
                    self.global_step += 1
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{loss:.4f}",
                        'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
                        'step': self.global_step
                    })
                    
                    # Validation
                    if self.global_step % self.validation_steps == 0:
                        val_loss = self.validate()
                        print(f"\nValidation loss: {val_loss:.4f}")
                        
                        # Check if best model
                        is_best = val_loss < self.best_val_loss
                        if is_best:
                            self.best_val_loss = val_loss
                            print(f"✓ New best validation loss: {val_loss:.4f}")
                        
                        # Generate samples
                        self.generate_samples()
                        
                        # Save checkpoint
                        self.save_checkpoint(is_best=is_best)
                    
                    # Save checkpoint
                    elif self.global_step % self.save_steps == 0:
                        self.save_checkpoint()
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Total steps: {self.global_step}")


def main():
    """Main training script."""
    print("=" * 70)
    print("CONTROLNET TRAINING - PHASE 1: DIFFUSION ONLY")
    print("=" * 70)
    
    # Print configuration
    config.print_config()
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = create_train_dataset(resolution=config.RESOLUTION)
    val_dataset = create_val_dataset(resolution=config.RESOLUTION)
    
    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Initialize model
    print("\nInitializing model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if config.DTYPE == "fp16" else torch.float32
    
    model = DiffusionControlNet(
        pretrained=True,
        device=device,
        dtype=dtype
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
