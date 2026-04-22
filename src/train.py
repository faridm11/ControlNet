"""
Basic training loop for ControlNet fine-tuning.
Phase 1: Diffusion loss only (no segmentor).

No Accelerate — pure PyTorch with manual AMP + gradient accumulation
so every step is transparent and debuggable.
"""

import gc
import os
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from datetime import datetime
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule,
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .models.diffusion_controlnet import DiffusionControlNet, create_pipeline
from .data import create_train_dataset, create_val_dataset, create_dataloader
from .evaluation import ControlNetEvaluator
from .plotting import plot_training_curves
from .sampling import generate_samples
from . import config


class Trainer:
    """ControlNet trainer — pure PyTorch, no Accelerate."""

    def __init__(
        self,
        model: DiffusionControlNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        num_epochs: int = None,
        output_dir: str = None,
    ):
        self.device = device
        self.num_epochs = num_epochs or config.NUM_EPOCHS
        self.output_dir = Path(output_dir) if output_dir else config.OUTPUT_DIR
        self.use_amp = config.MIXED_PRECISION in ("fp16", "bf16")
        self.amp_dtype = torch.bfloat16 if config.MIXED_PRECISION == "bf16" else torch.float16

        # Directories
        self.samples_dir = self.output_dir / "samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)

        # Model already on device (moved in main())
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimizer + scheduler
        self.optimizer = self._create_optimizer()
        self.lr_scheduler = self._create_scheduler()

        # GradScaler only needed for fp16 (bf16 has sufficient dynamic range without scaling)
        self.scaler = torch.amp.GradScaler('cuda', enabled=(config.MIXED_PRECISION == "fp16"))

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        # Run ID
        self.run_id = os.environ.get('SLURM_JOB_ID') or datetime.now().strftime("%Y%m%d_%H%M%S")

        # CSV logging
        self.metrics_log_path = config.LOG_DIR / f"metrics_{self.run_id}.csv"
        with open(self.metrics_log_path, 'w') as f:
            f.write("step,epoch,train_loss,val_loss,lr,grad_norm\n")

        # Evaluator
        self.evaluator = ControlNetEvaluator(device=self.device)
        self.pipeline = None

        # WandB
        if config.USE_WANDB and WANDB_AVAILABLE:
            wandb.init(
                project=config.WANDB_PROJECT,
                entity=config.WANDB_ENTITY,
                name=str(self.run_id),
                config=config.get_config_dict(),
                resume="allow",
            )
            print("WandB initialized")
        elif config.USE_WANDB and not WANDB_AVAILABLE:
            print("WARNING: USE_WANDB=True but wandb not installed")

        print("=" * 70)
        print("TRAINER INITIALIZED (pure PyTorch — no Accelerate)")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"AMP ({config.MIXED_PRECISION}): {self.use_amp}")
        print(f"ControlNet LoRA LR: {config.CONTROLNET_LORA_LR}")
        if config.USE_UNET_LORA:
            print(f"UNet LoRA LR: {config.UNET_LR}")
        else:
            print(f"UNet: FROZEN (no LoRA)")
        print(f"Gradient accumulation: {config.GRADIENT_ACCUMULATION_STEPS}")
        print(f"Effective batch size: {config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS}")
        print(f"Epochs: {self.num_epochs}")
        print(f"WandB: {'enabled' if config.USE_WANDB and WANDB_AVAILABLE else 'disabled'}")
        print("=" * 70)

    def _create_optimizer(self):
        """Create optimizer with per-group LR and weight decay."""
        param_groups = []

        controlnet_params = [p for p in self.model.controlnet.parameters() if p.requires_grad]
        if controlnet_params:
            param_groups.append({
                "params": controlnet_params,
                "lr": config.CONTROLNET_LORA_LR,
                "weight_decay": config.WEIGHT_DECAY,
                "name": "controlnet_lora",
            })

        unet_params = [p for p in self.model.unet.parameters() if p.requires_grad]
        if unet_params:
            param_groups.append({
                "params": unet_params,
                "lr": config.UNET_LR,
                "weight_decay": config.UNET_WEIGHT_DECAY,
                "name": "unet_lora",
            })

        # Flat list for gradient clipping
        self.trainable_params = []
        for group in param_groups:
            self.trainable_params.extend(group["params"])

        total = sum(p.numel() for p in self.trainable_params)
        cn_count = sum(p.numel() for p in controlnet_params)
        unet_count = sum(p.numel() for p in unet_params) if unet_params else 0

        print(f"\nTrainable parameters: {total:,}")
        print(f"  ControlNet LoRA: {cn_count:,}  (LR={config.CONTROLNET_LORA_LR}, WD={config.WEIGHT_DECAY})")
        print(f"  UNet LoRA:       {unet_count:,}  (LR={config.UNET_LR}, WD={config.UNET_WEIGHT_DECAY})")

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(config.ADAM_BETA1, config.ADAM_BETA2),
            eps=config.ADAM_EPSILON,
        )
        return optimizer

    def _create_scheduler(self):
        """Create LR scheduler. Steps counted in optimizer steps (not micro-batches)."""
        steps_per_epoch = len(self.train_loader) // config.GRADIENT_ACCUMULATION_STEPS
        total_steps = steps_per_epoch * self.num_epochs

        print(f"\nScheduler: {config.LR_SCHEDULER}")
        print(f"  Steps/epoch: {steps_per_epoch}")
        print(f"  Total optimizer steps: {total_steps}")
        print(f"  Warmup steps: {config.LR_WARMUP_STEPS}")

        if config.LR_SCHEDULER == "constant_with_warmup":
            return get_constant_schedule_with_warmup(self.optimizer, num_warmup_steps=config.LR_WARMUP_STEPS)
        elif config.LR_SCHEDULER == "cosine":
            return get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=config.LR_WARMUP_STEPS, num_training_steps=total_steps)
        elif config.LR_SCHEDULER == "constant":
            return get_constant_schedule(self.optimizer)
        else:
            raise ValueError(f"Unknown scheduler: {config.LR_SCHEDULER}")


    def train(self):
        """Main training loop with manual AMP + gradient accumulation."""
        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)

        self.model.train()
        accum_steps = config.GRADIENT_ACCUMULATION_STEPS

        for epoch in range(self.num_epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-" * 70)

            # LoRA health check
            cn_lora = sum(p.numel() for n, p in self.model.controlnet.named_parameters()
                         if 'lora' in n.lower() and p.requires_grad)
            unet_lora = sum(p.numel() for n, p in self.model.unet.named_parameters()
                          if 'lora' in n.lower() and p.requires_grad)
            print(f"[LoRA] ControlNet trainable: {cn_lora:,} | UNet trainable: {unet_lora:,}")
            if cn_lora == 0:
                raise RuntimeError(f"ControlNet LoRA params missing at epoch {epoch+1}!")
            if config.USE_UNET_LORA and unet_lora == 0:
                raise RuntimeError(f"UNet LoRA params missing at epoch {epoch+1}!")

            epoch_losses = []
            epoch_grad_norms = []

            self.optimizer.zero_grad()  # Zero once at start of epoch

            for batch_idx, batch in enumerate(self.train_loader):
                images = batch['image'].to(self.device)
                mask_rgb = batch['mask_rgb'].to(self.device)
                prompts = batch['prompt']

                # Forward + backward with AMP
                with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                    outputs = self.model(images, mask_rgb, prompts)
                    loss = outputs['loss'] / accum_steps

                self.scaler.scale(loss).backward()

                # Optimizer step every accum_steps micro-batches
                if (batch_idx + 1) % accum_steps == 0:
                    # Unscale before clipping
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(self.trainable_params, config.MAX_GRAD_NORM)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.lr_scheduler.step()

                    self.global_step += 1
                    step_loss = outputs['loss'].item()  # unscaled loss for logging
                    epoch_losses.append(step_loss)
                    epoch_grad_norms.append(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)

                    # Periodic logging
                    if self.global_step % config.LOGGING_STEPS == 0:
                        avg_loss = np.mean(epoch_losses[-config.LOGGING_STEPS:])
                        avg_gnorm = np.mean(epoch_grad_norms[-config.LOGGING_STEPS:])
                        lr = self.optimizer.param_groups[0]['lr']
                        print(f"Step {self.global_step} | Loss: {avg_loss:.4f} | GradNorm: {avg_gnorm:.4f} | LR: {lr:.2e}")

                        if config.USE_WANDB and WANDB_AVAILABLE:
                            wandb.log({
                                "train/loss": avg_loss,
                                "train/grad_norm": avg_gnorm,
                                "train/lr": lr,
                                "train/scaler_scale": self.scaler.get_scale(),
                            }, step=self.global_step)

            # Handle leftover micro-batches (if dataset not divisible by accum_steps)
            if len(self.train_loader) % accum_steps != 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(self.trainable_params, config.MAX_GRAD_NORM)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
                self.global_step += 1
                epoch_losses.append(outputs['loss'].item())
                epoch_grad_norms.append(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)

            # ---- End of epoch: validate + checkpoint ----
            epoch_avg_loss = np.mean(epoch_losses) if epoch_losses else float('nan')
            epoch_avg_gnorm = np.mean(epoch_grad_norms) if epoch_grad_norms else float('nan')
            val_loss = self.validate()
            lr = self.optimizer.param_groups[0]['lr']

            print(f"\n{'='*70}")
            print(f"EPOCH {epoch + 1}/{self.num_epochs} COMPLETE")
            print(f"{'='*70}")
            print(f"Train loss: {epoch_avg_loss:.4f}")
            print(f"Val loss:   {val_loss:.4f} | Best: {self.best_val_loss:.4f}")
            print(f"Avg grad norm: {epoch_avg_gnorm:.4f}")
            print(f"AMP scaler scale: {self.scaler.get_scale():.0f}")
            print(f"Total steps: {self.global_step}")

            # LoRA weight norm — proves LoRA is alive and changing each epoch
            lora_b_norms = [p.data.norm().item() for n, p in self.model.controlnet.named_parameters()
                            if 'lora_B' in n and p.requires_grad]
            if lora_b_norms:
                print(f"LoRA-B norm: mean={np.mean(lora_b_norms):.6f}  max={np.max(lora_b_norms):.6f}")

            is_best_val = val_loss < self.best_val_loss
            if is_best_val:
                self.best_val_loss = val_loss
                print("NEW BEST VAL LOSS!")

            with open(self.metrics_log_path, 'a') as f:
                f.write(f"{self.global_step},{self.epoch},{epoch_avg_loss:.6f},{val_loss:.6f},{lr:.8f},{epoch_avg_gnorm:.6f}\n")

            if config.USE_WANDB and WANDB_AVAILABLE:
                wandb.log({
                    "epoch/val_loss": val_loss,
                    "epoch/train_loss": epoch_avg_loss,
                    "epoch/grad_norm": epoch_avg_gnorm,
                    "epoch": epoch + 1,
                }, step=self.global_step)

            self.save_checkpoint(is_best_val=is_best_val)
            plot_training_curves(self.metrics_log_path, self.output_dir)
            print(f"{'='*70}\n")

        # ---- Post-training: load best model, FID + samples ----
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"Total steps: {self.global_step}")
        print(f"Run ID: {self.run_id}")
        print("=" * 70)

        best_val_path = config.CHECKPOINT_DIR / "best_val_model.pt"
        if best_val_path.exists():
            print(f"\nLoading best val model from {best_val_path}")
            checkpoint = torch.load(best_val_path, map_location='cpu', weights_only=False)
            cn_state = self.model.controlnet.state_dict()
            cn_state.update(checkpoint['controlnet_lora'])
            self.model.controlnet.load_state_dict(cn_state)
            if config.USE_UNET_LORA and 'unet_lora' in checkpoint:
                unet_state = self.model.unet.state_dict()
                unet_state.update(checkpoint['unet_lora'])
                self.model.unet.load_state_dict(unet_state)

        print("\nBuilding pipeline + running final evaluation...")
        self._build_pipeline()
        final_results = self.evaluate_full(control_strengths=[config.CONTROLNET_CONDITIONING_SCALE])
        print(f"FID: {final_results['best_fid']:.2f}")
        print(f"mIoU: {final_results['best_miou']:.4f} (placeholder)")

        print("\nGenerating final samples...")
        self._run_generate_samples()

        if config.USE_WANDB and WANDB_AVAILABLE:
            wandb.log({"final/fid": final_results['best_fid']}, step=self.global_step)
            wandb.finish()

    @torch.inference_mode()
    def validate(self):
        """Validation loss — random noise/timesteps each epoch (no fixed seed)."""
        self.model.eval()
        val_losses = []

        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state(self.device) if self.device.type == 'cuda' else None

        for batch in self.val_loader:
            images = batch['image'].to(self.device)
            mask_rgb = batch['mask_rgb'].to(self.device)
            prompts = batch['prompt']

            with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                outputs = self.model(images, mask_rgb, prompts)
            val_losses.append(outputs['loss'].item())

        torch.set_rng_state(rng_state)
        if self.device.type == 'cuda' and cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, self.device)

        self.model.train()
        return np.mean(val_losses)

    @torch.inference_mode()
    def evaluate_full(self, control_strengths=[1.0]):
        """Full evaluation: FID + mIoU (mIoU placeholder until Phase 2)."""
        self.model.eval()
        print("Running full evaluation...")

        results = []
        for strength in control_strengths:
            print(f"  Control strength: {strength}")

            miou_metrics = self.evaluator.evaluate_miou(
                self.model, self.val_loader, self.pipeline,
                control_strength=strength, num_inference_steps=config.NUM_INFERENCE_STEPS,
            )
            fid = self.evaluator.evaluate_fid(
                self.model, self.val_loader, self.pipeline,
                control_strength=strength, num_inference_steps=config.NUM_INFERENCE_STEPS,
            )

            results.append({
                'control_strength': strength,
                'miou': miou_metrics.get('miou', 0.0),
                'fid': fid,
            })
            print(f"    FID: {fid:.2f} | mIoU: {miou_metrics.get('miou', 0.0):.4f}")

            if self.evaluator.fid_metric is not None:
                self.evaluator.fid_metric.reset()
            gc.collect()
            torch.cuda.empty_cache()

        best_fid_result = min(results, key=lambda x: x['fid'])
        best_miou_result = max(results, key=lambda x: x['miou'])
        self.model.train()

        return {
            'results': results,
            'best_miou': best_miou_result['miou'],
            'best_fid': best_fid_result['fid'],
        }

    def _build_pipeline(self):
        """Merge LoRA and build inference pipeline. Called once after training."""
        print("Merging LoRA weights and building pipeline...")
        self.pipeline = create_pipeline(self.model, device=self.device)
        self.pipeline.set_progress_bar_config(disable=True)
        print("Pipeline ready.")

    def _run_generate_samples(self):
        """Generate sample grids from current pipeline."""
        self.model.eval()
        generate_samples(
            model=self.model, pipeline=self.pipeline, val_loader=self.val_loader,
            output_dir=self.output_dir, step=self.global_step, epoch=self.epoch, run_id=self.run_id,
        )
        self.model.train()

    def save_checkpoint(self, is_best_val: bool = False):
        """Save LoRA-only checkpoint when val loss improves (~5MB, not 5GB)."""
        if not is_best_val:
            return

        best_val_path = config.CHECKPOINT_DIR / "best_val_model.pt"
        checkpoint = {
            'controlnet_lora': {k: v for k, v in self.model.controlnet.state_dict().items() if 'lora_' in k},
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
        }
        if config.USE_UNET_LORA:
            checkpoint['unet_lora'] = {k: v for k, v in self.model.unet.state_dict().items() if 'lora_' in k}
        torch.save(checkpoint, best_val_path)
        print(f"★ BEST VAL MODEL saved (val_loss={self.best_val_loss:.4f}): {best_val_path}")


def main():
    """Main entry point."""
    print("=" * 70)
    print("CONTROLNET TRAINING - PHASE 1: DIFFUSION ONLY")
    print("=" * 70)

    # Reproducibility
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    config.print_config()

    # Datasets
    print("\nCreating datasets...")
    train_dataset = create_train_dataset(resolution=config.RESOLUTION)
    val_dataset = create_val_dataset(resolution=config.RESOLUTION)

    train_loader = create_dataloader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
    )
    val_loader = create_dataloader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
    )
    print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Val:   {len(val_dataset)} samples, {len(val_loader)} batches")

    # Sanity check
    print("\n" + "=" * 70)
    print("SANITY CHECK: Fetching one batch...")
    print("=" * 70)
    batch = next(iter(train_loader))
    print(f"Image: {batch['image'].shape}, range [{batch['image'].min():.2f}, {batch['image'].max():.2f}]")
    print(f"Mask RGB: {batch['mask_rgb'].shape}, range [{batch['mask_rgb'].min():.2f}, {batch['mask_rgb'].max():.2f}]")
    print(f"Prompt: {batch['prompt'][0][:100]}...")
    print("=" * 70)

    # Model
    print("\nInitializing model...")
    model = DiffusionControlNet(pretrained=True, device=device)

    # Train
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader, device=device,
    )
    trainer.train()


if __name__ == "__main__":
    main()
