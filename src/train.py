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
from .evaluation import ControlNetEvaluator
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
        gradient_accumulation_steps: int = None, # Rtx 3080 can't do without accumulation, but keep option for 3090+ users
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
        self.scaler = torch.amp.GradScaler('cuda') if self.mixed_precision == "fp16" else None
        
        # Training state - track multiple metrics
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')  # Diffusion loss (for monitoring only)
        self.best_overall_val_loss = self._load_best_overall_loss()
        self.best_miou = 0.0  # PRIMARY METRIC for model selection
        self.best_fid = float('inf')  # Secondary metric for realism
        
        # Use SLURM job ID if available, otherwise timestamp
        slurm_job_id = os.environ.get('SLURM_JOB_ID')
        if slurm_job_id:
            self.run_id = slurm_job_id
        else:
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup CSV logging
        self.metrics_log_path = config.LOG_DIR / f"metrics_{self.run_id}.csv"
        with open(self.metrics_log_path, 'w') as f:
            f.write("step,epoch,train_loss,val_loss,miou,fid,control_strength,lr\n")
        
        # Setup evaluator (segmentor will be loaded separately if needed)
        # NOTE: To enable mIoU computation, call:
        #   trainer.evaluator.load_segmentor(checkpoint_path)
        # before training or load segmentor_model in evaluator init
        self.evaluator = ControlNetEvaluator(device=self.model.device)
        
        # Create pipeline once (reused for all generation)
        # Pipeline holds references to model.unet and model.controlnet
        # so it automatically sees weight updates during training
        self.pipeline = create_pipeline(self.model, device=self.model.device)
        self.pipeline.set_progress_bar_config(disable=True)
        
        print("=" * 70)
        print("TRAINER INITIALIZED")
        print("=" * 70)
        print(f"Learning rate: {self.learning_rate}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"Mixed precision: {self.mixed_precision}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 70)
    
    def _load_best_overall_loss(self):
        """Load best validation loss from all previous runs."""
        best_overall_path = config.CHECKPOINT_DIR / "best_model_overall.pt"
        
        if best_overall_path.exists():
            try:
                checkpoint = torch.load(best_overall_path, map_location='cpu')
                best_loss = checkpoint.get('best_val_loss', float('inf'))
                print(f"📊 Loaded best overall validation loss: {best_loss:.4f}")
                return best_loss
            except Exception as e:
                print(f"⚠️  Could not load best overall checkpoint: {e}")
                return float('inf')
        else:
            print("📊 No previous best model found, starting fresh")
            return float('inf')
    
    def _create_optimizer(self):
        """Create optimizer with separate learning rates for ControlNet LoRA and UNet blocks."""
        # Separate parameter groups for different learning rates
        param_groups = []
        
        # ControlNet LoRA parameters
        controlnet_params = [p for p in self.model.controlnet.parameters() if p.requires_grad]
        if controlnet_params:
            param_groups.append({
                "params": controlnet_params,
                "lr": config.CONTROLNET_LORA_LR,
                "name": "controlnet_lora"
            })
        
        # UNet late blocks parameters (unfrozen for texture learning)
        unet_params = [p for p in self.model.unet.parameters() if p.requires_grad]
        if unet_params:
            param_groups.append({
                "params": unet_params,
                "lr": config.UNET_LR,
                "name": "unet_late_blocks"
            })
        
        # Store all trainable params for gradient clipping
        self.trainable_params = []
        for group in param_groups:
            self.trainable_params.extend(group["params"])
        
        # Print parameter counts
        total_param_count = sum(p.numel() for p in self.trainable_params)
        controlnet_param_count = sum(p.numel() for p in controlnet_params)
        unet_param_count = sum(p.numel() for p in unet_params) if unet_params else 0
        
        print("\n" + "=" * 70)
        print("🔴 OPTIMIZER PARAMETER COUNT")
        print("=" * 70)
        print(f"Total trainable parameters: {total_param_count:,}")
        print(f"  - ControlNet LoRA: {controlnet_param_count:,} (LR: {config.CONTROLNET_LORA_LR})")
        print(f"  - UNet late blocks: {unet_param_count:,} (LR: {config.UNET_LR})")
        print(f"Memory estimate (FP32): {total_param_count * 4 / 1e9:.2f} GB")
        print(f"Memory estimate (FP16): {total_param_count * 2 / 1e9:.2f} GB")
        print(f"AdamW state memory: ~{total_param_count * 8 / 1e9:.2f} GB (2x params for momentum + variance)")
        print("=" * 70 + "\n")
        
        if config.OPTIMIZER.lower() == "adamw":
            print("✓ Using AdamW optimizer")
            optimizer = torch.optim.AdamW(
                param_groups,
                betas=(config.ADAM_BETA1, config.ADAM_BETA2),
                weight_decay=config.WEIGHT_DECAY,
                eps=config.ADAM_EPSILON
            )
        elif config.OPTIMIZER.lower() == "adam":
            optimizer = torch.optim.Adam(
                param_groups,
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
            with torch.amp.autocast('cuda'):
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
                self.trainable_params,
                config.MAX_GRAD_NORM
            )
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # Zero gradients - must be after update()
            self.optimizer.zero_grad()
        else:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.trainable_params,
                config.MAX_GRAD_NORM
            )
            # Optimizer step
            self.optimizer.step()
            # Zero gradients
            self.optimizer.zero_grad()
        
        # Scheduler step
        self.lr_scheduler.step()
    
    @torch.inference_mode()
    def validate(self):
        """Run validation (diffusion loss only)."""
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
    
    @torch.inference_mode()
    def evaluate_full(self, val_loss: float, control_strengths=[1.0]):
        """
        Run full evaluation with mIoU and FID across multiple control strengths.
        
        Args:
            val_loss: Pre-computed validation loss (passed from train loop)
            control_strengths: List of control strength values to test
            
        Returns:
            dict with results for each control strength
        """
        self.model.eval()
        
        print("\nRunning full evaluation (mIoU + FID)...")
        
        results = []
        for strength in control_strengths:
            print(f"\n  Testing control strength: {strength}")
            
            # Evaluate mIoU (if segmentor available)
            miou_metrics = self.evaluator.evaluate_miou(
                self.model,
                self.val_loader,
                self.pipeline,
                control_strength=strength,
                num_inference_steps=config.NUM_INFERENCE_STEPS,
            )
            
            # Evaluate FID
            fid = self.evaluator.evaluate_fid(
                self.model,
                self.val_loader,
                self.pipeline,
                control_strength=strength,
                num_inference_steps=config.NUM_INFERENCE_STEPS,
            )
            
            results.append({
                'control_strength': strength,
                'miou': miou_metrics.get('miou', 0.0),
                'pixel_acc': miou_metrics.get('pixel_acc', 0.0),
                'fid': fid,
            })
            
            print(f"    mIoU: {miou_metrics.get('miou', 0.0):.4f}")
            print(f"    FID: {fid:.2f}")
        
        # Return best results (highest mIoU, lowest FID)
        best_miou_result = max(results, key=lambda x: x['miou'])
        best_fid_result = min(results, key=lambda x: x['fid'])
        
        self.model.train()
        
        return {
            'results': results,
            'best_miou': best_miou_result['miou'],
            'best_miou_strength': best_miou_result['control_strength'],
            'best_fid': best_fid_result['fid'],
            'best_fid_strength': best_fid_result['control_strength'],
        }
    
    def _mask_rgb_to_pil(self, mask_rgb: torch.Tensor, resolution: int) -> Image:
        """
        Convert (3, H, W) float tensor in [0,1] to PIL RGB image with NEAREST resize.
        
        Args:
            mask_rgb: (3, H, W) tensor with values in [0, 1]
            resolution: Target resolution for resize
            
        Returns:
            PIL Image with proper resolution
        """
        assert mask_rgb.ndim == 3 and mask_rgb.shape[0] == 3, f"Expected (3, H, W), got {mask_rgb.shape}"
        
        mask_np = (mask_rgb.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        pil_img = Image.fromarray(mask_np).convert("RGB")
        
        if pil_img.size != (resolution, resolution):
            pil_img = pil_img.resize((resolution, resolution), Image.NEAREST)
        
        return pil_img
    
    @torch.inference_mode()
    def generate_samples(self, num_samples: int = 4, num_images_per_prompt: int = 4):
        """
        Generate sample images for visualization.
        
        Args:
            num_samples: Number of different prompts to use
            num_images_per_prompt: Number of variations to generate per prompt
        """
        self.model.eval()
        
        # Get validation batch - sample different images
        all_samples = []
        for i, batch in enumerate(self.val_loader):
            all_samples.append(batch)
            if i + 1 >= num_samples:
                break
        
        # Use only the first sample from each batch to get variety
        prompts = [s['prompt'][0] for s in all_samples]
        mask_rgb = torch.cat([s['mask_rgb'][:1] for s in all_samples])
        original_images = torch.cat([s['image'][:1] for s in all_samples])
        
        # Deterministic sampling for reproducible results
        generator = torch.Generator(device=self.model.device).manual_seed(42)
        
        # Generate multiple images per prompt for variation
        all_generated = []
        for i in range(len(prompts)):
            # Convert mask to PIL with helper
            control_pil = self._mask_rgb_to_pil(mask_rgb[i], config.RESOLUTION)
            
            # Type assertions for safety
            assert isinstance(prompts[i], str), f"Expected str, got {type(prompts[i])}"
            assert isinstance(control_pil, Image.Image), f"Expected PIL Image, got {type(control_pil)}"
            
            generated = self.pipeline(
                prompt=prompts[i],
                image=control_pil,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,
                num_inference_steps=config.NUM_INFERENCE_STEPS,
                guidance_scale=config.GUIDANCE_SCALE,
                controlnet_conditioning_scale=config.CONTROLNET_CONDITIONING_SCALE,
            ).images
            all_generated.append(generated)
        
        # Save samples with concatenated visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_dir = self.samples_dir / f"step_{self.global_step:06d}_{timestamp}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        from PIL import ImageDraw, ImageFont
        
        for i, prompt in enumerate(prompts):
            generated_imgs = all_generated[i]
            
            # Prepare images for concatenation
            orig_img = original_images[i].cpu()
            orig_img = (orig_img * 0.5 + 0.5).clamp(0, 1)
            orig_img = (orig_img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            orig_pil = Image.fromarray(orig_img)
            
            mask_pil = self._mask_rgb_to_pil(mask_rgb[i], config.RESOLUTION)
            
            # Create concatenated image: [input | control | output1 | output2 | output3 | output4]
            img_width, img_height = orig_pil.size
            text_height = 60
            total_width = img_width * (2 + num_images_per_prompt)
            total_height = img_height + text_height
            
            concat_img = Image.new('RGB', (total_width, total_height), color='white')
            draw = ImageDraw.Draw(concat_img)
            
            # Try to use a font, fallback to default if not available
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Draw prompt text at top
            prompt_text = f"Step {self.global_step} | {prompt}"
            if len(prompt_text) > 150:
                prompt_text = prompt_text[:147] + "..."
            draw.text((10, 10), prompt_text, fill='black', font=font)
            
            # Paste images
            concat_img.paste(orig_pil, (0, text_height))
            concat_img.paste(mask_pil, (img_width, text_height))
            for j, gen_img in enumerate(generated_imgs):
                concat_img.paste(gen_img, ((2 + j) * img_width, text_height))
            
            # Save concatenated visualization
            concat_img.save(sample_dir / f"sample_{i}_concat.png")
            
            # Also save individual images
            orig_pil.save(sample_dir / f"sample_{i}_input.png")
            mask_pil.save(sample_dir / f"sample_{i}_control.png")
            for j, gen_img in enumerate(generated_imgs):
                gen_img.save(sample_dir / f"sample_{i}_output_{j}.png")
        
        print(f"✓ Saved {num_samples} samples to {sample_dir}")
        print(f"  - Concatenated visualizations: sample_*_concat.png")
        print(f"  - {num_images_per_prompt} variations per prompt")
        
        self.model.train()
    
    def save_checkpoint(self, is_best: bool = False, is_best_overall: bool = False, 
                       is_best_miou: bool = False, is_best_fid: bool = False):
        """
        Save training checkpoint.
        
        Args:
            is_best: Whether this is the best model in current run (by val_loss)
            is_best_overall: Whether this is the best model across all runs (by val_loss)
            is_best_miou: Whether this is the best model by mIoU (PRIMARY METRIC)
            is_best_fid: Whether this is the best model by FID (secondary metric)
        """
        # Generate timestamp for this run
        if not hasattr(self, 'run_timestamp'):
            self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save regular checkpoint with run timestamp
        checkpoint_path = config.CHECKPOINT_DIR / f"run_{self.run_timestamp}_step_{self.global_step:06d}.pt"
        
        checkpoint = {
            'model_unet': self.model.unet.state_dict(),
            'model_controlnet': self.model.controlnet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'scaler': self.scaler.state_dict() if self.scaler else None,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'best_miou': self.best_miou,
            'best_fid': self.best_fid,
            'run_timestamp': self.run_id,
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        
        # Save best model for current run (by val_loss)
        if is_best:
            best_current_path = config.CHECKPOINT_DIR / f"best_run_{self.run_id}.pt"
            torch.save(checkpoint, best_current_path)
            print(f"✓ Best model (current run) saved: {best_current_path}")
        
        # Save best model overall (by val_loss)
        if is_best_overall:
            best_overall_path = config.CHECKPOINT_DIR / "best_model_overall.pt"
            torch.save(checkpoint, best_overall_path)
            print(f"★ BEST MODEL OVERALL saved: {best_overall_path}")
        
        # Save best model by mIoU (PRIMARY METRIC for model selection)
        if is_best_miou:
            best_miou_path = config.CHECKPOINT_DIR / "best_miou_model.pt"
            torch.save(checkpoint, best_miou_path)
            print(f"★ BEST mIoU MODEL saved: {best_miou_path}")
        
        # Save best model by FID (secondary metric for realism)
        if is_best_fid:
            best_fid_path = config.CHECKPOINT_DIR / "best_fid_model.pt"
            torch.save(checkpoint, best_fid_path)
            print(f"★ BEST FID MODEL saved: {best_fid_path}")
        
        # Keep only last N checkpoints for current run
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints from current run, keeping only the last N."""
        # Only clean up regular checkpoints from current run
        if not hasattr(self, 'run_timestamp'):
            return
        
        # Get checkpoints from current run only (exclude best_* files)
        current_run_pattern = f"run_{self.run_timestamp}_step_*.pt"
        checkpoints = sorted(config.CHECKPOINT_DIR.glob(current_run_pattern))
        
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
            
            epoch_losses = []
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}", disable=True)
            
            for step, batch in enumerate(progress_bar):
                # Training step
                loss = self.train_step(batch)
                epoch_losses.append(loss)
                
                # Gradient accumulation
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer_step()
                    self.global_step += 1
                    
                    # Log every N steps for visibility in log files
                    if self.global_step % config.LOGGING_STEPS == 0:
                        avg_loss = np.mean(epoch_losses[-config.LOGGING_STEPS:])
                        lr = self.optimizer.param_groups[0]['lr']
                        print(f"Step {self.global_step} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")
                    
                    # Validation (diffusion loss only)
                    if self.global_step % self.validation_steps == 0:
                        val_loss = self.validate()
                        avg_train_loss = np.mean(epoch_losses[-self.validation_steps:]) if len(epoch_losses) >= self.validation_steps else np.mean(epoch_losses)
                        lr = self.optimizer.param_groups[0]['lr']
                        
                        print(f"\n{'='*70}")
                        print(f"VALIDATION - Step {self.global_step}")
                        print(f"{'='*70}")
                        print(f"Train loss: {avg_train_loss:.4f}")
                        print(f"Val loss: {val_loss:.4f}")
                        
                        # Check if best by val_loss (for monitoring)
                        is_best = val_loss < self.best_val_loss
                        if is_best:
                            self.best_val_loss = val_loss
                            print(f"✓ New best val loss: {val_loss:.4f}")
                        
                        # Check if best overall
                        is_best_overall = val_loss < self.best_overall_val_loss
                        if is_best_overall:
                            self.best_overall_val_loss = val_loss
                            print(f"★ NEW BEST OVERALL!")
                        
                        # Log diffusion loss only
                        with open(self.metrics_log_path, 'a') as f:
                            f.write(f"{self.global_step},{self.epoch},{avg_train_loss:.6f},{val_loss:.6f},,,1.0,{lr:.8f}\n")
                        
                        print(f"{'='*70}\n")
                        
                        # Generate samples
                        self.generate_samples()
                        
                        # Save checkpoint (no mIoU/FID flags during step validation)
                        self.save_checkpoint(
                            is_best=is_best,
                            is_best_overall=is_best_overall
                        )
                    
                    # Save checkpoint
                    elif self.global_step % self.save_steps == 0:
                        self.save_checkpoint()
            
            # End of epoch summary
            epoch_avg_loss = np.mean(epoch_losses)
            print(f"\n{'='*70}")
            print(f"EPOCH {epoch + 1}/{self.num_epochs} COMPLETE")
            print(f"{'='*70}")
            print(f"Average training loss: {epoch_avg_loss:.4f}")
            print(f"Total steps: {self.global_step}")
            
            # Run full evaluation at epoch end
            print("\n" + "-" * 70)
            print("END-OF-EPOCH FULL EVALUATION")
            print("-" * 70)
            
            # Compute validation loss first
            val_loss = self.validate()
            
            # Run full evaluation with single control strength (default 1.0)
            eval_results = self.evaluate_full(val_loss=val_loss, control_strengths=[1.0])
            miou = eval_results['best_miou']
            fid = eval_results['best_fid']
            miou_strength = eval_results['best_miou_strength']
            fid_strength = eval_results['best_fid_strength']
            lr = self.optimizer.param_groups[0]['lr']
            
            print(f"\nmIoU: {miou:.4f} | Best: {self.best_miou:.4f}")
            print(f"FID: {fid:.2f} | Best: {self.best_fid:.2f}")
            print(f"Val loss: {val_loss:.4f} | Best: {self.best_val_loss:.4f}")
            
            # Check for improvements
            is_best_miou = False
            is_best_fid = False
            is_best_val = val_loss < self.best_val_loss
            
            # Track FID (doesn't require segmentor)
            is_best_fid = fid < self.best_fid
            if is_best_fid:
                self.best_fid = fid
                print(f"★ NEW BEST FID!")
            
            # Track mIoU (only if segmentor available)
            if self.evaluator.segmentor is not None:
                is_best_miou = miou > self.best_miou
                if is_best_miou:
                    self.best_miou = miou
                    print(f"★ NEW BEST mIoU!")
            else:
                print("⚠️ No segmentor loaded - mIoU tracking disabled")
            
            if is_best_val:
                self.best_val_loss = val_loss
                print(f"✓ NEW BEST VAL LOSS!")
            
            # Log all control strengths to CSV
            for result in eval_results['results']:
                with open(self.metrics_log_path, 'a') as f:
                    f.write(f"{self.global_step},{self.epoch},{epoch_avg_loss:.6f},{val_loss:.6f},"
                           f"{result['miou']:.6f},{result['fid']:.2f},{result['control_strength']:.1f},{lr:.8f}\n")
            
            # Save checkpoint
            self.save_checkpoint(
                is_best=is_best_val,
                is_best_miou=is_best_miou,
                is_best_fid=is_best_fid
            )
            
            print(f"{'='*70}\n")
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Best mIoU (PRIMARY): {self.best_miou:.4f}")
        print(f"Best FID: {self.best_fid:.2f}")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"Total steps: {self.global_step}")
        print(f"Run ID: {self.run_id}")
        print(f"Metrics log: {self.metrics_log_path}")
        print("=" * 70)


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
    
    # Sanity check: fetch one batch
    print("\n" + "=" * 70)
    print("🔎 SANITY CHECK: Fetching one batch...")
    print("=" * 70)
    try:
        batch = next(iter(train_loader))
        print("✓ Batch keys:", batch.keys())
        print("✓ Image shape:", batch["image"].shape)
        print("✓ Mask shape:", batch["mask"].shape)
        print("✓ Number of prompts:", len(batch["prompt"]))
        print("✓ First prompt:", batch["prompt"][0][:100] + "..." if len(batch["prompt"][0]) > 100 else batch["prompt"][0])
        print("=" * 70 + "\n")
    except Exception as e:
        print(f"❌ FAILED to fetch batch: {e}")
        raise
    
    # Initialize model
    print("Initializing model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = DiffusionControlNet(
        pretrained=True,
        device=device
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
