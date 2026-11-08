"""
Basic training loop for ControlNet fine-tuning.
Phase 1: Diffusion loss only (no segmentor).

No Accelerate — pure PyTorch with manual AMP + gradient accumulation
so every step is transparent and debuggable.
"""

import gc
import json
import os
import random
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
        # bf16 needs an Ampere-or-newer GPU (compute capability >= 8.0); V100 is 7.0.
        # torch.cuda.is_bf16_supported() would also report True through slow emulation, so check the capability.
        if config.MIXED_PRECISION == "bf16" and device.type == "cuda" and torch.cuda.get_device_capability(device)[0] < 8:
            print(f"[INFO] {torch.cuda.get_device_name(device)} has no native bf16 — using fp16 with gradient scaling")
            config.MIXED_PRECISION = "fp16"
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
        self.best_kid = float('inf')
        self.best_epoch = None  # epochs completed when the saved checkpoint was taken
        self.best_gen_metrics = None  # generation metrics of the saved checkpoint (reused by the final eval)

        # Run ID
        # e.g. "sanpo_1838500" on dataset 2, "1838500" on dataset 1 (see config.DATASET_TAG)
        self.run_id = config.DATASET_TAG + (os.environ.get('SLURM_JOB_ID') or datetime.now().strftime("%Y%m%d_%H%M%S"))

        # Config snapshot — every run records the exact settings it used
        with open(config.LOG_DIR / f"{self.run_id}_config.json", 'w') as f:
            json.dump(config.get_config_dict(), f, indent=2, default=str)

        # CSV logging
        self.metrics_log_path = config.LOG_DIR / f"metrics_{self.run_id}.csv"
        with open(self.metrics_log_path, 'w') as f:
            f.write("step,epoch,train_loss,val_loss,lr,grad_norm,fid,kid,lpips,clip_prompt,clip_img,miou,miou_real,seg_loss,seg_weight\n")

        # Evaluator — shares the model's frozen segmentor when the Phase-2 loss loaded one,
        # so the checkpoint is not loaded (and kept on the GPU) twice
        self.evaluator = ControlNetEvaluator(device=self.device)
        if getattr(self.model, "_segmentor", None):
            self.evaluator.segmentor = self.model._segmentor[0]
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
        print(f"Run: {self.run_id} {config.RUN_NAME} | dataset: {config.DATASET_NAME} ({config.DATA_DIR})")
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
        print(f"Checkpoint selection: {config.CHECKPOINT_METRIC}")
        if config.USE_SEGMENTOR_LOSS:
            print(f"Segmentation loss: ON, weight {config.SEGMENTOR_LOSS_WEIGHT_START} → {config.SEGMENTOR_LOSS_WEIGHT} "
                  f"(linear over {config.SEGMENTOR_LOSS_WARMUP_STEPS} steps, then constant)")
        else:
            print("Segmentation loss: off")
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
        # The training loop does one step per accum window PLUS one leftover step
        # when the loader isn't divisible by accum (see train()). Count that
        # leftover so total_steps matches the real number of optimizer steps —
        # otherwise the cosine schedule reaches lr=0 a fraction of an epoch early.
        accum = config.GRADIENT_ACCUMULATION_STEPS
        n_batches = len(self.train_loader)
        steps_per_epoch = n_batches // accum + (1 if n_batches % accum else 0)
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

        # ---- Epoch 0: evaluate the starting weights (pretrained model, or INIT_LORA_CHECKPOINT) ----
        if config.EVAL_AT_START:
            start = f"INIT_LORA_CHECKPOINT {config.INIT_LORA_CHECKPOINT}" if config.INIT_LORA_CHECKPOINT else "pretrained model"
            print(f"\nEPOCH 0 = START OF RUN ({start}, before any training step)")
            print("-" * 70)
            val_loss = self.validate()
            print(f"Val loss: {val_loss:.6f}")
            gen_metrics = None
            if config.EVAL_EVERY_N_EPOCHS:
                gen_metrics = self._periodic_generation_eval()
                self._print_gen_metrics(gen_metrics)
            self._log_epoch_row(0, float('nan'), val_loss, 0.0, None, gen_metrics, None)
            self._update_best(0, val_loss, gen_metrics)

        for epoch in range(self.num_epochs):
            self.epoch = epoch
            if config.OVERFIT_FIXED_SEED:
                torch.manual_seed(config.SEED)
                torch.cuda.manual_seed(config.SEED)
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
            window_losses = []  # every micro-batch DIFFUSION loss in the current accumulation window
            epoch_seg_losses = []  # segmentation loss of the micro-batches where it applied (Phase 2)

            self.optimizer.zero_grad()  # Zero once at start of epoch

            for batch_idx, batch in enumerate(self.train_loader):
                images = batch['image'].to(self.device)
                mask_rgb = batch['mask_rgb'].to(self.device)
                prompts = batch['prompt']

                # Phase 2: segmentation-loss weight ramps linearly from its start value to its target
                seg_weight, mask_seg = 0.0, None
                if config.USE_SEGMENTOR_LOSS:
                    seg_weight = self._seg_weight()
                    mask_seg = batch['mask_seg'].to(self.device)

                # Forward + backward with AMP
                with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                    outputs = self.model(images, mask_rgb, prompts, mask_seg=mask_seg, seg_weight=seg_weight)
                    loss = outputs['loss'] / accum_steps

                self.scaler.scale(loss).backward()
                window_losses.append(outputs['diffusion_loss'].item())  # unscaled diffusion loss for logging
                if outputs['seg_loss'] is not None:
                    epoch_seg_losses.append(outputs['seg_loss'].item())

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
                    epoch_losses.append(float(np.mean(window_losses)))
                    window_losses = []
                    epoch_grad_norms.append(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)

                    # Periodic logging
                    if self.global_step % config.LOGGING_STEPS == 0:
                        avg_loss = np.mean(epoch_losses[-config.LOGGING_STEPS:])
                        avg_gnorm = np.mean(epoch_grad_norms[-config.LOGGING_STEPS:])
                        lr = self.optimizer.param_groups[0]['lr']
                        seg_info = f" | λ: {self._seg_weight():.4f}" if config.USE_SEGMENTOR_LOSS else ""
                        print(f"Step {self.global_step} | Loss: {avg_loss:.4f} | GradNorm: {avg_gnorm:.4f} | LR: {lr:.2e}{seg_info}")

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
                epoch_losses.append(float(np.mean(window_losses)))
                epoch_grad_norms.append(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)

            # ---- End of epoch: validate + checkpoint ----
            epoch_avg_loss = np.mean(epoch_losses) if epoch_losses else float('nan')
            epoch_avg_gnorm = np.mean(epoch_grad_norms) if epoch_grad_norms else float('nan')
            epoch_seg_loss = float(np.mean(epoch_seg_losses)) if epoch_seg_losses else None
            val_loss = self.validate()
            lr = self.optimizer.param_groups[0]['lr']

            print(f"\n{'='*70}")
            print(f"EPOCH {epoch + 1}/{self.num_epochs} COMPLETE")
            print(f"{'='*70}")
            print(f"Train loss: {epoch_avg_loss:.4f}")
            if epoch_seg_loss is not None:
                print(f"Seg loss:   {epoch_seg_loss:.4f}  (on {len(epoch_seg_losses)} of {len(self.train_loader)} batches)"
                      f" | λ at end of epoch: {self._seg_weight():.4f}")
            print(f"Val loss:   {val_loss:.4f} | Best: {self.best_val_loss:.4f}")
            print(f"Avg grad norm: {epoch_avg_gnorm:.4f}")
            print(f"AMP scaler scale: {self.scaler.get_scale():.0f}")
            print(f"Total steps: {self.global_step}")

            # LoRA weight norm — proves LoRA is alive and changing each epoch
            lora_b_norms = [p.data.norm().item() for n, p in self.model.controlnet.named_parameters()
                            if 'lora_B' in n and p.requires_grad]
            if lora_b_norms:
                print(f"LoRA-B norm: mean={np.mean(lora_b_norms):.6f}  max={np.max(lora_b_norms):.6f}")

            # Periodic generation-based eval (gives KID/LPIPS curves over training).
            # Always runs after the final epoch so the last state is a checkpoint candidate.
            gen_metrics = None
            is_last = (epoch + 1 == self.num_epochs)
            if config.EVAL_EVERY_N_EPOCHS and ((epoch + 1) % config.EVAL_EVERY_N_EPOCHS == 0 or is_last):
                print("Running periodic generation eval (FID/KID/LPIPS/CLIP)...")
                gen_metrics = self._periodic_generation_eval()
                self._print_gen_metrics(gen_metrics)

            self._log_epoch_row(epoch + 1, epoch_avg_loss, val_loss, lr, epoch_avg_gnorm, gen_metrics, epoch_seg_loss)
            self._update_best(epoch + 1, val_loss, gen_metrics)
            if config.SAVE_EVAL_CHECKPOINTS and gen_metrics is not None:
                self.save_checkpoint(epochs_done=epoch + 1)
            plot_training_curves(self.metrics_log_path, self.output_dir, run_id=self.run_id)
            print(f"{'='*70}\n")

        # ---- Post-training: load best model, FID + samples ----
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Best val loss: {self.best_val_loss:.6f}")
        print(f"Best KID: {self.best_kid:.5f}")
        print(f"Checkpoint taken after epoch: {self.best_epoch} "
              f"(0 = start of run: {'the INIT_LORA_CHECKPOINT' if config.INIT_LORA_CHECKPOINT else 'the pretrained model'})")
        print(f"Total steps: {self.global_step}")
        print(f"Run ID: {self.run_id}")
        print("=" * 70)

        best_val_path = config.CHECKPOINT_DIR / f"{self.run_id}_LoRA.pt"
        if not best_val_path.exists():
            # No candidate was ever selected (e.g. every generation eval failed) — keep the final weights
            print("[WARN] no best checkpoint was selected during training — saving final weights")
            self.best_epoch = self.num_epochs
            self.save_checkpoint()
        if best_val_path.exists():
            print(f"\nLoading best model from {best_val_path}")
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
        self.save_merged_controlnet()
        final_results = self.evaluate_full(control_strengths=[config.CONTROLNET_CONDITIONING_SCALE],
                                           reuse_metrics=self.best_gen_metrics)
        r = final_results['results'][0] if final_results['results'] else {}

        def _f(v, p=4):
            return f"{v:.{p}f}" if isinstance(v, (int, float)) else "n/a"

        print("\nFinal metrics (control_strength="
              f"{config.CONTROLNET_CONDITIONING_SCALE}):")
        print(f"  FID         : {_f(r.get('fid'), 2)}   (unreliable at n=50 — trend only)")
        print(f"  KID (mean)  : {_f(r.get('kid_mean'), 5)} ± {_f(r.get('kid_std'), 5)}   (trust this over FID)")
        print(f"  LPIPS↓ vs GT: {_f(r.get('lpips'))}   (lower = closer to real image)")
        print(f"  CLIP prompt↑: {_f(r.get('clip_prompt'))}   (image↔text alignment)")
        print(f"  CLIP img↑   : {_f(r.get('clip_img'))}   (image↔ground-truth similarity)")
        print(f"  mIoU↑       : {_f(r.get('miou'))}   (segmentor on generated images vs mask; "
              f"on the real images it reaches {_f(r.get('miou_real'))})")

        print("\nGenerating final samples...")
        self._run_generate_samples()

        if config.USE_WANDB and WANDB_AVAILABLE:
            wandb.log({
                f"final/{k}": v for k, v in r.items()
                if isinstance(v, (int, float))
            }, step=self.global_step)
            wandb.finish()

    @staticmethod
    def _print_gen_metrics(gen_metrics):
        if not gen_metrics:
            return
        def _g(v, p=4):
            return f"{v:.{p}f}" if isinstance(v, (int, float)) else "n/a"
        print(f"  FID={_g(gen_metrics['fid'],2)} | KID={_g(gen_metrics['kid_mean'],5)} ± {_g(gen_metrics['kid_std'],5)} "
              f"| LPIPS={_g(gen_metrics['lpips'])} | CLIP(prompt)={_g(gen_metrics['clip_prompt'])} "
              f"| CLIP(img)={_g(gen_metrics['clip_img'])} "
              f"| mIoU={_g(gen_metrics.get('miou'))} (real images: {_g(gen_metrics.get('miou_real'))})")

    def _log_epoch_row(self, epochs_done, train_loss, val_loss, lr, grad_norm, gen_metrics, seg_loss=None):
        """Append one row to the metrics CSV (+ WandB). epochs_done=0 is the start of the run (pretrained model, or INIT_LORA_CHECKPOINT)."""
        gm = gen_metrics or {}
        def _csv(v, p=6):
            return f"{v:.{p}f}" if isinstance(v, (int, float)) else ""
        with open(self.metrics_log_path, 'a') as f:
            f.write(
                f"{self.global_step},{epochs_done},{train_loss:.6f},{val_loss:.6f},"
                f"{lr:.8f},{_csv(grad_norm)},{_csv(gm.get('fid'),4)},{_csv(gm.get('kid_mean'),8)},"
                f"{_csv(gm.get('lpips'))},{_csv(gm.get('clip_prompt'))},{_csv(gm.get('clip_img'))},"
                f"{_csv(gm.get('miou'))},{_csv(gm.get('miou_real'))},{_csv(seg_loss)},"
                f"{_csv(self._seg_weight() if config.USE_SEGMENTOR_LOSS and epochs_done > 0 else None)}\n"
            )

        if config.USE_WANDB and WANDB_AVAILABLE:
            log_dict = {"epoch/val_loss": val_loss, "epoch": epochs_done}
            if epochs_done > 0:
                log_dict["epoch/train_loss"] = train_loss
                log_dict["epoch/grad_norm"] = grad_norm
            for k in ('fid', 'kid_mean', 'lpips', 'clip_prompt', 'clip_img', 'miou'):
                if isinstance(gm.get(k), (int, float)):
                    log_dict[f"epoch/{k}"] = gm[k]
            wandb.log(log_dict, step=self.global_step)

    def _update_best(self, epochs_done, val_loss, gen_metrics):
        """Track best val loss / KID and save the checkpoint when CHECKPOINT_METRIC improves."""
        val_loss = float(val_loss)
        is_best_val = val_loss < self.best_val_loss
        if is_best_val:
            self.best_val_loss = val_loss

        kid = (gen_metrics or {}).get('kid_mean')
        is_best_kid = isinstance(kid, (int, float)) and kid < self.best_kid
        if is_best_kid:
            self.best_kid = float(kid)

        if config.CHECKPOINT_METRIC == "last":
            is_best = True  # always keep the most recent weights (final eval = last epoch)
        elif config.CHECKPOINT_METRIC == "kid":
            is_best = is_best_kid
        else:
            is_best = is_best_val
        if is_best:
            self.best_epoch = epochs_done
            # the saved weights were evaluated with exactly the final-eval protocol at this epoch
            # (same images, seeds, settings) — keep the numbers so the final eval need not regenerate
            self.best_gen_metrics = gen_metrics
            print(f"NEW BEST ({config.CHECKPOINT_METRIC}) after epoch {epochs_done}")
            self.save_checkpoint()

    @torch.inference_mode()
    def validate(self):
        """Validation loss on FIXED noise and FIXED timesteps.

        The DDPM epsilon-loss is strongly timestep-dependent, so each val image is
        evaluated at the same VAL_NUM_TIMESTEPS timesteps (spread evenly over the
        schedule) with the same noise every epoch. val_loss then tracks the model,
        not which timesteps happened to be drawn. The global RNG is saved, reseeded,
        and restored so training's stochastic stream is untouched.
        """
        self.model.eval()
        val_losses = []

        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state(self.device) if self.device.type == 'cuda' else None

        torch.manual_seed(config.SEED)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(config.SEED)

        n_train_t = self.model.noise_scheduler.config.num_train_timesteps
        k = config.VAL_NUM_TIMESTEPS
        fixed_timesteps = [int((i + 0.5) * n_train_t / k) for i in range(k)]  # e.g. 62, 187, ..., 937

        for batch in self.val_loader:
            images = batch['image'].to(self.device)
            mask_rgb = batch['mask_rgb'].to(self.device)
            prompts = batch['prompt']

            for t in fixed_timesteps:
                timesteps = torch.full((images.shape[0],), t, device=self.device, dtype=torch.long)
                with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                    outputs = self.model(images, mask_rgb, prompts, timesteps=timesteps)
                val_losses.append(outputs['loss'].item())

        torch.set_rng_state(rng_state)
        if self.device.type == 'cuda' and cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, self.device)

        self.model.train()
        return float(np.mean(val_losses))

    @torch.inference_mode()
    def evaluate_full(self, control_strengths=[1.0], reuse_metrics=None):
        """Full evaluation: FID + KID + LPIPS + CLIP + segmentor mIoU.

        All generation-based metrics are computed in a single pass per control
        strength. Results are also written to a per-run JSON in LOG_DIR so runs
        can be compared cleanly afterwards.

        reuse_metrics: metrics already computed for these exact weights during training
        (the periodic eval of the selected epoch: same images, seeds and settings, and the
        merged copy used there equals the merged pipeline here). Used for the default
        control strength instead of generating the whole val set again.
        """
        self.model.eval()
        print("Running full evaluation...")

        def fmt(v, p=4):
            return f"{v:.{p}f}" if v is not None else "n/a"

        results = []
        for strength in control_strengths:
            print(f"  Control strength: {strength}")

            if reuse_metrics and strength == config.CONTROLNET_CONDITIONING_SCALE:
                print("    (reusing the metrics measured for this checkpoint during training)")
                metrics = reuse_metrics
            else:
                metrics = self.evaluator.evaluate(
                    self.model, self.val_loader, self.pipeline,
                    control_strength=strength, num_inference_steps=config.NUM_INFERENCE_STEPS,
                )
            row = {
                'control_strength': strength,
                'miou': metrics.get('miou'),
                'pixel_acc': metrics.get('pixel_acc'),
                'miou_real': metrics.get('miou_real'),
                'iou_per_class': metrics.get('iou_per_class'),
                'fid': metrics['fid'],
                'kid_mean': metrics['kid_mean'],
                'kid_std': metrics['kid_std'],
                'lpips': metrics['lpips'],
                'clip_prompt': metrics['clip_prompt'],
                'clip_img': metrics['clip_img'],
            }
            results.append(row)
            print(f"    FID: {fmt(row['fid'], 2)} | KID: {fmt(row['kid_mean'], 5)} "
                  f"| LPIPS: {fmt(row['lpips'])} | CLIP(prompt): {fmt(row['clip_prompt'])} "
                  f"| CLIP(img): {fmt(row['clip_img'])} | mIoU: {fmt(row['miou'])}")

            gc.collect()
            torch.cuda.empty_cache()

        # Best by each metric (guard against None when a metric is disabled)
        def best(key, mode):
            vals = [r for r in results if r[key] is not None]
            if not vals:
                return None
            return (min if mode == 'min' else max)(vals, key=lambda x: x[key])[key]

        best_fid = best('fid', 'min')
        best_miou = best('miou', 'max')

        # Persist per-run metrics for cross-run comparison
        eval_path = config.LOG_DIR / f"{self.run_id}_final_eval.json"
        try:
            import json
            with open(eval_path, 'w') as f:
                json.dump({'run_id': str(self.run_id), 'results': results}, f, indent=2)
            print(f"Final metrics written to {eval_path}")
        except Exception as e:
            print(f"[WARN] could not write eval JSON: {e}")

        self.model.train()
        return {
            'results': results,
            'best_miou': best_miou if best_miou is not None else 0.0,
            'best_fid': best_fid if best_fid is not None else float('inf'),
        }

    @torch.inference_mode()
    def _periodic_generation_eval(self):
        """Generation metrics (FID/KID/LPIPS/CLIP) DURING training — non-destructive.

        Builds a throwaway pipeline from a DEEPCOPY of the ControlNet, with LoRA
        merged into that copy only. The live trainable model is never mutated, so
        the optimizer's parameters and LoRA adapters stay intact. Any failure is
        caught and returns None so training is never jeopardised by this feature.
        """
        import copy
        from diffusers import StableDiffusionControlNetPipeline, DDIMScheduler

        cn_copy = None
        unet_copy = None
        try:
            cn_copy = copy.deepcopy(self.model.controlnet)
            merged_cn = cn_copy.merge_and_unload()  # destructive on the COPY only
            merged_cn.eval()

            # UNet: with USE_UNET_LORA=False (default) it's the frozen base UNet, which
            # has no merge_and_unload — we pass it through unchanged (shared read-only).
            # If UNet LoRA is ever enabled it's a PEFT model, so we merge a DEEPCOPY
            # (non-destructive on the live model), mirroring the ControlNet handling above.
            if hasattr(self.model.unet, "merge_and_unload"):
                unet_copy = copy.deepcopy(self.model.unet)
                eval_unet = unet_copy.merge_and_unload()
                eval_unet.eval()
            else:
                eval_unet = self.model.unet

            pipeline = StableDiffusionControlNetPipeline(
                vae=self.model.vae,
                text_encoder=self.model.text_encoder,
                tokenizer=self.model.tokenizer,
                unet=eval_unet,                  # frozen base (USE_UNET_LORA=False) — shared read-only
                controlnet=merged_cn,
                scheduler=DDIMScheduler.from_pretrained(config.SD_MODEL_ID, subfolder="scheduler"),
                safety_checker=None,
                feature_extractor=None,
            )
            pipeline.to(self.device)
            pipeline.set_progress_bar_config(disable=True)

            return self.evaluator.evaluate(
                self.model, self.val_loader, pipeline,
                control_strength=config.CONTROLNET_CONDITIONING_SCALE,
                num_inference_steps=config.NUM_INFERENCE_STEPS,
            )
        except Exception as e:
            print(f"[WARN] periodic generation eval failed (training continues): {e}")
            return None
        finally:
            del cn_copy
            del unet_copy
            gc.collect()
            torch.cuda.empty_cache()

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
            output_dir=self.output_dir, step=self.global_step, run_id=self.run_id,
            # label = epochs completed by the weights the samples come from (the selected checkpoint)
            epoch=self.best_epoch if self.best_epoch is not None else self.epoch + 1,
        )
        self.model.train()

    def _seg_weight(self):
        """Segmentation-loss weight λ for the current step: linear start → target over the warmup, then constant."""
        ramp = min(1.0, (self.global_step + 1) / max(1, config.SEGMENTOR_LOSS_WARMUP_STEPS))
        start = config.SEGMENTOR_LOSS_WEIGHT_START
        return start + (config.SEGMENTOR_LOSS_WEIGHT - start) * ramp

    def save_checkpoint(self, epochs_done=None):
        """Save LoRA-only checkpoint (~9MB incl. optimizer state). Without epochs_done: the selected
        checkpoint (overwrites the previous best); with it: an extra copy for that epoch."""
        if epochs_done is not None:
            lora_path = config.CHECKPOINT_DIR / f"{self.run_id}_ep{epochs_done}_LoRA.pt"
            torch.save({'controlnet_lora': {k: v for k, v in self.model.controlnet.state_dict().items() if 'lora_' in k},
                        'epochs_done': epochs_done, 'global_step': self.global_step,
                        'seg_weight': self._seg_weight() if config.USE_SEGMENTOR_LOSS else None}, lora_path)
            print(f"Epoch-{epochs_done} LoRA saved: {lora_path}")
            return
        lora_path = config.CHECKPOINT_DIR / f"{self.run_id}_LoRA.pt"
        checkpoint = {
            'controlnet_lora': {k: v for k, v in self.model.controlnet.state_dict().items() if 'lora_' in k},
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'global_step': self.global_step,
            'epochs_done': self.best_epoch,
            'best_val_loss': float(self.best_val_loss),
            'best_kid': float(self.best_kid),
            'checkpoint_metric': config.CHECKPOINT_METRIC,
        }
        if config.USE_UNET_LORA:
            checkpoint['unet_lora'] = {k: v for k, v in self.model.unet.state_dict().items() if 'lora_' in k}
        torch.save(checkpoint, lora_path)
        print(f"★ BEST MODEL saved (epoch {self.best_epoch}, val_loss={self.best_val_loss:.6f}, "
              f"kid={self.best_kid:.5f}): {lora_path}")

    def save_merged_controlnet(self):
        """Save merged ControlNet (base + LoRA baked in) after merge_and_unload(). ~1.4GB."""
        merged_path = config.CHECKPOINT_DIR / f"{self.run_id}_Full"
        merged_path.mkdir(parents=True, exist_ok=True)
        self.pipeline.controlnet.save_pretrained(merged_path)
        print(f"★ MERGED CONTROLNET saved: {merged_path}")


def main():
    """Main entry point."""
    print("=" * 70)
    print("CONTROLNET TRAINING - PHASE 1: DIFFUSION ONLY")
    print("=" * 70)

    # Reproducibility
    random.seed(config.SEED)
    np.random.seed(config.SEED)
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
    if config.INIT_LORA_CHECKPOINT:
        print(f"Starting from LoRA checkpoint: {config.INIT_LORA_CHECKPOINT}")
        model.load_checkpoint(config.INIT_LORA_CHECKPOINT)

    # Train
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader, device=device,
    )
    trainer.train()


if __name__ == "__main__":
    main()
