"""
ControlNet + Stable Diffusion 1.5 Architecture with LoRA
Loads pretrained SD 1.5 and ControlNet with optional LoRA for efficient fine-tuning.
"""

import torch
import torch.nn as nn
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Optional
import os

from .. import config
from ..losses import diffusion_loss

# Try to import PEFT for LoRA
try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    raise ImportError("PEFT is required for LoRA training. Install it with: pip install peft")

class DiffusionControlNet(nn.Module):
    """
    Wrapper for Stable Diffusion 1.5 + ControlNet architecture.
    
    Components:
    - VAE: Encodes/decodes images to/from latent space
    - Text Encoder: CLIP model for text conditioning
    - UNet: Denoising network (will be fine-tuned)
    - ControlNet: Additional conditioning from segmentation masks
    - Scheduler: Noise scheduler for diffusion process
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        device: str = None,
    ):
        """
        Initialize ControlNet + SD 1.5 architecture.
        
        Args:
            pretrained: Load pretrained weights
            device: Device to load models on (auto-detect if None)
        """
        super().__init__()
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.resolution = config.RESOLUTION
        self.num_classes = config.NUM_CLASSES
        
        print("=" * 60)
        print("INITIALIZING CONTROLNET + STABLE DIFFUSION 1.5")
        print("=" * 60)
        
        # 1. Load VAE (Variational Autoencoder)
        print(f"\n[1/5] Loading VAE from {config.SD_MODEL_ID}...")
        self.vae = AutoencoderKL.from_pretrained(
            config.SD_MODEL_ID,
            subfolder="vae"
        )
        self.vae.to(device=device)  # Keep in FP32 for mixed precision
        self.vae.requires_grad_(False)  # Freeze VAE
        print("✓ VAE loaded and frozen")
        
        # 2. Load Text Encoder (CLIP)
        print(f"\n[2/5] Loading text encoder from {config.SD_MODEL_ID}...")
        self.text_encoder = CLIPTextModel.from_pretrained(
            config.SD_MODEL_ID,
            subfolder="text_encoder"
        )
        self.text_encoder.to(device=device)  # Keep in FP32 for mixed precision
        self.text_encoder.requires_grad_(False)  # Freeze text encoder
        print("✓ Text encoder loaded and frozen")
        
        # 3. Load Tokenizer
        print(f"\n[3/5] Loading tokenizer from {config.SD_MODEL_ID}...")
        self.tokenizer = CLIPTokenizer.from_pretrained(
            config.SD_MODEL_ID,
            subfolder="tokenizer"
        )
        print("✓ Tokenizer loaded")
        
        # 4. Load UNet (Denoising network — fully frozen, LoRA applied below)
        print(f"\n[4/5] Loading UNet from {config.SD_MODEL_ID}...")
        self.unet = UNet2DConditionModel.from_pretrained(
            config.SD_MODEL_ID,
            subfolder="unet"
        )
        self.unet.to(device=device)
        self.unet.requires_grad_(False)  # Freeze entirely — LoRA adapters added below
        print("✓ UNet loaded and frozen (LoRA will be applied after ControlNet)")
        
        # 5. Load ControlNet (Segmentation conditioning)
        print(f"\n[5/5] Loading ControlNet from {config.CONTROLNET_MODEL_ID}...")
        if pretrained:
            self.controlnet = ControlNetModel.from_pretrained(
                config.CONTROLNET_MODEL_ID
            )
            print("✓ ControlNet loaded with pretrained weights")
        else:
            # Initialize from UNet structure (for training from scratch)
            self.controlnet = ControlNetModel.from_unet(self.unet)
            print("✓ ControlNet initialized from UNet (no pretrained weights)")
        
        self.controlnet.to(device=device)  # Keep in FP32 for mixed precision
        
        # Apply LoRA to ControlNet (attention + conv_in for new color adaptation)
        if config.USE_LORA:
            print("  Applying LoRA to ControlNet...")
            controlnet_lora_config = LoraConfig(
                r=config.CONTROLNET_LORA_RANK,
                lora_alpha=config.CONTROLNET_LORA_ALPHA,
                target_modules=config.CONTROLNET_LORA_TARGET_MODULES,
                lora_dropout=config.CONTROLNET_LORA_DROPOUT,
                bias="none",
            )
            self.controlnet = get_peft_model(self.controlnet, controlnet_lora_config)
            print(f"  ✓ ControlNet LoRA (rank={config.CONTROLNET_LORA_RANK}, alpha={config.CONTROLNET_LORA_ALPHA}, dropout={config.CONTROLNET_LORA_DROPOUT})")
        else:
            raise ValueError("USE_LORA=False — full ControlNet fine-tuning is not allowed. Set USE_LORA=True in config.")

        # Apply LoRA to UNet (attention layers only — no conv_in, UNet processes latents not colors)
        if config.USE_UNET_LORA:
            print("  Applying LoRA to UNet...")
            unet_lora_config = LoraConfig(
                r=config.UNET_LORA_RANK,
                lora_alpha=config.UNET_LORA_ALPHA,
                target_modules=config.UNET_LORA_TARGET_MODULES,
                lora_dropout=config.UNET_LORA_DROPOUT,
                bias="none",
            )
            self.unet = get_peft_model(self.unet, unet_lora_config)
            print(f"  ✓ UNet LoRA (rank={config.UNET_LORA_RANK}, alpha={config.UNET_LORA_ALPHA}, dropout={config.UNET_LORA_DROPOUT})")
        else:
            print("✓ UNet remains frozen (USE_UNET_LORA=False)")
        
        # Gradient checkpointing DISABLED — breaks LoRA gradient flow.
        # PyTorch checkpoint() skips backward for blocks whose inputs don't require_grad.
        # With frozen base weights, deeper LoRA layers get zero gradients.
        # A100 has 40GB and we use ~8GB, so memory is not a concern.
        print("\nGradient checkpointing: DISABLED (not needed, breaks LoRA grad flow)")
        
        # 6. Load Noise Scheduler
        print("\nLoading noise scheduler...")
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            config.SD_MODEL_ID,
            subfolder="scheduler"
        )
        print("✓ DDPM scheduler loaded")
        
        print("\n" + "=" * 60)
        print("MODEL INITIALIZATION COMPLETE")
        print("=" * 60)
        self._print_model_info()
    
    def _print_model_info(self):
        """Print model information and parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        controlnet_trainable = sum(p.numel() for p in self.controlnet.parameters() if p.requires_grad)
        unet_trainable = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"  - ControlNet LoRA: {controlnet_trainable:,}")
        print(f"  - UNet LoRA: {unet_trainable:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")
        print(f"  - VAE: frozen")
        print(f"  - Text Encoder: frozen")
        print(f"  - UNet base weights: frozen (only LoRA adapters train)")
        print(f"  - ControlNet base weights: frozen (only LoRA adapters train)")
        print(f"\nMemory footprint (approx): {total_params * 4 / 1e9:.2f} GB (fp32)")
        print(f"Trainable memory (approx): {trainable_params * 4 / 1e9:.2f} GB (fp32)")
    
    def encode_text(self, prompts: list):
        """
        Encode text prompts to embeddings.
        
        Args:
            prompts: List of text prompts
            
        Returns:
            Text embeddings tensor
        """
        # Tokenize
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids.to(self.device)

        # Encode
        # Force fp32 — CLIP text encoder is frozen and sensitive to fp16
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            text_embeddings = self.text_encoder(text_input_ids)[0]

        return text_embeddings
    
    def encode_images(self, images: torch.Tensor):
        """
        Encode images to latent space using VAE.
        
        Args:
            images: Tensor of shape (B, C, H, W), range [-1, 1]
            
        Returns:
            Latents tensor of shape (B, 4, H//8, W//8)
        """
        # Force fp32 — VAE is frozen and known to produce artifacts in fp16
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            dist = self.vae.encode(images.float()).latent_dist
            # eval: use mode (deterministic mean) — no stochastic VAE sampling noise
            # train: sample from the posterior for proper ELBO training signal
            latents = dist.mode() if not self.training else dist.sample()
            latents = latents * self.vae.config.scaling_factor
        return latents
    
    def decode_latents(self, latents: torch.Tensor):
        """
        Decode latents to images using VAE.
        
        Args:
            latents: Tensor of shape (B, 4, H//8, W//8) . 3 downsampling so /8
            
        Returns:
            Images tensor of shape (B, C, H, W), range [-1, 1]
        """
        latents = latents / self.vae.config.scaling_factor
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            images = self.vae.decode(latents.float()).sample
        return images
    
    def prepare_mask_conditioning(self, mask_rgb: torch.Tensor):
        """
        Prepare RGB segmentation masks for ControlNet conditioning.
        RGB masks already come from dataset with proper ADE20K colors.
        
        Args:
            mask_rgb: Tensor of shape (B, 3, H, W) with RGB colors, range [0, 1]
            
        Returns:
            Conditioning tensor of shape (B, 3, H, W), range [0, 1]
        """
        # Return unchanged - pipeline uses [0, 1] range (verified with forward hook in verify_pipeline_actual.py)
        # Both training and inference use [0, 1]

        # This is checked in the original lllyasviel/ControlNet code. tutorial_dataset.py also uses same logic
        return mask_rgb
    
    def forward(
        self,
        images: torch.Tensor,
        mask_rgb: torch.Tensor,
        prompts: list,
        timesteps: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for training.
        
        Args:
            images: Ground truth images (B, 3, H, W), range [-1, 1]
            mask_rgb: RGB segmentation masks (B, 3, H, W) with ADE20K colors, range [0, 1]
            prompts: List of text prompts
            timesteps: Optional timesteps (if None, will sample random)
            
        Returns:
            Dictionary with:
            - loss: Diffusion loss
            - noise_pred: Predicted noise
            - noise: Ground truth noise
        """
        batch_size = images.shape[0]

        # One-time range check (only on first forward pass)
        if not hasattr(self, '_ranges_verified'):
            assert images.min() >= -1.0 and images.max() <= 1.0, \
                f"Images out of range [-1, 1]: [{images.min():.3f}, {images.max():.3f}]"
            assert mask_rgb.min() >= 0.0 and mask_rgb.max() <= 1.0, \
                f"Mask RGB out of range [0, 1]: [{mask_rgb.min():.3f}, {mask_rgb.max():.3f}]"
            self._ranges_verified = True
        
        # 1. Encode images to latents
        latents = self.encode_images(images)
        
        # 2. Sample noise
        noise = torch.randn_like(latents)
        
        # 3. Sample timesteps
        if timesteps is None:
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (batch_size,), device=images.device
            ).long()
        
        # 4. Add noise to latents
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # 5. Text dropout for CFG training — 15% of samples get empty prompt.
        # Required so the model learns unconditional generation, which CFG needs at inference.
        # Only applied during training (self.training=True), never during validation/inference.
        if self.training and config.TEXT_DROPOUT_PROB > 0:
            prompts = [
                "" if torch.rand(1).item() < config.TEXT_DROPOUT_PROB else p
                for p in prompts
            ]

        text_embeddings = self.encode_text(prompts)
        
        # 6. Prepare mask conditioning (normalize RGB to [-1, 1])
        mask_conditioning = self.prepare_mask_conditioning(mask_rgb)
        
        # 7. ControlNet forward pass — no conditioning_scale here.
        # conditioning_scale is an inference-time knob (controls how strongly the
        # ControlNet guides the UNet). During training we always want full-strength
        # residuals (scale=1.0, the diffusers default) so gradients are clean.
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_embeddings,
            controlnet_cond=mask_conditioning,
            return_dict=False
        )
        
        # 8. UNet forward pass with ControlNet conditioning
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_embeddings,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        ).sample
        
        # 9. Compute loss
        loss = diffusion_loss(noise_pred, noise)
        
        return {
            "loss": loss,
            "noise_pred": noise_pred,
            "noise": noise,
        }
    
    def get_trainable_parameters(self):
        """Get all trainable parameters (ControlNet LoRA + UNet LoRA adapters)."""
        return filter(lambda p: p.requires_grad, self.parameters())
    
    def save_checkpoint(self, save_path: str):
        """Save LoRA-only checkpoint (~5MB, not the full 5GB model)."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        lora_state = {
            "controlnet_lora": {k: v for k, v in self.controlnet.state_dict().items() if "lora_" in k},
        }
        if config.USE_UNET_LORA:
            lora_state["unet_lora"] = {k: v for k, v in self.unet.state_dict().items() if "lora_" in k}
        torch.save(lora_state, save_path)
        print(f"✓ LoRA checkpoint saved to {save_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load LoRA-only checkpoint back into the already-initialized model."""
        lora_state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        # Merge LoRA keys into current state dict (base weights stay unchanged)
        cn_state = self.controlnet.state_dict()
        cn_state.update(lora_state["controlnet_lora"])
        self.controlnet.load_state_dict(cn_state)
        if config.USE_UNET_LORA and "unet_lora" in lora_state:
            unet_state = self.unet.state_dict()
            unet_state.update(lora_state["unet_lora"])
            self.unet.load_state_dict(unet_state)
        print(f"✓ LoRA checkpoint loaded from {checkpoint_path}")


def create_pipeline(model: DiffusionControlNet, device: str = None):
    """
    Create inference pipeline from trained model.
    Called once after training is complete — merges LoRA weights in-place.
    Do NOT call this more than once (merge_and_unload is destructive).

    Args:
        model: Trained DiffusionControlNet model
        device: Device to run pipeline on (None = auto-detect)

    Returns:
        StableDiffusionControlNetPipeline for inference
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    controlnet = model.controlnet.merge_and_unload()
    unet = model.unet.merge_and_unload() if hasattr(model.unet, 'merge_and_unload') else model.unet

    pipeline = StableDiffusionControlNetPipeline(
        vae=model.vae,
        text_encoder=model.text_encoder,
        tokenizer=model.tokenizer,
        unet=unet,
        controlnet=controlnet,
        scheduler=DDIMScheduler.from_pretrained(config.SD_MODEL_ID, subfolder="scheduler"),
        safety_checker=None,
        feature_extractor=None,
    )
    pipeline.to(device)
    return pipeline

