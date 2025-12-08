"""
ControlNet + Stable Diffusion 1.5 Architecture
Loads pretrained SD 1.5 and ControlNet for segmentation conditioning.
No LoRA - pure fine-tuning for now.
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
from typing import Optional, Dict, Any
import os

# Import config from parent package
from .. import config
import os 

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
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize ControlNet + SD 1.5 architecture.
        
        Args:
            pretrained: Load pretrained weights
            device: Device to load models on (auto-detect if None)
            dtype: Model precision (float16/float32)
        """
        super().__init__()
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.dtype = dtype
        self.resolution = config.RESOLUTION
        self.num_classes = config.NUM_CLASSES
        
        print("=" * 60)
        print("INITIALIZING CONTROLNET + STABLE DIFFUSION 1.5")
        print("=" * 60)
        
        # 1. Load VAE (Variational Autoencoder)
        print(f"\n[1/5] Loading VAE from {config.SD_MODEL_ID}...")
        self.vae = AutoencoderKL.from_pretrained(
            config.SD_MODEL_ID,
            subfolder="vae",
            torch_dtype=dtype
        )
        self.vae.requires_grad_(False)  # Freeze VAE
        self.vae.to(device)
        print("✓ VAE loaded and frozen")
        
        # 2. Load Text Encoder (CLIP)
        print(f"\n[2/5] Loading text encoder from {config.SD_MODEL_ID}...")
        self.text_encoder = CLIPTextModel.from_pretrained(
            config.SD_MODEL_ID,
            subfolder="text_encoder",
            torch_dtype=dtype
        )
        self.text_encoder.requires_grad_(False)  # Freeze text encoder
        self.text_encoder.to(device)
        print("✓ Text encoder loaded and frozen")
        
        # 3. Load Tokenizer
        print(f"\n[3/5] Loading tokenizer from {config.SD_MODEL_ID}...")
        self.tokenizer = CLIPTokenizer.from_pretrained(
            config.SD_MODEL_ID,
            subfolder="tokenizer"
        )
        print("✓ Tokenizer loaded")
        
        # 4. Load UNet (Denoising network - will be fine-tuned)
        print(f"\n[4/5] Loading UNet from {config.SD_MODEL_ID}...")
        self.unet = UNet2DConditionModel.from_pretrained(
            config.SD_MODEL_ID,
            subfolder="unet",
            torch_dtype=dtype
        )
        self.unet.requires_grad_(True)  # UNet will be fine-tuned
        self.unet.to(device)
        print("✓ UNet loaded (trainable)")
        
        # 5. Load ControlNet (Segmentation conditioning)
        print(f"\n[5/5] Loading ControlNet from {config.CONTROLNET_MODEL_ID}...")
        if pretrained:
            self.controlnet = ControlNetModel.from_pretrained(
                config.CONTROLNET_MODEL_ID,
                torch_dtype=dtype
            )
            print("✓ ControlNet loaded with pretrained weights")
        else:
            # Initialize from UNet structure (for training from scratch)
            self.controlnet = ControlNetModel.from_unet(self.unet)
            print("✓ ControlNet initialized from UNet (no pretrained weights)")
        
        self.controlnet.requires_grad_(True)  # ControlNet will be fine-tuned
        self.controlnet.to(device)
        
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
        
        unet_params = sum(p.numel() for p in self.unet.parameters())
        controlnet_params = sum(p.numel() for p in self.controlnet.parameters())
        
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"  - UNet: {unet_params:,}")
        print(f"  - ControlNet: {controlnet_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")
        print(f"  - VAE: frozen")
        print(f"  - Text Encoder: frozen")
        print(f"\nMemory footprint (approx): {total_params * 2 / 1e9:.2f} GB (fp16)")
    
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
        with torch.no_grad():
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
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        return latents
    
    def decode_latents(self, latents: torch.Tensor):
        """
        Decode latents to images using VAE.
        
        Args:
            latents: Tensor of shape (B, 4, H//8, W//8)
            
        Returns:
            Images tensor of shape (B, C, H, W), range [-1, 1]
        """
        latents = latents / self.vae.config.scaling_factor
        with torch.no_grad():
            images = self.vae.decode(latents).sample
        return images
    
    def prepare_mask_conditioning(self, masks: torch.Tensor):
        """
        Prepare segmentation masks for ControlNet conditioning.
        Converts class IDs to RGB visualization for ControlNet input.
        
        Args:
            masks: Tensor of shape (B, H, W) with class IDs (0-34)
            
        Returns:
            Conditioning tensor of shape (B, 3, H, W), range [0, 1]
        """
        batch_size = masks.shape[0]
        height, width = masks.shape[1], masks.shape[2]
        
        # Create RGB visualization (simple: map class ID to grayscale)
        # ControlNet expects (B, 3, H, W) RGB input
        mask_rgb = torch.zeros((batch_size, 3, height, width), dtype=self.dtype, device=self.device)
        
        # Normalize class IDs to [0, 1] range for visualization
        # You can replace this with a proper color map later
        normalized = masks.float() / (self.num_classes - 1)
        mask_rgb[:, 0, :, :] = normalized  # R channel
        mask_rgb[:, 1, :, :] = normalized  # G channel
        mask_rgb[:, 2, :, :] = normalized  # B channel
        
        return mask_rgb
    
    def forward(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        prompts: list,
        timesteps: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for training.
        
        Args:
            images: Ground truth images (B, 3, H, W), range [-1, 1]
            masks: Segmentation masks (B, H, W) with class IDs
            prompts: List of text prompts
            timesteps: Optional timesteps (if None, will sample random)
            
        Returns:
            Dictionary with:
            - loss: Diffusion loss
            - noise_pred: Predicted noise
            - noise: Ground truth noise
        """
        batch_size = images.shape[0]
        
        # 1. Encode images to latents
        latents = self.encode_images(images)
        
        # 2. Sample noise
        noise = torch.randn_like(latents)
        
        # 3. Sample timesteps
        if timesteps is None:
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (batch_size,), device=self.device
            ).long()
        
        # 4. Add noise to latents
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # 5. Encode text prompts
        text_embeddings = self.encode_text(prompts)
        
        # 6. Prepare mask conditioning
        mask_conditioning = self.prepare_mask_conditioning(masks)
        
        # 7. ControlNet forward pass
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
        
        # 9. Compute loss (MSE between predicted and true noise)
        loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction="mean")
        
        return {
            "loss": loss,
            "noise_pred": noise_pred,
            "noise": noise,
        }
    
    def get_trainable_parameters(self):
        """Get all trainable parameters (UNet + ControlNet)."""
        params = []
        params.extend(self.unet.parameters())
        params.extend(self.controlnet.parameters())
        return params
    
    def save_checkpoint(self, save_path: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            "unet": self.unet.state_dict(),
            "controlnet": self.controlnet.state_dict(),
        }
        torch.save(checkpoint, save_path)
        print(f"✓ Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.unet.load_state_dict(checkpoint["unet"])
        self.controlnet.load_state_dict(checkpoint["controlnet"])
        print(f"✓ Checkpoint loaded from {checkpoint_path}")


def create_pipeline(model: DiffusionControlNet, device: str = None):
    """
    Create inference pipeline from trained model.
    
    Args:
        model: Trained DiffusionControlNet model
        device: Device to run pipeline on (None = auto-detect)
        
    Returns:
        StableDiffusionControlNetPipeline for inference
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    pipeline = StableDiffusionControlNetPipeline(
        vae=model.vae,
        text_encoder=model.text_encoder,
        tokenizer=model.tokenizer,
        unet=model.unet,
        controlnet=model.controlnet,
        scheduler=DDIMScheduler.from_pretrained(config.SD_MODEL_ID, subfolder="scheduler"),
        safety_checker=None,  # Disable safety checker
        feature_extractor=None,
    )
    pipeline.to(device)
    return pipeline
