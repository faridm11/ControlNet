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
from typing import Optional, Dict, Any
import os

# Import config from parent package
from .. import config
import os

# Try to import PEFT for LoRA
try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("⚠ PEFT not available. LoRA training will be disabled.")

# Color palette for segmentation mask visualization (matches dataset.py)
PALETTE = torch.tensor([
    [0, 0, 0],        # 0: background (void, sky, buildings)
    [128, 64, 128],   # 1: road
    [244, 35, 232],   # 2: walkable (sidewalk, crosswalk, etc)
    [220, 20, 60],    # 3: pedestrian
    [0, 0, 142],      # 4: vehicle
    [220, 220, 0],    # 5: traffic control
    [190, 153, 153],  # 6: obstacle
    [107, 142, 35],   # 7: environment
], dtype=torch.float32) / 255.0 

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
        
        # 4. Load UNet (Denoising network - selective unfreezing)
        print(f"\n[4/5] Loading UNet from {config.SD_MODEL_ID}...")
        self.unet = UNet2DConditionModel.from_pretrained(
            config.SD_MODEL_ID,
            subfolder="unet"
        )
        self.unet.to(device=device)  # Keep in FP32 for mixed precision
        
        # Freeze entire UNet first
        self.unet.requires_grad_(False)
        
        # Unfreeze only late blocks for texture/realism learning
        print("  Unfreezing late UNet blocks for texture learning...")
        
        # Unfreeze last up_block (texture & realism)
        for param in self.unet.up_blocks[-1].parameters():
            param.requires_grad = True
        print(f"  ✓ Unfroze up_blocks[-1] ({sum(p.numel() for p in self.unet.up_blocks[-1].parameters()):,} params)")
        
        # Optionally unfreeze mid_block
        if config.UNFREEZE_MID_BLOCK:
            for param in self.unet.mid_block.parameters():
                param.requires_grad = True
            print(f"  ✓ Unfroze mid_block ({sum(p.numel() for p in self.unet.mid_block.parameters()):,} params)")
        
        unet_trainable = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        print(f"  ✓ Total UNet trainable params: {unet_trainable:,}")
        print("✓ UNet loaded with selective unfreezing (late blocks only)")
        
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
        
        # Apply LoRA to ControlNet if enabled (attention + conv_in)
        if config.USE_LORA and PEFT_AVAILABLE:
            print("  Applying LoRA to ControlNet (attention + conv_in)...")
            controlnet_lora_config = LoraConfig(
                r=config.CONTROLNET_LORA_RANK,
                lora_alpha=config.CONTROLNET_LORA_ALPHA,
                target_modules=config.CONTROLNET_LORA_TARGET_MODULES,
                lora_dropout=config.CONTROLNET_LORA_DROPOUT,
                bias="none",
            )
            self.controlnet = get_peft_model(self.controlnet, controlnet_lora_config)
            print(f"  ✓ ControlNet LoRA applied (rank={config.CONTROLNET_LORA_RANK}, alpha={config.CONTROLNET_LORA_ALPHA})")
        else:
            self.controlnet.requires_grad_(True)  # Full fine-tuning
            print("✓ ControlNet will be fine-tuned (no LoRA)")
        
        # Enable gradient checkpointing
        print("\nEnabling gradient checkpointing...")
        if hasattr(self.controlnet, 'enable_gradient_checkpointing'):
            self.controlnet.enable_gradient_checkpointing()
        # Enable for UNet (not wrapped in PEFT)
        if hasattr(self.unet, 'enable_gradient_checkpointing'):
            self.unet.enable_gradient_checkpointing()
        print("✓ Gradient checkpointing enabled")
        
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
        print(f"  - UNet (late blocks): {unet_trainable:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")
        print(f"  - VAE: frozen")
        print(f"  - Text Encoder: frozen")
        print(f"  - UNet (early/mid blocks): frozen")
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
            # Scale latents by VAE scaling factor (as per SD implementation)
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
        Converts class IDs to RGB visualization using color palette.
        
        Args:
            masks: Tensor of shape (B, H, W) with class IDs (0-7)
            
        Returns:
            Conditioning tensor of shape (B, 3, H, W), range [-1, 1]
        """
        batch_size, height, width = masks.shape
        
        # Use color palette to convert class IDs to RGB (matches dataset.py)
        palette = PALETTE.to(self.device)
        
        # Flatten masks for indexing: (B, H, W) -> (B*H*W,)
        masks_flat = masks.view(-1).long()
        
        # Map class IDs to RGB colors: (B*H*W, 3)
        mask_rgb_flat = palette[masks_flat]
        
        # Reshape back to (B, H, W, 3) then permute to (B, 3, H, W)
        mask_rgb = mask_rgb_flat.view(batch_size, height, width, 3).permute(0, 3, 1, 2)
        
        # Normalize to [-1, 1] to match pipeline preprocessing
        mask_rgb = mask_rgb * 2.0 - 1.0
        
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
        """Get all trainable parameters (ControlNet only - UNet frozen)."""
        return self.controlnet.parameters()
    
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
    
    # Unwrap PEFT models - pipeline needs raw ControlNetModel
    controlnet = model.controlnet
    unet = model.unet
    
    if hasattr(controlnet, 'get_base_model'):
        print("Unwrapping PEFT ControlNet for pipeline...")
        controlnet = controlnet.get_base_model()
    
    if hasattr(unet, 'get_base_model'):
        print("Unwrapping PEFT UNet for pipeline...")
        unet = unet.get_base_model()
    
    print("CONTROLNET TYPE:", type(controlnet))
    # The LoRA is not applied to UNet yet,but if it were, we would need to unwrap it as well.
    print("UNET TYPE:", type(unet))
    
    pipeline = StableDiffusionControlNetPipeline(
        vae=model.vae,
        text_encoder=model.text_encoder,
        tokenizer=model.tokenizer,
        unet=unet,
        controlnet=controlnet,
        scheduler=DDIMScheduler.from_pretrained(config.SD_MODEL_ID, subfolder="scheduler"),
        safety_checker=None,  # Disable safety checker
        feature_extractor=None,
    )
    pipeline.to(device)
    return pipeline
