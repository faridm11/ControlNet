"""
Configuration for ControlNet + Stable Diffusion 1.5 training.
All hyperparameters and paths defined here for easy HPC job setup.
"""

import os
from pathlib import Path
from src.utils import setup_paths_from_env, create_directory_structure

# ============================================================================
# PROJECT PATHS (Auto-configured for HPC/Local)
# ============================================================================
# Use environment-aware path setup
_PATHS = setup_paths_from_env()

PROJECT_ROOT = _PATHS['project_root']
DATA_DIR = _PATHS['data_root']

# Training data
TRAIN_IMAGES_DIR = DATA_DIR / "train" / "images"
TRAIN_MASKS_DIR = DATA_DIR / "train" / "labels"
TRAIN_PROMPTS_FILE = DATA_DIR / "train" / "prompts" / "prompts.csv"

# Test/Validation data
VAL_IMAGES_DIR = DATA_DIR / "test" / "images"
VAL_MASKS_DIR = DATA_DIR / "test" / "labels"
VAL_PROMPTS_FILE = DATA_DIR / "test" / "prompts" / "prompts.csv"

OUTPUT_DIR = _PATHS['output_root']
CHECKPOINT_DIR = _PATHS['checkpoint_dir']

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Base Stable Diffusion model
SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"
SD_REVISION = "main"

# ControlNet pretrained weights
CONTROLNET_MODEL_ID = "lllyasviel/control_v11p_sd15_seg"  # Pretrained for segmentation
CONTROLNET_REVISION = "main"

USE_XFORMERS = True  # Memory efficient attention

# ============================================================================
# LORA CONFIGURATION
# ============================================================================
USE_LORA = True  # Enable LoRA for ControlNet

# ControlNet LoRA (mandatory)
CONTROLNET_LORA_RANK = 16
CONTROLNET_LORA_ALPHA = 16
CONTROLNET_LORA_DROPOUT = 0.1  # Higher dropout for small dataset (~1k images)
CONTROLNET_LORA_TARGET_MODULES = [
    "to_q", "to_k", "to_v", "to_out.0",
    "conv_in",  # Required: maps new RGB colors → features; 14/35 classes use new colors
]

# UNet LoRA — disabled: freeze UNet entirely, only ControlNet LoRA trains (matches HF reference)
USE_UNET_LORA = False
UNET_LORA_RANK = 8
UNET_LORA_ALPHA = 8
UNET_LORA_DROPOUT = 0.1
UNET_LORA_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]  # Not used if USE_UNET_LORA=False

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
# Basic training
CONTROLNET_LORA_LR = 1e-5  # Conservative LR — preserve pretrained ControlNet knowledge
UNET_LR = 5e-6              # Unused (USE_UNET_LORA=False)
BATCH_SIZE = 2  # Increase if GPU memory allows
GRADIENT_ACCUMULATION_STEPS = 4 
NUM_EPOCHS = 30  # Increased from 25 to see if more training helps
MAX_TRAIN_STEPS = None  # If set, overrides NUM_EPOCHS

# Text dropout (classifier-free guidance training)
# 15% of samples get empty prompt "" so the model learns unconditional generation.
# Required for CFG to work at inference time.
TEXT_DROPOUT_PROB = 0.15

# Optimizer
OPTIMIZER = "adamw"  # Options: "adamw", "adam", "sgd" (will use AdamW8bit)
WEIGHT_DECAY = 1e-5       # Low weight decay — LoRA adapters are near pretrained distribution
UNET_WEIGHT_DECAY = 1e-5  # Unused (USE_UNET_LORA=False)
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8

# Learning rate scheduler
# cosine: ramps up linearly over LR_WARMUP_STEPS, then decays smoothly to ~0 at end
# of training.  This avoids the cold-start spike (constant hit pretrained weights at
# full LR immediately) and lets the model converge rather than oscillate at late epochs.
LR_SCHEDULER = "cosine"
LR_WARMUP_STEPS = 100  # ~1 epoch (118 steps/epoch) — gentle linear ramp before peak LR

# Mixed precision — bf16 preferred on A100:
#   - 8 exponent bits vs fp16's 5 → no gradient underflow in deep LoRA layers
#   - Same A100 tensor-core speed as fp16
#   - No GradScaler needed (bf16 dynamic range matches fp32)
MIXED_PRECISION = "bf16"  # Options: "no", "fp16", "bf16"

# Gradient settings
MAX_GRAD_NORM = 1.0

# ============================================================================
# DIFFUSION PARAMETERS
# ============================================================================
# Inference/sampling
NUM_INFERENCE_STEPS = 35  # DDIM steps (20-30 as per doc)
GUIDANCE_SCALE = 6.0  # CFG scale (5-7 as per doc to reduce artifacts)
CONTROLNET_CONDITIONING_SCALE = 1.0  # Reduced from 1.3 to give model more freedom

# Sample generation (for visualization during training)
NUM_SAMPLES_TO_GENERATE = 4  # Number of different prompts to visualize (0 to disable)
NUM_IMAGES_PER_PROMPT = 4  # Variations per prompt

# Noise scheduler
NOISE_SCHEDULER = "ddpm"  # Options: "ddpm", "ddim", "pndm"

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
# Image/Mask settings
RESOLUTION = 512  # SD 1.5 native resolution
NUM_CLASSES = 35  # All 35 original classes (21 use ADE20K colors, 14 use new unique colors)

# Data loading
NUM_WORKERS = 4
PREFETCH_FACTOR = 2
PIN_MEMORY = False  # Disabled to save GPU memory

# ============================================================================
# MASK AUGMENTATION (applied during training only)
# ============================================================================
USE_MASK_AUGMENTATION = False  # Enable mask augmentation for training

# Augmentation probabilities - DISABLED for now to avoid train/eval mismatch
# Set to 0.0 but keep parameters so they can be re-enabled later
MASK_JITTER_PROB = 0.0  # Disabled
MASK_DILATE_ERODE_PROB = 0.0  # Disabled due to spatial misalignment issues
MASK_ELASTIC_PROB = 0.0  # Disabled
MASK_OCCLUSION_PROB = 0.0  # Disabled
    
# Augmentation parameters
MASK_JITTER_PIXELS = 1
MASK_MORPH_KERNEL_SIZE = 2 # Around 1 pixel dilation/erosion (in 512x512) to simulate annotation uncertainty without drastic changes
MASK_ELASTIC_ALPHA = 5.0
MASK_ELASTIC_SIGMA = 2.0
MASK_OCCLUSION_PATCHES = 1
MASK_OCCLUSION_SIZE = 8

# ============================================================================
# SEGMENTOR-IN-THE-LOOP (to be implemented)
# ============================================================================
USE_SEGMENTOR_LOSS = False  # Semantic consistency loss
SEGMENTOR_CHECKPOINT = None  # Path to frozen segmentor model
SEGMENTOR_LOSS_WEIGHT = 0.1  # Lambda (λ) - ramp 0.1 → 0.5 as per doc

# ============================================================================
# ============================================================================
# LOGGING & CHECKPOINTING
# ============================================================================
# Logging
LOG_DIR = _PATHS['log_dir']
LOGGING_STEPS = 50
VALIDATION_STEPS = 250  # ~2 epochs with 1k images, batch 2, accum 4 (125 steps/epoch)
SAVE_STEPS = 500

# Wandb (optional)
USE_WANDB = False
WANDB_PROJECT = "controlnet-sidewalk-segmentation"
WANDB_ENTITY = None  # Your wandb username/team

# Checkpointing
SAVE_TOTAL_LIMIT = 0 # Keep only best models (best_run, best_fid, best_miou, best_overall)
RESUME_FROM_CHECKPOINT = None  # Path to checkpoint to resume from

# ============================================================================
# HPC CONFIGURATION
# ============================================================================
# Distributed training (for multi-GPU)
DISTRIBUTED = False
WORLD_SIZE = 1
LOCAL_RANK = -1

# Device
DEVICE = "cuda"  # Will be auto-detected

# Reproducibility
SEED = 42

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def create_output_dirs():
    """Create necessary output directories."""
    create_directory_structure(_PATHS)
    
def get_config_dict():
    """Return all config as dictionary for logging."""
    return {
        k: v for k, v in globals().items() 
        if not k.startswith('_') and k.isupper()
    }

def print_config():
    """Print current configuration for verification."""
    print("=" * 80)
    print("DDPM Training Configuration")
    print("=" * 80)
    print(f"Environment: {_PATHS['environment'].upper()}")
    print(f"\nPaths:")
    print(f"  Project Root: {PROJECT_ROOT}")
    print(f"  Data Dir: {DATA_DIR}")
    print(f"  Output Dir: {OUTPUT_DIR}")
    print(f"  Checkpoints: {CHECKPOINT_DIR}")
    print(f"  Logs: {LOG_DIR}")
    print(f"\nModel:")
    print(f"  SD Model: {SD_MODEL_ID}")
    print(f"  ControlNet: {CONTROLNET_MODEL_ID}")
    print(f"  Resolution: {RESOLUTION}")
    print(f"\nTraining:")
    print(f"  ControlNet LR: {CONTROLNET_LORA_LR}")
    print(f"  UNet LR: {UNET_LR}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Gradient Accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print("=" * 80)
