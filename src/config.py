"""
Configuration for ControlNet + Stable Diffusion 1.5 training.
All hyperparameters and paths defined here for easy HPC job setup.
"""

import os
from pathlib import Path
from .utils import setup_paths_from_env, create_directory_structure

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

# Model precision
DTYPE = "fp16"  # Options: "fp32", "fp16", "bf16"
USE_XFORMERS = True  # Memory efficient attention

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
# Basic training
LEARNING_RATE = 1e-5
BATCH_SIZE = 1  # Increase if GPU memory allows
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
NUM_EPOCHS = 20
MAX_TRAIN_STEPS = None  # If set, overrides NUM_EPOCHS

# Optimizer
OPTIMIZER = "adamw"  # Options: "adamw", "adam", "sgd"
WEIGHT_DECAY = 0.01
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8

# Learning rate scheduler
LR_SCHEDULER = "constant_with_warmup"  # Options: "constant", "constant_with_warmup", "cosine", "linear"
LR_WARMUP_STEPS = 500

# Mixed precision
MIXED_PRECISION = "fp16"  # Options: "no", "fp16", "bf16"

# Gradient settings
MAX_GRAD_NORM = 1.0

# ============================================================================
# DIFFUSION PARAMETERS
# ============================================================================
# Inference/sampling
NUM_INFERENCE_STEPS = 20  # DDIM steps (20-30 as per doc)
GUIDANCE_SCALE = 6.0  # CFG scale (5-7 as per doc to reduce artifacts)
CONTROLNET_CONDITIONING_SCALE = 1.0  # 0.8-1.2 as per doc

# Noise scheduler
NOISE_SCHEDULER = "ddpm"  # Options: "ddpm", "ddim", "pndm"

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
# Image/Mask settings
RESOLUTION = 512  # SD 1.5 native resolution
NUM_CLASSES = 35  # Total classes in segmentation (0-34)

# Data loading
NUM_WORKERS = 4
PREFETCH_FACTOR = 2
PIN_MEMORY = True

# Data augmentation (to be implemented)
USE_MASK_AUGMENTATION = False  # Jitter, dilate/erode, elastic warp, occlusions
MASK_JITTER_PROB = 0.0
MASK_DILATE_ERODE_PROB = 0.0
MASK_ELASTIC_WARP_PROB = 0.0
MASK_OCCLUSION_PROB = 0.0

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
VALIDATION_STEPS = 500
SAVE_STEPS = 1000

# Wandb (optional)
USE_WANDB = False
WANDB_PROJECT = "controlnet-sidewalk-segmentation"
WANDB_ENTITY = None  # Your wandb username/team

# Checkpointing
SAVE_TOTAL_LIMIT = 3  # Keep only last N checkpoints
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
# VALIDATION & EVALUATION
# ============================================================================
# Number of samples to generate during validation
NUM_VALIDATION_SAMPLES = 4
VALIDATION_PROMPT = "urban street scene, noon bright sunlight, clear weather, phone photo"

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
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Gradient Accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print("=" * 80)
