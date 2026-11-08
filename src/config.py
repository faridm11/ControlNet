"""
Configuration for ControlNet + Stable Diffusion 1.5 training.
All hyperparameters and paths defined here for easy HPC job setup.
"""

import os
from pathlib import Path
from src.utils import setup_paths_from_env, create_directory_structure


def _env(name, default, cast=None):
    """Read an override from the environment variable DDPM_<name>, else use default.

    Lets job scripts sweep hyperparameters without editing this file, e.g.
        DDPM_CONTROLNET_LORA_LR=1e-4 DDPM_RUN_NAME=lr1e-4 sbatch jobs/train_phase1.sh
    """
    raw = os.environ.get(f"DDPM_{name}")
    if raw is None or raw == "":
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return (cast or type(default))(raw)

# Free-text tag for this run (printed and stored in the config snapshot).
RUN_NAME = _env("RUN_NAME", "")

# ============================================================================
# CLASS VERSION
# ============================================================================
# v1: 35 classes (20 ADE20K + 10 new-color + 5 void-white)
# v2: 21 classes (20 ADE20K + 1 void-white) — new-color classes merged into nearest ADE20K neighbor
# "sanpo" = dataset 2 (SANPO-Real): labels stay in SANPO's own 31 ids, each conditioned
#           with an ADE20K colour (data_prep/sanpo/class_mapping_sanpo.py).
# A data root can declare its label space in a LABEL_SPACE file (data_sanpo/ does);
# it then becomes the default. Override via env var: DDPM_CLASS_VERSION=v1 sbatch ...

# ============================================================================
# PROJECT PATHS (Auto-configured for HPC/Local)
# ============================================================================
# Use environment-aware path setup
_PATHS = setup_paths_from_env()

_LABEL_SPACE_FILE = _PATHS['data_root'] / "LABEL_SPACE"
_DATA_LABEL_SPACE = _LABEL_SPACE_FILE.read_text().strip() if _LABEL_SPACE_FILE.exists() else None
CLASS_VERSION = _env("CLASS_VERSION", _DATA_LABEL_SPACE or "v2")  # "v1" = 35, "v2" = 21 ADE20K-only, "sanpo" = 31
# Dataset name, put in front of every output file name (logs, checkpoints, plots, samples)
# so runs on different datasets cannot be confused. Dataset 1 keeps its old, unprefixed names.
DATASET_NAME = "sanpo" if _DATA_LABEL_SPACE == "sanpo" else "sidewalk"
DATASET_TAG = "" if DATASET_NAME == "sidewalk" else f"{DATASET_NAME}_"
# Safety net: a job can state which dataset it expects (DDPM_EXPECT_DATASET=sanpo|sidewalk);
# if the data root resolves to a different one, stop instead of silently training on it.
_EXPECT = os.environ.get("DDPM_EXPECT_DATASET")
if _EXPECT and _EXPECT != DATASET_NAME:
    raise ValueError(f"Job expects dataset '{_EXPECT}' but the data root {_PATHS['data_root']} is '{DATASET_NAME}'")
if (_DATA_LABEL_SPACE == "sanpo") != (CLASS_VERSION == "sanpo"):
    raise ValueError(f"Data root {_PATHS['data_root']} has label space {_DATA_LABEL_SPACE or 'v1 ids'} "
                     f"but CLASS_VERSION={CLASS_VERSION}")

PROJECT_ROOT = _PATHS['project_root']
DATA_DIR = _PATHS['data_root']

# Training data
TRAIN_IMAGES_DIR = DATA_DIR / "train" / "images"
TRAIN_MASKS_DIR = DATA_DIR / "train" / "labels"
# Prompt CSV file name inside <split>/prompts/ — override to compare prompt variants.
# Default per dataset: dataset 2 (SANPO) only has prompts_meta.csv (recorded weather/scene + object cues).
PROMPTS_FILE_NAME = _env("PROMPTS_FILE_NAME", "prompts_meta.csv" if _DATA_LABEL_SPACE == "sanpo" else "prompts.csv")
TRAIN_PROMPTS_FILE = DATA_DIR / "train" / "prompts" / PROMPTS_FILE_NAME  # same for v1/v2 — prompts are semantic

# Test/Validation data
VAL_IMAGES_DIR = DATA_DIR / "test" / "images"
VAL_MASKS_DIR = DATA_DIR / "test" / "labels"
VAL_PROMPTS_FILE = DATA_DIR / "test" / "prompts" / PROMPTS_FILE_NAME

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
CONTROLNET_LORA_RANK = _env("CONTROLNET_LORA_RANK", 8)
CONTROLNET_LORA_ALPHA = _env("CONTROLNET_LORA_ALPHA", 16)
CONTROLNET_LORA_DROPOUT = _env("CONTROLNET_LORA_DROPOUT", 0.1)  # Higher dropout for small dataset (~1k images)
CONTROLNET_LORA_TARGET_MODULES = [
    "to_q", "to_k", "to_v", "to_out.0",
    "controlnet_cond_embedding.conv_in",  # Maps RGB colors → features (new colors in v1; all ADE20K in v2)
]
# DDPM_LORA_TARGETS=attn_ff additionally adapts the transformer projections and feed-forward layers
if _env("LORA_TARGETS", "attn") == "attn_ff":
    CONTROLNET_LORA_TARGET_MODULES += ["proj_in", "proj_out", "ff.net.0.proj", "ff.net.2"]

# UNet LoRA — disabled: freeze UNet entirely, only ControlNet LoRA trains (matches HF reference)
USE_UNET_LORA = _env("USE_UNET_LORA", False)
UNET_LORA_RANK = _env("UNET_LORA_RANK", 8)
UNET_LORA_ALPHA = _env("UNET_LORA_ALPHA", 8)
UNET_LORA_DROPOUT = _env("UNET_LORA_DROPOUT", 0.1)
UNET_LORA_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]  # Not used if USE_UNET_LORA=False

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
# Basic training

CONTROLNET_LORA_LR = _env("CONTROLNET_LORA_LR", 1e-5)  # Conservative LR — preserve pretrained ControlNet knowledge
UNET_LR = _env("UNET_LR", 5e-6)          # Unused (USE_UNET_LORA=False)
BATCH_SIZE = _env("BATCH_SIZE", 2)  # Increase if GPU memory allows
GRADIENT_ACCUMULATION_STEPS = _env("GRADIENT_ACCUMULATION_STEPS", 4)
NUM_EPOCHS = _env("NUM_EPOCHS", 30)  # Increased from 25 to see if more training helps
MAX_TRAIN_STEPS = None  # If set, overrides NUM_EPOCHS

# Text dropout (classifier-free guidance training)
# 15% of samples get empty prompt "" so the model learns unconditional generation.
# Required for CFG to work at inference time.
TEXT_DROPOUT_PROB = _env("TEXT_DROPOUT_PROB", 0.15)

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
MIXED_PRECISION = _env("MIXED_PRECISION", "bf16")  # Options: "no", "fp16", "bf16"
# GPUs older than Ampere (V100, RTX 2080 Ti) have no native bf16: train.py then falls back
# to fp16 with gradient scaling (GradScaler) and says so in the log.

# Gradient settings
MAX_GRAD_NORM = 1.0

# ============================================================================
# DIFFUSION PARAMETERS
# ============================================================================
# Inference/sampling
NUM_INFERENCE_STEPS = 35  # DDIM steps (20-30 as per doc)
GUIDANCE_SCALE = _env("GUIDANCE_SCALE", 6.0)  # CFG scale (5-7 as per doc to reduce artifacts)
NEGATIVE_PROMPT = _env("NEGATIVE_PROMPT", "")  # e.g. "blurry, distorted, deformed, hazy, low contrast" ("" = none)
CONTROLNET_CONDITIONING_SCALE = _env("CONTROLNET_CONDITIONING_SCALE", 1.0)  # Reduced from 1.3 to give model more freedom

# Sample generation (for visualization during training)
NUM_SAMPLES_TO_GENERATE = 4  # Number of different prompts to visualize (0 to disable)
NUM_IMAGES_PER_PROMPT = 4  # Variations per prompt

# Noise scheduler
NOISE_SCHEDULER = "ddpm"  # Options: "ddpm", "ddim", "pndm"

# ============================================================================
# EVALUATION METRICS (final eval after training — never affect the loss)
# ============================================================================
# FID is unreliable at n=50 (rank-deficient Inception covariance) — kept only
# as a relative trend. KID is the unbiased small-sample alternative to trust.
# LPIPS/CLIP are per-sample (gen vs paired GT / prompt) → reliable at n=50.
EVAL_FID = True
EVAL_KID = True
EVAL_LPIPS = True
EVAL_CLIP = True
EVAL_MIOU = True  # segmentor mIoU of generated images vs the mask (skipped if segmentor files are missing)
# KID subset_size. Must be < number of eval samples (val set = 50): with subset == n
# every subset is the whole set and the reported KID std collapses to ~0.
EVAL_KID_SUBSET = _env("EVAL_KID_SUBSET", 25)

# Evaluate (validation loss + generation metrics) on a fixed subset of the test split.
# 0 = all test images. The subset is evenly spaced over the test list, which is sorted
# by recording session, so every session is represented. Downstream segmentor tests
# always use the full test split.
VAL_SUBSET_SIZE = _env("VAL_SUBSET_SIZE", 0)
EVAL_SAVE_DIR = _env("EVAL_SAVE_DIR", "")  # if set: evaluate() also saves every generated image there (same file name as the real photo)
EVAL_SEED = _env("EVAL_SEED", 42)  # base seed for generation in eval (batch i uses EVAL_SEED + i)
CLIP_MODEL_ID = "openai/clip-vit-base-patch16"  # cached in HF_HOME

# Periodic generation-based eval DURING training (to get KID/LPIPS curves).
# Every N epochs we build a throwaway pipeline from a deepcopy of the ControlNet
# (LoRA merged into the copy only — the live trainable model is never mutated),
# generate the val set, and log the metrics into the CSV so they can be plotted.
# Set 0 to disable (then metrics are computed only once at the end of training).
# Cost: ~1 full val-set generation per triggered epoch.
EVAL_EVERY_N_EPOCHS = _env("EVAL_EVERY_N_EPOCHS", 5)

# Run validation + generation eval once BEFORE any training step ("epoch 0").
# Normally LoRA-B is zero-initialised, so this measures the untouched pretrained
# ControlNet with exactly the same protocol as later epochs. With INIT_LORA_CHECKPOINT
# it measures that starting checkpoint instead (logged and plotted as "start of run").
EVAL_AT_START = _env("EVAL_AT_START", True)

# Validation loss is averaged over this many FIXED timesteps per image (spread
# evenly over the schedule) instead of one random-but-fixed timestep.
VAL_NUM_TIMESTEPS = 8

# Which metric picks the saved "best" LoRA checkpoint:
#   "kid"      — lowest KID from the periodic generation eval (default)
#   "val_loss" — lowest validation loss (nearly flat for ControlNet; weak signal)
#   "last"     — no selection: keep the weights of the last epoch
CHECKPOINT_METRIC = _env("CHECKPOINT_METRIC", "kid")
if CHECKPOINT_METRIC not in ("kid", "val_loss", "last"):
    raise ValueError(f"CHECKPOINT_METRIC must be 'kid', 'val_loss' or 'last', got '{CHECKPOINT_METRIC}'")

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
# Image/Mask settings
RESOLUTION = 512  # SD 1.5 native resolution
NUM_CLASSES = {"v1": 35, "v2": 21, "sanpo": 31}[CLASS_VERSION]  # conditioning classes

# Data loading
NUM_WORKERS = _env("NUM_WORKERS", 4)
PREFETCH_FACTOR = 2
PIN_MEMORY = False  # Disabled to save GPU memory

# ============================================================================
# MASK AUGMENTATION 
# ============================================================================
USE_MASK_AUGMENTATION = False  # Enable mask augmentation for training

# Horizontal flip of image AND mask together (training only). Prompts contain no
# left/right words, so the text stays correct; labels stay exact.
TRAIN_HFLIP = _env("TRAIN_HFLIP", False)

# Training-time prompt augmentation (src/data/dataset.py: augment_prompt). Each condition phrase and
# each object of the true prompt is kept with this probability, order shuffled. 1.0 = off.
# Evaluation always uses the full prompt.
TRAIN_PROMPT_KEEP_PROB = _env("TRAIN_PROMPT_KEEP_PROB", 1.0)

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
# Phase 2: L = L_diffusion + λ · L_seg, where L_seg compares the frozen segmentor's
# prediction on the decoded denoised image with the ground-truth mask.
# The segmentor itself is configured in src/segmentor.py.
USE_SEGMENTOR_LOSS = _env("USE_SEGMENTOR_LOSS", False)
SEGMENTOR_LOSS_WEIGHT = _env("SEGMENTOR_LOSS_WEIGHT", 0.1)   # λ after the ramp
SEGMENTOR_LOSS_WEIGHT_START = _env("SEGMENTOR_LOSS_WEIGHT_START", 0.0)  # λ at step 0
SEGMENTOR_LOSS_WARMUP_STEPS = _env("SEGMENTOR_LOSS_WARMUP_STEPS", 200)  # λ ramps linearly start → target
# The denoised estimate is only a recognisable image at low noise, so the seg loss
# is applied only to samples whose timestep is below this value.
SEGMENTOR_LOSS_MAX_TIMESTEP = _env("SEGMENTOR_LOSS_MAX_TIMESTEP", 400)
# Decoding through the VAE with gradients is memory-heavy: cap samples per batch.
SEGMENTOR_LOSS_MAX_SAMPLES = _env("SEGMENTOR_LOSS_MAX_SAMPLES", 2)

# Start training from an existing LoRA checkpoint (e.g. the best Phase-1 run). Rank must match.
INIT_LORA_CHECKPOINT = _env("INIT_LORA_CHECKPOINT", "")

# Also keep the LoRA weights of every evaluated epoch (<run>_ep<N>_LoRA.pt, small), e.g. when λ changes
# during training and any point of the curve may need a full evaluation later.
SAVE_EVAL_CHECKPOINTS = _env("SAVE_EVAL_CHECKPOINTS", False)

#===============================================
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
SEED = _env("SEED", 42)
OVERFIT_FIXED_SEED = False  # Reset seed every epoch — only for overfit sanity checks

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
