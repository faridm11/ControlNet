# ControlNet Fine-tuning for Urban Scene Generation

Fine-tuning a pretrained ControlNet (SD 1.5) with LoRA to generate photorealistic urban/sidewalk images conditioned on semantic segmentation masks.

---

## Overview

The model takes a **semantic segmentation mask** (color-coded, 35 classes) and a **text prompt** as input, and generates a realistic urban scene image that matches the spatial layout described by the mask.

**Base models:**
- Stable Diffusion 1.5 — `runwayml/stable-diffusion-v1-5`
- ControlNet segmentation — `lllyasviel/control_v11p_sd15_seg`

**Training strategy:** LoRA fine-tuning on ControlNet only. UNet stays fully frozen (matches HF reference). VAE and text encoder are also frozen throughout.

---

## Project Structure

```
DDPM/
├── src/
│   ├── train.py                      # Main training loop (Trainer class + main())
│   ├── config.py                     # All hyperparameters and paths
│   ├── losses.py                     # Diffusion MSE loss (Phase 1), seg loss (Phase 2)
│   ├── plotting.py                   # Loss / FID / grad norm curve plots
│   ├── sampling.py                   # Sample image grids after training
│   ├── evaluation.py                 # FID + mIoU evaluation
│   ├── models/
│   │   └── diffusion_controlnet.py   # Full model wrapper (VAE, CLIP, UNet, ControlNet)
│   ├── data/
│   │   ├── dataset.py                # ControlNetDataset, DataLoader, train/val helpers
│   │   └── transforms.py             # Mask augmentation (jitter, morph, elastic, occlusion)
│   └── testing/
│       ├── overfit_8_samples.py      # Overfit sanity check (8 samples, 200 epochs)
│       └── overfit_rtx3080.sh        # SLURM job for overfit test (A100)
├── data_prep/
│   ├── class_mapping.py              # 35-class color palette — single source of truth
│   ├── generate_prompts.py           # Auto-generate prompts from mask class statistics
│   └── prompt_templates.py           # Prompt templates per scene type
├── data/
│   ├── train/
│   │   ├── images/                   # ~950 RGB training images (not in git)
│   │   ├── labels/                   # Segmentation masks (not in git)
│   │   └── prompts/prompts.csv       # Pre-generated text prompts
│   └── test/
│       ├── images/                   # ~50 RGB validation images (not in git)
│       ├── labels/                   # (not in git)
│       └── prompts/prompts.csv
├── jobs/
│   └── train_phase1.sh               # SLURM job script (A100 partition, 8h)
├── inference.py                      # Standalone inference script
└── outputs/
    ├── checkpoints/                  # <jobid>_LoRA.pt + <jobid>_Full/ saved here
    ├── logs/                         # metrics CSV + SLURM .out / .err files
    ├── samples/                      # Generated image grids (end of training)
    └── plots/                        # PNG plots (loss, grad norm)
```

---

## Dataset

| Split | Images | Description |
|-------|--------|-------------|
| Train | ~950   | Urban/sidewalk scenes with paired segmentation masks and text prompts |
| Val   | ~50    | Held-out set for loss monitoring |

### 35 Semantic Classes

| Category | Classes |
|---|---|
| Flat surfaces | road, sidewalk, crosswalk, cycling_lane, parking_driveway, railtrack, curb |
| Humans | person, rider |
| Vehicles | car, truck, bus, tramtrain, motorbike, bicycle, caravan, trailer |
| Construction | building, door, wall, fence, bridge, tunnel, stairs |
| Objects | pole, sign, traffic_light |
| Nature | vegetation, earth, sky, field |
| Void | unlabeled, dynamic_void, static_void, unclear_void |

**Color palette:** 21 classes reuse original ADE20K colors (pretrained ControlNet already understands these). 14 classes use new unique colors chosen to be visually distinct from all ADE20K entries. All colors defined in `data_prep/class_mapping.py` — single source of truth.

---

## Training

### Phase 1 — Diffusion loss only

Standard DDPM epsilon-prediction loss: MSE between the noise predicted by the UNet and the actual noise added to the latent. The ControlNet LoRA adapts the pretrained segmentation ControlNet to the custom 35-class palette and urban domain.

**Hyperparameters (`src/config.py`):**

| Parameter | Value | Notes |
|---|---|---|
| LoRA rank / alpha | 16 / 16 | Scale = alpha/rank = 1.0 |
| LoRA target modules | `to_q, to_k, to_v, to_out.0, controlnet_cond_embedding.conv_in` | `conv_in` targets only the conditioning input, not the latent input |
| LoRA dropout | 0.1 | Regularization for small dataset |
| Learning rate | 1e-5 | Conservative — keeps LoRA adapters close to pretrained init |
| LR schedule | Cosine + 100 warmup steps | Linear ramp avoids cold-start spike |
| Batch size | 2 × 4 grad accum = 8 effective | A100 memory budget |
| Epochs | 30 | ~3540 optimizer steps total |
| Mixed precision | bf16 | A100 native; no GradScaler needed |
| Text dropout | 15% | Classifier-free guidance training |
| UNet | Fully frozen | Avoids moving-target problem; matches HF reference |

### Phase 2 — Segmentation consistency loss (planned)

Adds a frozen segmentor in the loop: generate an image from the mask, run the segmentor on it, and penalize the model if the output mask diverges from the input. Controlled by `USE_SEGMENTOR_LOSS` in config (currently `False`).

---

## Running on HPC (SLURM)

```bash
# Submit full training job (A100, 8h)
sbatch jobs/train_phase1.sh

# Monitor live
tail -f outputs/logs/train_phase1_<JOB_ID>.out

# Overfit sanity check (8 samples, 200 epochs — verifies gradient flow)
sbatch src/testing/overfit_rtx3080.sh
```

The job script sets `DDPM_DATA_ROOT` and `DDPM_OUTPUT_ROOT`. Paths resolve automatically in `src/config.py` via `setup_paths_from_env()` — no code changes needed between local and HPC runs.

### Running locally

```bash
conda activate control2
export DDPM_DATA_ROOT=$(pwd)/data
export DDPM_OUTPUT_ROOT=$(pwd)/outputs
python -m src.train
```

---

## Outputs

**Per epoch (during training):**
- `outputs/logs/metrics_<JOB_ID>.csv` — step, epoch, train_loss, val_loss, lr, grad_norm
- `outputs/plots/loss_curve.png` — smoothed train + val loss
- `outputs/plots/grad_norm_curve.png` — gradient norm over epochs

**On val loss improvement:**
- `outputs/checkpoints/<JOB_ID>_LoRA.pt` — LoRA adapters only (~5 MB); reloadable on top of any base model

**After training completes:**
- `outputs/checkpoints/<JOB_ID>_Full/` — merged ControlNet (~1.4 GB); load directly without LoRA
- `outputs/samples/` — generated image grids (mask | generated images)

---

## Inference

```bash
# From merged checkpoint (recommended)
python inference.py \
    --controlnet outputs/checkpoints/1581867_Full \
    --mask data/test/labels/image_128.png \
    --prompt "urban sidewalk with trees and parked cars" \
    --output results/image_128.png

# From LoRA checkpoint
python inference.py \
    --lora outputs/checkpoints/1581867_LoRA.pt \
    --mask data/test/labels/image_128.png \
    --prompt "urban sidewalk with trees and parked cars"
```

Options: `--num-images`, `--steps`, `--guidance`, `--control-strength`, `--seed`

Saves a side-by-side grid (mask | gen_0 | gen_1 | ...) plus individual images.

---

## Evaluation

| Metric | What it measures |
|---|---|
| Train MSE loss | Noise prediction error — training signal |
| Val MSE loss | Same on held-out set; expect ±0.01 fluctuation per epoch (random timestep sampling) |
| FID | Fréchet Inception Distance — unreliable below ~2048 images; use visual inspection instead |
| mIoU | Segmentor accuracy on generated images vs. input mask (Phase 2 only) |

---

## Environment

```bash
conda activate control2
# See requirements.txt or environment.yml for full package list
# Key: torch, diffusers, transformers, peft, torchvision, scipy, torchmetrics
```
