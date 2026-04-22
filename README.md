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
│   ├── sampling.py                   # Sample image grids during/after training
│   ├── evaluation.py                 # FID + mIoU evaluation
│   ├── models/
│   │   └── diffusion_controlnet.py   # Full model wrapper (VAE, CLIP, UNet, ControlNet)
│   └── data/
│       ├── dataset.py                # ControlNetDataset, DataLoader, train/val helpers
│       └── transforms.py             # Mask augmentation (jitter, morph, elastic, occlusion)
├── data_prep/
│   ├── class_mapping.py              # 35-class color palette — single source of truth
│   ├── generate_prompts.py           # Auto-generate prompts from mask class statistics
│   └── prompt_templates.py           # Prompt templates per scene type
├── data/
│   ├── train/
│   │   ├── images/                   # ~950 RGB training images
│   │   ├── labels/                   # Segmentation masks (PNG, color-coded by class)
│   │   └── prompts/prompts.csv
│   └── test/
│       ├── images/                   # ~50 RGB validation images
│       ├── labels/
│       └── prompts/prompts.csv
├── jobs/
│   └── train_phase1.sh               # SLURM job script (A100 partition, 8h)
└── outputs/
    ├── checkpoints/                  # LoRA weights — saved only on val loss improvement
    ├── logs/                         # metrics CSV + SLURM .out / .err files
    ├── samples/                      # Generated image grids per epoch
    └── plots/                        # PNG plots (loss, FID, grad norm)
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

**Color palette:** 21 classes reuse the original ADE20K colors (the pretrained ControlNet already understands these). 14 classes use new unique colors chosen to be visually distinct from all ADE20K entries. All colors are defined in `data_prep/class_mapping.py` and loaded by the dataset at runtime — this is the single source of truth.

---

## Training

### Phase 1 — Diffusion loss only

Standard DDPM epsilon-prediction loss: MSE between the noise predicted by the UNet and the actual noise added to the latent. The ControlNet LoRA adapts the pretrained segmentation ControlNet to the custom 35-class palette and urban domain.

**Hyperparameters (`src/config.py`):**

| Parameter | Value | Notes |
|---|---|---|
| LoRA rank / alpha | 16 / 16 | Scale = alpha/rank = 1.0 |
| LoRA target modules | `to_q, to_k, to_v, to_out.0, conv_in` | 
| LoRA dropout | 0.1 | Regularization for small dataset |
| Learning rate | 1e-5 | Conservative — keeps LoRA adapters close to pretrained init |
| LR schedule | Cosine + 100 warmup steps | Linear ramp avoids full-LR cold-start spike |
| Batch size | 2 × 4 grad accum = 8 effective | A100 memory budget |
| Epochs | 30 | ~3540 optimizer steps total |
| Mixed precision | bf16 | A100 native; no GradScaler needed |
| Text dropout | 15% | Classifier-free guidance training — empty prompt for unconditional branch |
| UNet | Fully frozen | Avoids the moving-target problem; matches HF ControlNet reference |

### Phase 2 — Segmentation consistency loss (planned)

Adds a frozen segmentor in the loop: generate an image from the mask, run the segmentor on it, and penalize the model if the output mask diverges from the input. Controlled by `USE_SEGMENTOR_LOSS` in config (currently `False`).

---

## Running on HPC (SLURM)

```bash
# Submit job
sbatch jobs/train_phase1.sh

# Monitor live
tail -f outputs/logs/train_phase1_<JOB_ID>.out
```

The job script sets `DDPM_DATA_ROOT` and `DDPM_OUTPUT_ROOT`. Paths resolve automatically in `src/config.py` via `setup_paths_from_env()` — no code changes needed between local and HPC runs.


### Running locally (RTX GPU)

```bash
conda activate control2
export DDPM_DATA_ROOT=$(pwd)/data
export DDPM_OUTPUT_ROOT=$(pwd)/outputs
python -m src.train
```

For an RTX 4050 (6 GB), reduce memory usage in `src/config.py`:
```python
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 2
```

---

## Outputs

**Per epoch:**
- `outputs/logs/metrics_<JOB_ID>.csv` — step, epoch, train_loss, val_loss, lr, grad_norm
- `outputs/plots/loss_curve.png` — smoothed train loss + val loss
- `outputs/plots/grad_norm_curve.png` — average gradient norm over epochs

**On val loss improvement:**
- `outputs/checkpoints/best_val_model.pt` — LoRA-only weights (~5 MB, not the full 5 GB model)

**After training:**
- `outputs/samples/` — generated image grids (mask | generated | reference)
- FID score printed to log

---

## Evaluation

| Metric | What it measures |
|---|---|
| Train MSE loss | Noise prediction error — training signal only |
| Val MSE loss | Same metric on held-out set; expect natural epoch-to-epoch fluctuation (±0.01) since timesteps are sampled randomly |
| FID | Fréchet Inception Distance between generated and real images (lower = better) |
| mIoU | Segmentor accuracy on generated images vs. input mask (Phase 2 only) |


