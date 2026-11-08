# ControlNet testing — experiment log

Running record of every step in this folder: what was run, with which settings, what came out, and where the files are. Newest entries are at the bottom. Updated after every round of work.

## Fixed protocol

Unless an entry says otherwise, every number in this log was measured like this:

| Item | Value |
|---|---|
| Base model | Stable Diffusion 1.5 + `lllyasviel/control_v11p_sd15_seg` |
| Data | segments/sidewalk-semantic, 950 train / 50 test images, 1920×1080 |
| Preprocessing | short side resized to 512, centre crop 512×512 |
| Conditioning | 21-class "v2" mapping, ADE20K colours, void = white |
| Evaluation set | the 50 test images, one generated image each |
| Sampler | DDIM, 35 steps, guidance scale 6.0, control strength 1.0 |
| Generation seed | 42 + batch index (same noise for every model) |
| Metrics | KID (subset 25), FID, LPIPS vs the paired real image, CLIP image↔prompt, CLIP image↔real image |
| Hardware | one NVIDIA A100 40 GB (tinygpu), bf16 mixed precision |
| Code state | git commit `e79b7c0` plus the uncommitted step 0 changes listed below |

How to read the metrics: KID and FID compare the set of generated images with the set of real ones (lower is better). FID is unreliable with 50 images and is kept only as a trend. KID differences below about 0.01 are within seed noise (see step 3).

---

## 2026-10-10 — Audit of the existing code and runs

Read every file in `DDPM/` and the logs of the earlier runs (jobs 1693046, 1698836). Findings that drove the work below:

| # | Finding | Evidence |
|---|---|---|
| 1 | No baseline: the pretrained ControlNet had never been evaluated | no such run in `outputs/logs` |
| 2 | Validation loss nearly flat, so "best checkpoint" was arbitrary | 0.1581 → 0.1569 over 30 epochs (job 1698836) |
| 3 | Prompts carry random lighting/weather/camera words unrelated to the image | 129 of 950 train prompts say "night", about 300 say rain or drizzle |
| 4 | Prompt object lists came from the full frame, not the 512 crop; objects counted from a single pixel | "sign" present in 448 images, tiny in 355 |
| 5 | Logged train loss used only the last micro-batch of each accumulation window | `train.py` |
| 6 | Learning rate 1e-5 is a full-fine-tune value, low for LoRA | gradient norm 0.003–0.005, clipping at 1.0 never active |
| 7 | `evaluate.py` crashed (old checkpoint keys) | keys `model_unet` / `model_controlnet` absent from checkpoints |
| 8 | `inference.py` loaded class-ID masks as grey images instead of palette colours | pixel values 1–33 instead of colours on `image_128.png` |
| 9 | `load_checkpoint` failed on real checkpoints (`weights_only=True` vs a numpy float) | reproduced on `1698836_LoRA.pt` |
| 10 | KID spread was meaningless: subset size equalled sample count (50) | reported std ≈ 1e-7 |

Things that checked out: ADE20K colours correct for all 20 non-void classes; image and mask sizes match for all 1000 pairs; no train/test overlap; conditioning range [0, 1] in training and inference.

---

## Step 0 — Code fixes (2026-10-10, no GPU)

Originals of all 13 edited files: `_backup_before_step0/`. Verification: `step0_fixes/test_step0_cpu.py`, 7/7 checks passed.

| Change | File |
|---|---|
| Baseline evaluation before the first training step, logged as epoch 0 | `src/train.py` |
| Best checkpoint chosen by KID instead of validation loss (switchable); final epoch always evaluated | `src/train.py`, `src/config.py` |
| Validation loss averaged over 8 fixed timesteps per image (62, 187, …, 937) instead of one | `src/train.py` |
| Train loss averaged over every micro-batch | `src/train.py` |
| KID subset size 50 → 25 | `src/config.py` |
| Hyperparameters overridable from the job script as `DDPM_<NAME>=value`; config snapshot saved per run | `src/config.py`, `src/train.py` |
| Class-ID masks coloured and cropped like training | `inference.py` |
| Rewritten for the current checkpoint format; no checkpoint = pretrained baseline | `evaluate.py` |
| Checkpoint loading fixed; metadata saved as plain numbers | `src/models/diffusion_controlnet.py`, `src/train.py` |
| Offline Hugging Face mode in the job script | `jobs/train_phase1.sh` |
| Versions synced to the real environment; full freeze saved | `requirements.txt`, `step0_fixes/env_control2_pip_freeze_2026-10-10.txt` |
| Prompt generator: objects from the 512 crop, minimum area, objects-only variant | `data_prep/generate_prompts.py`, `data_prep/prompt_templates.py` |

Not changed: default hyperparameters, the old tests, and the reduced-size gradient step at the end of an epoch when the batch count is not divisible by the accumulation steps.

Note: validation-loss values from here on are not comparable with earlier runs (different timesteps).

New prompt files, next to the untouched `prompts.csv` in `data/{train,test}/prompts/`:

| File | Content | Example |
|---|---|---|
| `prompts.csv` | original: random scene, lighting, weather, camera + objects from the full frame | "suburban sidewalk, mixed artificial and natural light, humid haze, phone photo, car, street pole, fence, and stairs visible, calm urban area" |
| `prompts_crop.csv` | same style; objects from the 512 crop, each covering ≥ 0.2 % of it | "city pedestrian zone, dusk lighting, drizzle, high ISO noise, car visible, …" |
| `prompts_objects.csv` | no random style words | "street-level photo of a city sidewalk, car and street pole visible" |

---

## Step 1 — CPU checks of data and prompts (2026-10-10)

Script: `step1_cpu_checks/check_data_and_prompts.py`. Result: 7/7 passed.

| Check | Result |
|---|---|
| v2 palette colours unique; remap covers all 35 original IDs | pass |
| Dataset tensors: 3×512×512, image in [-1, 1], conditioning in [0, 1] | pass |
| Conditioning image contains only exact palette colours | pass |
| Prompt length vs the 77-token CLIP limit | none truncated; longest 50 (`prompts.csv`), 48 (`prompts_crop.csv`), 28 (`prompts_objects.csv`) |
| Image/mask alignment after the crop | visually correct, overlays in `step1_cpu_checks/overlays/` |

---

## Step 2 — Segmentor for Phase 2 (2026-10-10, job 1836679, RTX 3080)

Script: `step2_segmentor/derive_mapping_and_ceiling.py`. Outputs: `step2_segmentor/segmentor_report.txt`, `step2_segmentor/segmentor_mapping_and_ceiling.json`.

- **Model:** UNet++ with EfficientNet-B4 encoder, checkpoint `sensation-models/UnetPlusPlus/efficientnet-b4/ckpt/sensation/epoch=251-val_loss=0.26557-val_iou=0.58698.ckpt`. Trained at 1024×512 with ImageNet normalisation.
- **Its 10 classes:** background, road, sidewalk, bikelane, person, car, bicycle, traffic sign, traffic light, obstacle.
- **Libraries:** `segmentation-models-pytorch` 0.3.4 and dependencies, installed in `/home/woody/iwi5/iwi5388h/software/private/segdeps_smp034` (the `control2` environment was not modified).
- **Class mapping:** the original conversion table was not found, so it was derived from data. The segmentor was run on 300 real training images; each of our 35 classes is mapped to the class the segmentor predicts for at least half of its pixels, otherwise to background.

| Our class | Share of its pixels the segmentor assigns | Mapped to |
|---|---|---|
| road | 89 % road | road |
| sidewalk | 94 % sidewalk | sidewalk |
| cycling lane | 64 % bikelane | bikelane |
| car | 95 % car | car |
| caravan, trailer | 52 %, 68 % car | car |
| bicycle | 72 % bicycle | bicycle |
| sign | 64 % traffic sign | traffic sign |
| pole | 69 % obstacle | obstacle |
| person | 31 % person | background (not reliably detected) |
| traffic light | 20 % traffic light | background (not reliably detected) |
| everything else (building, vegetation, sky, wall, …) | mostly background | background |

Segmentor accuracy on the 50 real test images, background ignored. This is the ceiling a generated image can reach:

| Input geometry | mIoU | Pixel accuracy |
|---|---|---|
| 1024×512 full frame (as it was trained) | 0.787 | 0.927 |
| 512×512 centre crop (as we generate) | 0.807 | 0.956 |

Per class at the 512 crop: road 0.84, sidewalk 0.97, bikelane 0.95, car 0.93, bicycle 0.84, traffic sign 0.56, obstacle 0.56.

Open points: the mapping is not confirmed against the original table; it is unknown whether the segmentor's training set included our 50 test images (if so, the ceiling is optimistic).

---

## Step 3 — Baseline: pretrained ControlNet, no fine-tuning (2026-10-10)

Script: `evaluate.py` without a checkpoint, via `jobs/baseline_eval.sh`. Three generation seeds (42, 43, 44) per prompt file. Raw results: `step3_baseline/results/*.json`.

| Prompt file | Job | KID ↓ | FID ↓ | LPIPS ↓ | CLIP prompt ↑ | CLIP image ↑ |
|---|---|---|---|---|---|---|
| `prompts.csv` (original) | 1836672 | 0.146 ± 0.007 | 243 ± 7 | 0.692 | 0.283 | 0.669 |
| `prompts_crop.csv` | 1836673 | 0.144 ± 0.007 | 244 ± 7 | 0.689 | 0.288 | 0.671 |
| `prompts_objects.csv` | 1836674 | 0.095 ± 0.003 | 197 ± 2 | 0.648 | 0.292 | 0.693 |

Values are mean ± standard deviation over the three seeds.

- The random lighting/weather/camera words account for about a third of the baseline's distance to real images (KID 0.146 → 0.095 without them).
- Deriving objects from the crop makes no measurable difference.
- Seed-to-seed KID spread is about 0.007 at this level, which sets the noise floor for comparisons.

---

## Step 4 — Smoke test of the fixed pipeline (2026-10-10, job 1836671)

One epoch, LR 1e-4, rank 16, alpha 16, dropout 0, batch 4 × accumulation 1, original prompts. Output: `step4_smoke/run/`. Runtime 13 min, peak GPU memory 16.9 GB.

| Stage | Val loss | KID ↓ | FID ↓ | LPIPS ↓ |
|---|---|---|---|---|
| Before training (epoch 0) | 0.1491 | 0.147 | 239 | 0.688 |
| After 1 epoch (238 steps) | 0.1480 | 0.045 | 155 | 0.634 |

- Every stage ran: baseline evaluation, training, checkpoint selection by KID, reload, merge, final evaluation, sample grids.
- One epoch at LR 1e-4 matches the earlier 30-epoch run at LR 1e-5 (job 1698836: KID 0.044, FID 160).
- Known small inconsistency: final KID 0.0457 vs periodic 0.0452 for the same weights, because KID draws random subsets (FID identical). To be fixed by seeding the subsets.

---

## Step 5 — Phase 1 learning-rate sweep (started 2026-10-10, running)

30 epochs each, original prompts, diffusion loss only. Output: `step5_phase1_sweep/<name>/`.

| Name | Job | LR | Rank / alpha | Dropout | Batch × accumulation |
|---|---|---|---|---|---|
| `lr5e-5_r16` | 1836675 | 5e-5 | 16 / 16 | 0.0 | 4 × 1 |
| `lr1e-4_r16` | 1836676 | 1e-4 | 16 / 16 | 0.0 | 4 × 1 |
| `lr2e-4_r16` | 1836677 | 2e-4 | 16 / 16 | 0.0 | 4 × 1 |
| `oldconfig_lr1e-5_r8` | 1836678 | 1e-5 | 8 / 16 | 0.1 | 2 × 4 |

Interim values after 5 of 30 epochs (not final):

| Name | KID ↓ | FID ↓ | LPIPS ↓ |
|---|---|---|---|
| baseline (epoch 0) | 0.147 | 239 | 0.688 |
| `oldconfig_lr1e-5_r8` | 0.086 | 191 | 0.659 |
| `lr5e-5_r16` | 0.032 | 146 | 0.621 |
| `lr1e-4_r16` | 0.026 | 141 | 0.615 |
| `lr2e-4_r16` | 0.023 | 138 | 0.613 |

Final results: to be added when the jobs finish.

---

## Planned next

1. Record the final sweep results and pick the best setting.
2. Add segmentor mIoU to the evaluation, so every model including the baseline gets a mask-adherence score.
3. Train the best setting with `prompts_objects.csv`.
4. Build and test the Phase 2 segmentation loss, then train with it.
5. Final comparison table: baseline vs Phase 1 vs Phase 2.

---

## 2026-10-11 — Results document

- `RESULTS.docx` (this folder) holds the raw results: protocol, tables, curves and sample images. It is generated by `reporting/build_results_doc.py` from the result files, so it is rebuilt after every round and never edited by hand. New runs are registered in `RUN_GROUPS` at the top of that script.
- The Word library (`python-docx`) lives in `/home/woody/iwi5/iwi5388h/software/private/reportdeps`, outside the `control2` environment.
- `reporting/make_comparison_grid.py` + `jobs/comparison_grid.sh` produce side-by-side sheets (real | mask | pretrained | models) with identical mask, prompt and noise per row.

## Step 2b — Segmentor wired into the code (2026-10-11)

Originals of the files edited here: `_backup_before_phase2/`.

- **`src/segmentor.py` (new):** loads the frozen segmentor and defines the target mapping used from now on. It differs from the first measurement in step 2: confident background is scored as a class, and ambiguous classes are ignored.

| Segmentor class | Dataset classes |
|---|---|
| road | road |
| sidewalk | sidewalk |
| bikelane | cycling lane |
| car | car, trailer |
| bicycle | bicycle |
| traffic sign | sign |
| obstacle | pole |
| background | truck, bus, building, door, wall, bridge, tunnel, stairs, vegetation, earth, sky |
| ignored | unlabeled, crosswalk, parking/driveway, rail track, curb, person, rider, tram, motorbike, caravan, fence, traffic light, the four void classes |

- **Dataset** returns this target mask (`mask_seg`), built from the original 35-class IDs because the 21-class conditioning merges cycling lane into road.
- **Evaluation** now reports `miou` (segmentor on generated images vs the mask) and `miou_real` (same segmentor on the real photos, the ceiling) in every evaluation. The metrics CSV gained the columns `miou`, `miou_real`, `seg_loss`.
- **KID** subsets are now drawn with a fixed seed, so the same images always give the same number.
- With this mapping the ceiling on the 50 real test images is **mIoU 0.802**.

## Step 3b — Baseline re-run with mIoU (2026-10-11, jobs 1836697–1836699)

Same protocol as step 3, three seeds each. The first run's files are kept in `step3_baseline/results_run1_without_miou/`. KID/FID/LPIPS reproduce the first run.

| Prompt file | KID ↓ | FID ↓ | LPIPS ↓ | mIoU ↑ | Pixel accuracy ↑ |
|---|---|---|---|---|---|
| `prompts.csv` (original) | 0.144 ± 0.006 | 243 ± 7 | 0.692 | 0.266 ± 0.026 | 0.667 |
| `prompts_crop.csv` | 0.143 ± 0.008 | 244 ± 7 | 0.689 | 0.268 ± 0.020 | 0.669 |
| `prompts_objects.csv` | 0.094 ± 0.004 | 197 ± 2 | 0.648 | 0.372 ± 0.012 | 0.750 |
| real photos (ceiling) | – | – | – | 0.802 | – |

- The pretrained model follows the mask poorly: mIoU 0.27 against a ceiling of 0.80.
- Seed-to-seed mIoU spread is about 0.02; smaller differences are not meaningful.
- Per-class IoU of the pretrained model (seed 42, original prompts): background 0.58, road 0.23, sidewalk 0.63, bikelane 0.03, car 0.43, bicycle 0.19, traffic sign 0.18, obstacle 0.15.
- Bikelane is near zero by construction: the 21-class conditioning paints cycling lanes in the road colour, so the generator cannot know where they are. This is a limit of the v2 class mapping, not of training.

## Step 6 — Phase 2: segmentation-consistency loss (2026-10-11)

**Implementation** (`src/models/diffusion_controlnet.py`, `src/losses.py`, `src/train.py`, `src/config.py`):

1. From the predicted noise, compute the denoised latent estimate x0 = (x_t − √(1−ᾱ_t)·ε_pred) / √ᾱ_t.
2. Decode it with the frozen VAE (gradients flow through), giving an image.
3. Run the frozen segmentor on that image.
4. Loss = cross-entropy + soft Dice against the 10-class target mask, ignored pixels skipped.
5. Total loss = diffusion loss + weight × segmentation loss. The weight ramps linearly from 0 over the first 200 optimizer steps.
6. Applied only to samples with timestep < 400 (the estimate is not a recognisable image at higher noise) and to at most 2 samples per batch (memory).

Switches: `DDPM_USE_SEGMENTOR_LOSS`, `DDPM_SEGMENTOR_LOSS_WEIGHT`, `DDPM_SEGMENTOR_LOSS_MAX_TIMESTEP`, `DDPM_SEGMENTOR_LOSS_MAX_SAMPLES`, `DDPM_SEGMENTOR_LOSS_WARMUP_STEPS`. Also new: `DDPM_INIT_LORA_CHECKPOINT` (start from an existing LoRA) and `DDPM_CHECKPOINT_METRIC=last` (no selection, keep the last epoch).

**CPU checks** (`step6_phase2/test_phase2_cpu.py`): 5/5 passed — target mask values valid; segmentor reaches pixel accuracy 0.97 on 10 real images through the new code; loss is 0.70 for the correct mask vs 15.1 for a wrong one; gradient reaches the input image; fully ignored mask gives zero loss.

**GPU smoke test** (job 1836696, `step6_phase2/smoke/`): one epoch with weight 0.1 at LR 5e-5, starting from the step 4 smoke LoRA. Runtime 14 min, peak GPU memory 32.7 GB of 40.

| Stage | KID ↓ | LPIPS ↓ | mIoU ↑ |
|---|---|---|---|
| Start (Phase 1 smoke LoRA, 1 epoch) | 0.045 | 0.634 | 0.396 |
| After 1 epoch with segmentation loss | 0.051 | 0.626 | 0.415 |

- The loss runs end to end; it applied on 209 of 238 batches; mean segmentation loss 1.85.
- Gradient norm rose from about 0.003 (Phase 1) to 0.099, so at weight 0.1 the segmentation term already dominates the gradient.
- The changes in KID and mIoU after one epoch are within seed noise; this run proves the mechanics, not the benefit.

## Step 5 — sweep status at 20 of 30 epochs (interim)

| Run | KID ↓ at epoch 5 / 10 / 15 / 20 |
|---|---|
| old config (LR 1e-5, rank 8) | 0.086 / 0.048 / 0.048 / – |
| LR 5e-5 | 0.032 / 0.030 / 0.024 / 0.025 |
| LR 1e-4 | 0.026 / 0.024 / 0.020 / 0.024 |
| LR 2e-4 | 0.023 / 0.028 / 0.023 / 0.021 |

The three new learning rates are indistinguishable within noise and plateau around KID 0.02–0.025, about half the old config's value.

## Jobs queued on 2026-10-11 (results to be added)

| Job | Name | What |
|---|---|---|
| 1836710 | `lr1e-4_r16_objects` | Phase 1, LR 1e-4, trained and evaluated with `prompts_objects.csv` |
| 1836711–1836714 | `ev_*` | mIoU of each sweep run's selected checkpoint (the sweep started before mIoU existed); output in `step5_phase1_sweep/eval_with_miou/` |
| 1836715 | `cmp_grid` | side-by-side sheet: pretrained vs old config vs LR 1e-4 vs LR 2e-4; output in `reporting/comparison/phase1/` |
| 1836716 | `control_no_segloss` | Phase 2 control: continue the LR 1e-4 LoRA (job 1836676) for 10 epochs at LR 5e-5, no segmentation loss |
| 1836717–1836719 | `segloss_w0.03`, `_w0.1`, `_w0.3` | same continuation with segmentation-loss weight 0.03, 0.1, 0.3 |

All Phase 2 runs evaluate every 2 epochs and keep the last epoch (no checkpoint selection), so the comparison with the control is like for like.

---

## Step 5 — Phase 1 sweep, final results (finished 2026-10-11 01:05)

Selected checkpoint = lowest KID among the evaluated epochs. Single generation seed. mIoU measured afterwards on the selected checkpoint (jobs 1836711–1836714, `step5_phase1_sweep/eval_with_miou/`).

| Run | Job | Epoch selected | KID ↓ | FID ↓ | LPIPS ↓ | mIoU ↑ |
|---|---|---|---|---|---|---|
| pretrained (epoch 0) | – | 0 | 0.147 | 239 | 0.688 | 0.27 |
| old config (LR 1e-5, rank 8) | 1836678 | 30 | 0.042 | 159 | 0.633 | 0.38 |
| LR 5e-5, rank 16 | 1836675 | 30 | 0.024 | 140 | 0.615 | 0.42 |
| LR 1e-4, rank 16 | 1836676 | 15 | 0.022 | 139 | 0.609 | 0.46 |
| LR 2e-4, rank 16 | 1836677 | 30 | 0.019 | 140 | 0.605 | 0.42 |

- The three new learning rates end at the same quality within noise (KID 0.019–0.024); all halve the old configuration's KID.
- Runtime: 66 min per run (old configuration 85 min).
- Validation loss moved 0.1491 → 0.1468–0.1472; it orders the runs correctly but the differences are tiny.

**Measurement caveat found here.** The same LoRA (job 1836676) gave mIoU 0.462 in the standalone evaluation and 0.418 when evaluated as the starting point of the Phase 2 runs. The only difference is the batch size of the evaluation loader (2 vs 4), which changes the per-batch generation seed. So a single-seed mIoU on 50 images can move by about 0.04 for identical weights. Step 7 re-measures the key models with three seeds at one batch size.

## Step 5b — Phase 1 with objects-only prompts (job 1836710, finished 2026-10-11 01:34)

LR 1e-4, rank 16, trained and evaluated with `prompts_objects.csv`. Output: `step5_phase1_sweep/lr1e-4_r16_objects/`.

| Stage | KID ↓ | FID ↓ | LPIPS ↓ | mIoU ↑ |
|---|---|---|---|---|
| Before training (objects prompts) | 0.078 | 179 | 0.646 | 0.40 |
| After 30 epochs (selected: epoch 30) | 0.010 | 129 | 0.585 | 0.44 |

- Best model so far on every realism metric: KID 0.010 against 0.022 for the same settings with the original prompts.
- mIoU over training: 0.42 / 0.43 / 0.47 / 0.43 / 0.44 / 0.44 at epochs 5–30, i.e. flat within noise after the first epochs.
- Note the two prompt sets are different evaluation conditions; the fair statement is "objects prompts + fine-tuning gives KID 0.010", not a controlled prompt ablation of the training alone.

## Step 6 — Phase 2 results, original prompts (finished 2026-10-11 01:35)

All four runs continue the Phase 1 LR 1e-4 LoRA (job 1836676) for 10 epochs at LR 5e-5; last epoch reported, single seed. Output: `step6_phase2/<name>/`. Runtime 37 min (control) and 45 min (with segmentation loss).

| Run | Job | KID ↓ | FID ↓ | LPIPS ↓ | mIoU ↑ | Val loss |
|---|---|---|---|---|---|---|
| start (Phase 1 LoRA) | 1836676 | 0.021 | 140 | 0.609 | 0.418 | 0.1471 |
| control, no seg loss | 1836716 | 0.022 | 141 | 0.611 | 0.421 | 0.1470 |
| seg-loss weight 0.03 | 1836717 | 0.032 | 152 | 0.606 | 0.474 | 0.1490 |
| seg-loss weight 0.1 | 1836718 | 0.051 | 163 | 0.618 | 0.432 | 0.1520 |
| seg-loss weight 0.3 | 1836719 | 0.080 | 192 | 0.634 | 0.416 | 0.1556 |

mIoU at epochs 2 / 4 / 6 / 8 / 10 (same seeds in every run, so the rows are paired):

| Run | mIoU |
|---|---|
| control | 0.420 / 0.401 / 0.417 / 0.419 / 0.421 |
| weight 0.03 | 0.454 / 0.451 / 0.463 / 0.478 / 0.474 |
| weight 0.1 | 0.438 / 0.463 / 0.425 / 0.431 / 0.432 |
| weight 0.3 | 0.463 / 0.374 / 0.412 / 0.420 / 0.416 |

Per-class IoU at the last epoch:

| Run | road | sidewalk | bikelane | car | bicycle | traffic sign | obstacle |
|---|---|---|---|---|---|---|---|
| control | 0.28 | 0.83 | 0.00 | 0.55 | 0.58 | 0.05 | 0.27 |
| weight 0.03 | 0.44 | 0.90 | 0.21 | 0.62 | 0.53 | 0.02 | 0.27 |
| weight 0.1 | 0.37 | 0.92 | 0.15 | 0.51 | 0.42 | 0.00 | 0.28 |

- **Weight 0.03** raises mIoU by about 0.05 over the control at every one of the five evaluations, mainly on road, sidewalk and bikelane. It costs realism: KID 0.022 → 0.032.
- **Weights 0.1 and 0.3** damage image quality (KID 0.05 and 0.08) without a reliable mIoU gain. Gradient norm was 0.08 / 0.30 / 1.1 for weights 0.03 / 0.1 / 0.3, against 0.007 without the loss, so the segmentation term dominates the update even at the smallest weight.
- The measured segmentation loss fell only slightly during training (about 1.6 → 1.4–1.5).
- The mIoU gain at weight 0.03 is about the size of the single-seed noise; it is consistent across epochs but needs the three-seed check in step 7 before it is claimed.

## Side-by-side sheet, Phase 1 (job 1836715)

`reporting/comparison/phase1/comparison_sheet.jpg`: real | mask | pretrained | old config | LR 1e-4 | LR 2e-4 for six test images. Visible in it: fine-tuned models reproduce the dataset's paving and street look where the pretrained model does not; the prompt's random weather/lighting words (fog, night) are followed by all models and make the outputs differ from the daytime photos.

## Step 7 — queued 2026-10-11 12:30 (results to be added)

| Job | Name | What |
|---|---|---|
| 1836950–1836956 | `ms_*` | three-seed evaluation (seeds 42–44, evaluation batch size 4) of: pretrained with original and objects prompts, Phase 1 LR 1e-4, Phase 2 control, Phase 2 weights 0.03 and 0.1, Phase 1 objects. Output: `step7_multiseed_eval/` |
| 1836957 | `objects_control` | Phase 2 control on the objects-prompt LoRA (job 1836710): 10 epochs, LR 5e-5, no segmentation loss |
| 1836958, 1836959 | `objects_segloss_w0.01`, `_w0.03` | same with segmentation-loss weight 0.01 and 0.03 |
| 1836960 | `cmp_grid` | side-by-side sheet Phase 1 vs control vs weight 0.03 vs weight 0.1 → `reporting/comparison/phase2/` |
| 1836961 | `cmp_grid` | side-by-side sheet for objects prompts: pretrained vs fine-tuned → `reporting/comparison/objects/` |

---

## Step 7 — Three-seed evaluation of the key models (jobs 1836950–1836956, 2026-10-11 12:45)

Each model generated the 50 test images three times (seeds 42, 43, 44; evaluation batch size 4). Mean ± standard deviation over seeds. Raw files: `step7_multiseed_eval/*.json`. **These are the numbers to quote.**

| Model | Prompts | KID ↓ | FID ↓ | LPIPS ↓ | mIoU ↑ | Pixel acc. ↑ |
|---|---|---|---|---|---|---|
| Real photos (ceiling) | – | – | – | – | 0.802 | – |
| Pretrained | original | 0.144 ± 0.010 | 240 ± 7 | 0.688 | 0.257 ± 0.004 | 0.649 |
| Phase 1, LR 1e-4 (job 1836676) | original | 0.021 ± 0.001 | 139 ± 1 | 0.610 | 0.396 ± 0.020 | 0.844 |
| Phase 1 + 10 epochs, no seg loss (control, 1836716) | original | 0.021 ± 0.001 | 141 ± 1 | 0.612 | 0.400 ± 0.027 | 0.839 |
| Phase 2, seg-loss weight 0.03 (1836717) | original | 0.032 ± 0.003 | 149 ± 4 | 0.606 | 0.494 ± 0.015 | 0.876 |
| Phase 2, seg-loss weight 0.1 (1836718) | original | 0.057 ± 0.005 | 171 ± 6 | 0.619 | 0.454 ± 0.017 | 0.869 |
| Pretrained | objects-only | 0.085 ± 0.006 | 188 ± 8 | 0.647 | 0.363 ± 0.027 | 0.760 |
| Phase 1, LR 1e-4 (job 1836710) | objects-only | 0.008 ± 0.002 | 127 ± 4 | 0.584 | 0.423 ± 0.023 | 0.858 |

Per-class IoU (mean of three seeds):

| Model | road | sidewalk | bikelane | car | bicycle | traffic sign | obstacle |
|---|---|---|---|---|---|---|---|
| Pretrained, original prompts | 0.21 | 0.54 | 0.00 | 0.37 | 0.17 | 0.06 | 0.17 |
| Phase 1, LR 1e-4 | 0.32 | 0.83 | 0.00 | 0.54 | 0.35 | 0.08 | 0.25 |
| Control (no seg loss) | 0.30 | 0.82 | 0.00 | 0.54 | 0.39 | 0.09 | 0.26 |
| Phase 2, weight 0.03 | 0.38 | 0.91 | 0.19 | 0.61 | 0.62 | 0.11 | 0.31 |
| Phase 2, weight 0.1 | 0.33 | 0.92 | 0.16 | 0.53 | 0.49 | 0.06 | 0.32 |
| Phase 1, objects prompts | 0.34 | 0.85 | 0.01 | 0.56 | 0.42 | 0.10 | 0.29 |

- **Phase 1:** KID 0.144 → 0.021 and mIoU 0.26 → 0.40 over the pretrained model. Ten more epochs without the segmentation loss change nothing (control = Phase 1).
- **Phase 2, weight 0.03:** mIoU 0.400 → 0.494 against the control, a gain of 0.09 with seed spreads of 0.015–0.027, so it is a real effect. Every class improves. The cost is realism: KID 0.021 → 0.032.
- **Phase 2, weight 0.1:** smaller mIoU gain (0.454) and a larger realism cost (KID 0.057); worse than 0.03 on both.
- **Objects-only prompts:** best realism of all (KID 0.008); mask adherence similar to the original-prompt Phase 1 model.
- Small classes (traffic sign, obstacle, bikelane) stay low in every model. Bikelane cannot be conditioned at all with the 21-class mapping.

## Side-by-side sheets (jobs 1836960, 1836961)

- `reporting/comparison/phase2/comparison_sheet.jpg`: Phase 1 | control | seg loss 0.03 | seg loss 0.1.
- `reporting/comparison/objects/comparison_sheet.jpg`: pretrained | fine-tuned, both with objects-only prompts.

## Phase 2 on the objects-prompt model — running (jobs 1836957–1836959)

Interim, single seed, epoch 4 of 10: control mIoU 0.455 / KID 0.009; weight 0.01 mIoU 0.496 / KID 0.027; weight 0.03 mIoU 0.434 / KID 0.039. Final values to be added.

---

## Step 6b / 7b — Phase 2 on the objects-prompt model, three seeds (jobs 1836957–1836959, 1836978–1836980)

Continues the objects-prompt Phase 1 LoRA (job 1836710) for 10 epochs at LR 5e-5; `prompts_objects.csv` throughout. Three generation seeds, evaluation batch size 4.

| Model | KID ↓ | FID ↓ | LPIPS ↓ | mIoU ↑ |
|---|---|---|---|---|
| Pretrained, objects prompts | 0.085 ± 0.006 | 188 | 0.647 | 0.363 ± 0.027 |
| Phase 1, objects prompts (1836710) | 0.008 ± 0.002 | 127 | 0.584 | 0.423 ± 0.023 |
| Control: + 10 epochs, no seg loss (1836957) | 0.011 ± 0.002 | 126 | 0.582 | 0.438 ± 0.016 |
| Phase 2, seg-loss weight 0.01 (1836958) | 0.017 ± 0.002 | 136 | 0.589 | 0.486 ± 0.014 |
| Phase 2, seg-loss weight 0.03 (1836959) | 0.032 ± 0.001 | 144 | 0.604 | 0.483 ± 0.017 |

- Weight 0.01 gives the same mIoU gain as 0.03 (+0.05 over the control) at half the realism cost. It is the best trade-off measured: KID 0.017 is still better than the original-prompt Phase 1 model (0.021), with mIoU 0.49.
- Per class (weight 0.01 vs control): road 0.40 vs 0.36, sidewalk 0.89 vs 0.86, bikelane 0.17 vs 0.01, car 0.70 vs 0.65, bicycle 0.48 vs 0.40.

## Step 8 — Downstream test: does the synthetic data help a segmentation model? (2026-10-11)

Scripts: `step8_downstream/generate_synthetic.py`, `step8_downstream/train_segmentor.py`, `jobs/downstream.sh`. Results: `step8_downstream/results/*.json`. Synthetic images: `step8_downstream/synthetic/<generator>/`.

**Design**
- Synthetic sets: for each of the 950 training masks, one 512×512 image per generator (DDIM 35 steps, guidance 6.0, half precision), labelled with the mask it was generated from. Generators: pretrained ControlNet (job 1836969), Phase 1 LR 1e-4 (1836972), Phase 2 weight 0.03 (1836975); original prompts. About 14 min per set.
- Segmentor: U-Net, ResNet-34 encoder (ImageNet weights), 21 classes, void ignored. Not the segmentor used for the Phase 2 loss.
- Training: 6,000 steps, batch 8, AdamW, one-cycle LR peaking at 3e-4, horizontal flip, mixed precision. Identical in every setting.
- Test: the 50 real test images, mIoU over the classes present. Three training seeds per setting.

**Results (jobs 1836968, 1836970–71, 1836973–74, 1836976–77)**

| Training data | Test mIoU ↑ | Pixel acc. ↑ |
|---|---|---|
| Real only (950) | 0.506 ± 0.011 | 0.931 |
| Synthetic only, pretrained generator | 0.306 ± 0.004 | 0.803 |
| Synthetic only, Phase 1 generator | 0.400 ± 0.006 | 0.885 |
| Synthetic only, Phase 2 generator (weight 0.03) | 0.379 ± 0.015 | 0.882 |
| Real + synthetic, pretrained generator | 0.417 ± 0.003 | 0.916 |
| Real + synthetic, Phase 1 generator | 0.442 ± 0.009 | 0.923 |
| Real + synthetic, Phase 2 generator (weight 0.03) | 0.433 ± 0.006 | 0.924 |

Selected per-class IoU:

| Training data | road | sidewalk | person | car | bicycle | fence | pole | sign |
|---|---|---|---|---|---|---|---|---|
| Real only | 0.77 | 0.92 | 0.77 | 0.90 | 0.66 | 0.64 | 0.56 | 0.17 |
| Synthetic only, pretrained | 0.39 | 0.79 | 0.46 | 0.68 | 0.14 | 0.17 | 0.33 | 0.01 |
| Synthetic only, Phase 1 | 0.64 | 0.87 | 0.57 | 0.74 | 0.42 | 0.24 | 0.43 | 0.01 |
| Synthetic only, Phase 2 | 0.58 | 0.88 | 0.45 | 0.78 | 0.11 | 0.32 | 0.42 | 0.00 |
| Real + Phase 1 | 0.76 | 0.92 | 0.60 | 0.87 | 0.26 | 0.57 | 0.48 | 0.00 |
| Real + Phase 2 | 0.76 | 0.92 | 0.52 | 0.86 | 0.05 | 0.60 | 0.51 | 0.00 |

- **Fine-tuning the generator makes its images much more useful for training:** synthetic-only mIoU 0.31 → 0.40 from the pretrained to the Phase 1 generator.
- **The Phase 2 generator is not better than Phase 1 here** (0.38 vs 0.40, within about one standard deviation). The mask-adherence gain measured by the frozen segmentor did not carry over to this test.
- **Adding synthetic images to the real ones lowers accuracy** (0.51 → 0.42–0.44) in this setup. Large classes are unchanged (road, sidewalk, car, building); the loss is in small and rare classes (bicycle 0.66 → 0.05–0.26, person 0.77 → 0.5–0.6, sign 0.17 → 0.00).
- Likely causes, each tested by a queued run: (a) the generated images often do not contain the small object the mask says is there, so the label is wrong; (b) the original prompts ask for night, fog and rain, which the daytime test set does not contain; (c) with a fixed step budget the mixed runs see each real image half as often.
- Preview of real vs synthetic training images: `step8_downstream/synthetic_preview.jpg`.

**Follow-up runs queued 2026-10-11 23:35**

| Job | Name | Tests |
|---|---|---|
| 1837181, 1837185 | `gen_phase1_objects`, `gen_phase2_objects_w0.01` | synthetic sets from the objects-prompt generators (no weather/lighting words) |
| 1837182–84, 1837186–88 | `synth_only_*`, `real_plus_*`, `real_plus_*_frac25` | those sets alone, mixed by count, and mixed with 25 % synthetic |
| 1837189 | `real_only_12k` | real only with twice the steps (is 6,000 enough?) |
| 1837190 | `real_plus_phase1_12k` | real + Phase 1 synthetic with twice the steps (same real exposure as real-only at 6,000) |
| 1837191 | `real_plus_phase1_frac25` | real + Phase 1 synthetic, 25 % synthetic |

---

## Step 8 — Downstream test, follow-up results (jobs 1837181–1837191, finished 2026-10-12)

Same segmentor and protocol as above (three training seeds, 50 real test images). New synthetic sets from the objects-prompt generators: Phase 1 (job 1837181) and Phase 2 weight 0.01 (job 1837185).

| Training data | Steps | Test mIoU ↑ |
|---|---|---|
| Real only | 6,000 | 0.506 ± 0.011 |
| Real only | 12,000 | 0.550 ± 0.002 |
| Synthetic only, Phase 1 (original prompts) | 6,000 | 0.400 ± 0.005 |
| Synthetic only, Phase 1 (objects prompts) | 6,000 | 0.404 ± 0.009 |
| Synthetic only, Phase 2 weight 0.01 (objects prompts) | 6,000 | 0.402 ± 0.011 |
| Real + Phase 1 synthetic, 1:1 | 6,000 | 0.442 ± 0.009 |
| Real + Phase 1 synthetic, 1:1 | 12,000 | 0.510 ± 0.011 |
| Real + Phase 1 synthetic, 25 % synthetic | 6,000 | 0.451 ± 0.010 |
| Real + Phase 1 objects synthetic, 1:1 | 6,000 | 0.460 ± 0.013 |
| Real + Phase 1 objects synthetic, 25 % synthetic | 6,000 | 0.471 ± 0.003 |
| Real + Phase 2 (0.01) objects synthetic, 1:1 | 6,000 | 0.456 ± 0.019 |
| Real + Phase 2 (0.01) objects synthetic, 25 % synthetic | 6,000 | 0.455 ± 0.015 |

- **6,000 steps under-trains the segmentor.** Real-only rises from 0.506 to 0.550 with twice the steps, so part of the earlier gap between real-only and mixed runs was the halved exposure to real images.
- **With that removed, synthetic data still does not help.** Real + synthetic at 12,000 steps (0.510) sees the real images as often as real-only at 6,000 steps (0.506) and scores the same; at equal total steps it is 4 points below real-only (0.510 vs 0.550).
- **Objects-prompt synthetic images hurt less** when mixed (0.46–0.47 vs 0.44), consistent with the weather/lighting mismatch being one cause. They are still below real-only.
- **A smaller synthetic share (25 %) helps only marginally.**
- **The Phase 2 generator gives no downstream advantage** in any setting (synthetic-only 0.402 vs 0.404; mixed 0.456 vs 0.460).
- Small classes remain the weak point of every mixed run (bicycle 0.26–0.53 vs 0.66 real-only; sign 0.00–0.05 vs 0.17).

**Conclusion so far:** on this dataset, with one generated image per known training mask, the generated data does not improve a segmentor that already has all 950 real images. Fine-tuning the generator makes its images far more usable (synthetic-only 0.31 → 0.40), but the segmentation loss adds nothing on top.

GPU usage 2026-10-10 to 2026-10-13: 22.5 A100 GPU-hours.

---

## Step 8d — three follow-up ideas (queued 2026-10-13, about 3.6 GPU-hours)

Each bundle is one job (`step8_downstream/run_ideas.sh <bundle>`). All use the objects-prompt Phase 1 synthetic set (`synthetic/phase1_objects`), the one that mixed best so far. Three training seeds each.

| Job | Bundle | Runs |
|---|---|---|
| 1838294 | `lowdata` | Only the first 100 real images: (a) real only; (b) + synthetic images from those same 100 masks; (c) + all 950 synthetic images. 50 % synthetic in (b) and (c); 6,000 steps |
| 1838295 | `filter` | `filter_synthetic.py`: labels set to void wherever the frozen segmentor disagrees with the mask. Then (a) filtered synthetic only, 6,000 steps (compare 0.404 unfiltered); (b) real + filtered synthetic, 12,000 steps (compare real-only 0.550) |
| 1838296 | `multi` | Three generated images per mask instead of one (2,850 images, seed 4321), then real + synthetic at 50 %, 12,000 steps (compare real-only 0.550) |

Change from the plan: the third idea was first described as "generate from flipped masks". That would not add anything, because the segmentor training already flips every image and mask horizontally. It was replaced by more generated variants per mask, which tests whether appearance diversity helps.

Caveat for the low-data runs: the generator was trained on all 950 real images, so (b) and (c) contain information from images the "100-image" segmentor is not supposed to have. (c) additionally assumes 950 labelled masks exist without photos.

## Step 8d — results (jobs 1838294–1838296, finished 2026-10-13)

Three training seeds each, 50 real test images. Runtimes 92 / 66 / 82 min (≈ 4.0 GPU-hours).

| Setting | Training images | Steps | Test mIoU ↑ |
|---|---|---|---|
| Reference: real only (950) | 950 | 12,000 | 0.550 ± 0.002 |
| Reference: real + Phase 1 synthetic (original prompts) | 1,900 | 12,000 | 0.510 ± 0.011 |
| Real + Phase 1 objects synthetic, **labels filtered** | 1,900 | 12,000 | 0.516 ± 0.011 |
| Real + Phase 1 objects synthetic, **3 images per mask** (50 % synthetic) | 3,800 | 12,000 | 0.485 ± 0.004 |
| Synthetic only, Phase 1 objects (reference) | 950 | 6,000 | 0.404 ± 0.009 |
| Synthetic only, Phase 1 objects, **labels filtered** | 950 | 6,000 | 0.399 ± 0.017 |
| **Low data:** 100 real only | 100 | 6,000 | 0.279 ± 0.011 |
| **Low data:** 100 real + synthetic from the same 100 masks (50 %) | 200 | 6,000 | 0.274 ± 0.006 |
| **Low data:** 100 real + synthetic from all 950 masks (50 %) | 1,050 | 6,000 | 0.394 ± 0.008 |

- **Label filtering** removed 18.5 % of the pixels the frozen segmentor can judge (17 % of all pixels), a direct measure of how often generated content disagrees with its mask. It did not change synthetic-only accuracy and brought the mixed run to 0.516, still below real-only (0.550).
- **Three images per mask** made the mixed run worse (0.485), so more appearance variety of the same scenes does not help.
- **Low data:** synthetic images of masks the segmentor already has add nothing (0.279 → 0.274). Synthetic images from 850 additional masks add 11.5 points (0.279 → 0.394), closing about half the gap to training on all 950 real photos (0.506 at the same step count). This is the one setting where the generated data clearly helps; the extra information comes from the additional masks (labels without photos), and the generator was trained on the photos of those masks, so it is an optimistic estimate.

**Overall downstream conclusion:** generated images help a segmentor only when they bring labels it does not otherwise have (more masks than photos). They do not improve a segmentor that already has the real photos of the same scenes, with or without the segmentation loss, label filtering, more variants per mask, or a smaller synthetic share.

GPU usage 2026-10-10 to 2026-10-14: about 26.5 A100 GPU-hours.

---

## Dataset 2: SANPO-Real (prepared 2026-10-17, CPU only)

Code and documentation: `data_prep/sanpo/` (README, index, selection, class mapping, conversion, checks). Processed data: `data_sanpo/` (8.9 GB), same layout and label format as `data/`; switch datasets with `DDPM_DATA_ROOT=<DDPM>/data_sanpo`.

**Source.** SANPO-Real (Google Research, CC BY 4.0): egocentric videos recorded while walking in four US cities, chest-mounted ZED-2i camera, 2208×1242. Of 701 sessions, 116 train and 30 test sessions (official split) have chest-camera segmentation masks.

**Selection** (`select_frames.py`, list in `data_prep/sanpo/selection.csv`):
- Only human-annotated frames (one every 0.4 s); machine-propagated labels are not used.
- Thinned to every 3rd annotated frame for train (about 1.2 s apart) and every 5th for test (about 2 s apart), because neighbouring frames are near-duplicates.
- Result: **3,172 train frames from 116 sessions, 522 test frames from 30 sessions**; train and test never share a session.
- Why this many: scene variety is limited by the number of sessions (116), not frames; denser sampling would mostly add duplicates. It is still 3.3× dataset 1.

**Conversion** (`download_and_convert.py`): frames resized to 1920×1080 (same size and aspect as dataset 1); SANPO class IDs (mask channel 0) mapped to this project's 35-class IDs with `class_mapping_sanpo.py`, nearest-neighbour resize. 3,694 of 3,694 frames converted, none failed.

**Class mapping, main decisions:** vehicle → car; wall/fence, guard rail, hand rail, gate → fence; paved trail and other walkable surface → sidewalk; obstacle, bus stop, bike rack, water body → static void; tree + vegetation → vegetation; terrain → earth. SANPO has no bicycle or bike-lane class.

**Prompts** (`generate_prompts.py --data-root data_sanpo`, objects from the 512 crop, ≥ 0.2 % area): `prompts.csv` (dataset-1 style, random weather), `prompts_objects.csv` (objects only), and new `prompts_meta.csv`, built from the session's recorded weather and environment, e.g. "street-level photo of a sidewalk, urban street, sunny day, pedestrian, car, street pole, fence, and stairs visible".

**Checks** (`check_dataset.py`): 11/11 passed — every frame has image, label and prompt rows; 1920×1080; label IDs 0–34; the project's dataset class loads it; overlays aligned (`data_sanpo/check_overlays/`).

**Content compared with dataset 1** (pixel share, train): sidewalk 38 % (34 %), building 13 % (10 %), vegetation 10 % (14 %), fence 9 % (1 %), road 6 % (11 %), car 3 % (4 %), person 2.3 % (0.3 %), crosswalk 2.8 % (1.7 %), sky 1 % (4 %). Weather: about one third of sessions snowy, the rest sunny/cloudy; none rainy or low-visibility.

**Mapping verified against the official ADE20K palette (2026-10-17).** `data_prep/sanpo/verify_mapping_against_ade20k.py` follows every SANPO class through project ID → 21-class ID → conditioning colour → ADE20K class, using the 150-class table in `data_prep/color_coding_semantic_segmentation_classes.xlsx` (exported once to `data_prep/ade20k_150_palette.csv`). All 20 non-void conditioning colours are exact ADE20K colours of the intended class (road, sidewalk, person, car, …); 0 mismatches. Seven SANPO classes end as white void because neither the project's classes nor ADE20K have a fitting colour (unlabeled, animal, water body, inaccessible surface, obstacle, bus stop, bike rack); together about 7 % of dataset-2 pixels, mostly "obstacle" (5.7 %).

## Dataset 2: switched to a direct SANPO → ADE20K mapping (2026-10-17, user request)

The first conversion pushed SANPO through the project's 21-class set, built for dataset 1. At the user's request dataset 2 now maps SANPO's 31 classes directly onto the ADE20K classes the ControlNet was trained on, exactly where ADE20K has the class and to the closest class otherwise. Originals of the edited files: `_backup_before_sanpo_ade/`.

**How the unclear classes were decided.** SANPO's original masks for the selected frames were downloaded (`data_prep/sanpo/fetch_raw_masks.py`, 47 MB) and four large examples of each unclear class were inspected. Findings: "obstacle" is mostly snow piles plus bins and barriers; "wall/fence" is mostly metal fences; "terrain" is mostly lawns; "vegetation" mostly bushes and hedges; "other walkable surface" paved plazas; "bike rack" metal racks and bike docks; "inaccessible surface" mixed; "bus stop" does not occur.

**Result** (`data_prep/sanpo/verify_mapping_against_ade20k.py`, 0 mismatches with the official table):

| Kind | Classes | Pixel share |
|---|---|---|
| Exact ADE20K class | road, sidewalk, guard rail → railing, paved trail → path, building, wall/fence → fence, hand rail → bannister/handrail, door, pedestrian → person, animal, stairs, water body → water, vehicle → car, traffic sign → signboard, traffic light, pole, sky, tree, vegetation → plant, terrain → grass | 87.3 % |
| Closest ADE20K class | curb → sidewalk, crosswalk → road, other walkable surface → sidewalk, opening-gate → fence, rider → person, railway track → road, bike rack → railing | 6.4 % |
| Void (white) | unlabeled, inaccessible surface, bus stop, obstacle | 6.4 % |

Changes compared with the first version: paved trail (17 % of pixels) gets ADE "path" instead of sidewalk; vegetation gets "plant" and terrain "grass" instead of tree/earth; hand rail, guard rail and bike rack get railing/handrail instead of fence or void; water and animal get their own ADE classes instead of void.

**Code.**
- Labels in `data_sanpo/` now keep SANPO's own ids (`data_prep/sanpo/build_labels.py`); `data_sanpo/LABEL_SPACE` = `sanpo`.
- `src/config.py`: new `CLASS_VERSION = "sanpo"` (31 classes), chosen automatically from the data root's `LABEL_SPACE`; a mismatch raises an error.
- `src/data/dataset.py`: per class version the raw id range, the colour palette and the segmentor target (`SANPO_TO_V1` → `V1_TO_SEG`); `inference.py` and `step8_downstream/train_segmentor.py` follow the active version.
- `data_prep/generate_prompts.py`: `--label-space sanpo`. Regenerated prompts are identical to the earlier ones (checked on the test split).

**Checks.** Dataset-2 checks 12/12 passed (including the label-space marker); step 0 CPU checks 7/7 still pass; with the default data root the project still runs dataset 1 as `v2` with 21 classes.

---

## Step 9 — Audit of the dataset-2 data and mask pipeline (2026-10-17, CPU only)

Script: `step9_sanpo_audit/audit_pipeline.py` (outputs `audit_run1.txt`, `audit_run2.txt`). Each stage is compared with an independent re-computation. Final result: **21/21 checks passed**.

| Stage | What was verified |
|---|---|
| Raw SANPO mask → stored label | label equals the nearest-resized original mask pixel for pixel (40 frames); class shares unchanged; images 1920×1080 |
| Label → dataset sample | crop geometry (910×512, left offset 199) identical for mask and photo; conditioning colours equal the ADE20K colour of each id; palette tensor in id order |
| Sample → generation pipeline | float → 8-bit conversion keeps all 31 colours (now rounded instead of truncated in `evaluation.py` / `sampling.py`, safe for any palette) |
| Segmentor target | dataset target equals SANPO → project id → segmentor class applied independently |
| Splits and prompts | train/test sessions disjoint (116 / 30); no duplicate names; one prompt row per frame; object lists in all 522 test prompts equal an independent recomputation from the masks; weather words match the recorded session weather |
| New training options | joint flip moves photo, mask, colours and segmentor target together; evaluation subset is fixed, 150 samples, covers all 30 test sessions; downstream tests still use all 522 |

The first run reported one failure (5 "unsupported" prompt words). That was a bug in the check, which did not parse the ", and fence visible" wording; with correct parsing there are 0 mismatches.

**Finding: the frozen segmentor is much weaker on SANPO.** On 30 real SANPO test photos it reaches mIoU 0.41–0.45 and pixel accuracy 0.85 (dataset 1: 0.80 / 0.96); road 0.26–0.36, car 0.39–0.64, sidewalk 0.81. So on dataset 2 the mIoU metric has a low ceiling and the Phase 2 loss gets a less reliable signal.

## Step 10 — configuration for dataset 2 (prepared, not yet run)

Changes to the code: `VAL_SUBSET_SIZE` (fixed evaluation subset), `EVAL_KID_SUBSET` and `NUM_WORKERS` as settings, `TRAIN_HFLIP` (joint horizontal flip of photo and mask during ControlNet training). Defaults keep dataset 1 runs unchanged.

| Setting | Dataset 1 (best run) | Dataset 2 plan | Why |
|---|---|---|---|
| Prompts | objects-only | `prompts_meta.csv` (recorded weather + environment + objects) | random style words hurt on dataset 1; SANPO has real weather labels |
| LR / rank / alpha / dropout | 1e-4 / 16 / 16 / 0 | same | best on dataset 1; LRs 5e-5–2e-4 were equal |
| Batch × accumulation | 4 × 1 | same | |
| Training length | 30 epochs = 7,140 steps (plateau after ~3,500) | 8 epochs = 6,344 steps | similar number of updates; 3.3× more images, so fewer passes |
| Evaluation | 50 images every 5 epochs | fixed 150-image subset every 2 epochs; KID subset 50 | larger test split; 150 keeps cost per evaluation at ~9 min |
| Augmentation | none | joint horizontal flip | free, exact labels, no left/right words in prompts. Random crops were not added because prompts list objects of the centre crop |
| Checkpoint selection | lowest KID | lowest KID | |
| Phase 2 | weight 0.01 best (objects prompts) | weight 0.01 + control, 3 epochs from the Phase 1 LoRA, last epoch kept | |

Submission script: `step10_sanpo/submit_stage_a.sh` (Phase 1 → Phase 2 + control). Estimated cost about 4.3 A100 GPU-hours.

## Output naming, plots, and a random-state bug (2026-10-17)

**Dataset name in every output.** `config.DATASET_NAME` comes from the data root (`sanpo` for `data_sanpo/`); runs on dataset 2 get the run id `sanpo_<jobid>`, so logs (`metrics_sanpo_<id>.csv`, `sanpo_<id>_config.json`, `sanpo_<id>_final_eval.json`), checkpoints (`sanpo_<id>_LoRA.pt`, `sanpo_<id>_Full/`), plots (`sanpo_<id>_overview.png`, …) and sample folders (`sanpo_<id>_epoch…`) all carry it. SLURM job names and output folders also start with `sanpo_`. Dataset-1 names are unchanged.

**Plots reworked** (`src/plotting.py`). Previous set: loss curve (train and validation on one axis, which hid the validation trend), FID, KID, LPIPS and gradient-norm curves. New set, written after every epoch:
- `<run>_overview.png` — KID, segmentor mIoU (with the real-photo ceiling), LPIPS, CLIP image–prompt, CLIP image–real photo, validation loss; each panel shows the untrained model (epoch 0) as a dashed reference line.
- `<run>_diagnostics.png` — train loss, gradient norm, learning rate, and Phase-2 segmentation loss when used.
- FID is no longer plotted (unreliable at 50–150 images; still in the CSV and JSON).

**Bug found through the new diagnostics plot: repeated training noise after every evaluation.** Since go 2 (when KID subsets were seeded for reproducibility), `evaluation.py` called `torch.manual_seed(0)` and restored only the CPU random state. `manual_seed` also resets the GPU generator, so the training epoch after every evaluation drew exactly the same noise and timesteps. Visible as a regular dip in epoch train loss (e.g. job 1836710: 0.141 after each evaluation vs 0.155 otherwise). Fixed with `torch.random.fork_rng` over CPU and all GPUs.
- Affected runs: the objects-prompt Phase 1 run (1836710; 6 of 30 epochs), all Phase 2 runs (1836716–19, 1836957–59; every second epoch), the Phase 2 smoke test. Not affected: the Phase 1 learning-rate sweep, all three-seed evaluations (generation only), all downstream segmentor runs.
- Expected impact: small. Images were still shuffled differently, so only the noise draws repeated, and control and segmentation-loss runs had identical schedules, so their comparison is like for like. The runs are not repeated because of the GPU budget; the report should mention it.

## Step 10 — SANPO Phase 1 submitted (2026-10-17)

`step10_sanpo/submit_phase1.sh`: LR 1e-4, rank 16, dropout 0, batch 4, 8 epochs (6,344 steps), `prompts_meta.csv`, joint horizontal flip, evaluation on a fixed 150-image subset every 2 epochs (plus epoch 0), KID subset 50. Estimated 2.3 GPU-hours. Phase 2 on SANPO is postponed: the frozen segmentor is too weak on SANPO (mIoU ≈ 0.43 on real photos) to be a useful judge; a better one is needed first. The earlier combined Phase 1 + Phase 2 script is kept as `_draft_submit_stage_a_with_phase2.sh`.
- SANPO Phase 1 job: 1843065

---

## Step 11 — hyperparameter screen before the final runs (submitted 2026-10-17)

**Epoch choice from the logs (dataset 1).** KID of the objects-prompt run (1836710) by epoch: 5 → 0.025, 10 → 0.013, 15 → 0.012, 20 → 0.010, 25 → 0.011, 30 → 0.010. With original prompts at LR 1e-4: 0.026 / 0.024 / 0.020 / 0.024 / 0.021 / 0.021. Quality plateaus at about 15–20 epochs (3,600–4,800 steps); later changes are within single-seed noise. Plan for the final dataset-1 run: 20 epochs.

**New settings** (`src/config.py`): `DDPM_LORA_TARGETS=attn_ff` (also adapts `proj_in`, `proj_out`, `ff.net.0.proj`, `ff.net.2` of the 7 ControlNet transformer blocks, in addition to attention and the conditioning input conv), `DDPM_GUIDANCE_SCALE`, `DDPM_CONTROLNET_CONDITIONING_SCALE`. `jobs/baseline_eval.sh` passes `EXTRA_ARGS` to `evaluate.py`.

**Runs** (`step11_hparam_screen/submit_screen.sh`, about 2.9 GPU-hours). Dataset 1, objects prompts, LR 1e-4, batch 4, one seed; all four trainings use the same 10-epoch schedule so they are comparable with each other.

| Job | Name | Change vs reference |
|---|---|---|
| 1843072 | `reference` | none (rank 16, attention + conditioning conv, no flip) |
| 1843073 | `rank32` | LoRA rank 32, alpha 32 |
| 1843074 | `attn_ff` | LoRA also on transformer projections and feed-forward layers |
| 1843075 | `hflip` | joint horizontal flip of photo and mask |
| 1843076–78 | `guidance4.0/6.0/7.5` | no training: best existing checkpoint (1836710) evaluated at guidance 4.0, 6.0, 7.5 × control strength 0.8, 1.0, 1.2 |

Single-seed differences below about 0.003 KID / 0.02 mIoU are within noise (step 7).

---

## FINAL runs submitted (2026-10-17)

**Bug found and fixed just before submission: job scripts overwrote the chosen dataset.** `jobs/train.sh`, `baseline_eval.sh`, `comparison_grid.sh`, `downstream.sh` and `step8_downstream/run_ideas.sh` set `DDPM_DATA_ROOT` to dataset 1 unconditionally, so a job submitted for SANPO would have run on dataset 1 without any error. No completed job was affected: every job that ran used dataset 1 on purpose, and the first SANPO job (1843065) was cancelled while still queued. Fix: the scripts keep a data root chosen by the submitter and print it; new safety check `DDPM_EXPECT_DATASET` in `src/config.py` stops a job whose data root is not the dataset it expects (tested: SANPO-expecting job with the default data root raises an error).

**Final recipe** (`FINAL/submit_final.sh`, job list in `FINAL/submitted_jobs.txt`): LR 1e-4, LoRA rank 16 / alpha 16 on attention + conditioning conv, dropout 0, batch 4, KID-based checkpoint selection, no augmentation (the screen in step 11 tests rank 32, extra LoRA targets, flip and inference settings; changes are applied only if they clearly win).

| Dataset | Prompts | Epochs (steps) | Evaluation during training | Train job | 3-seed eval | Pretrained 3-seed | Sample sheet |
|---|---|---|---|---|---|---|---|
| sidewalk (dataset 1) | `prompts_objects.csv` | 20 (4,760) | all 50 test images every 4 epochs | 1843084 | 1843085 | 1843086 | 1843087 |
| SANPO (dataset 2) | `prompts_meta.csv` | 8 (6,344) | fixed 150-image subset every 2 epochs, KID subset 50 | 1843088 | 1843089 | 1843090 | 1843091 |

Outputs: `FINAL/<dataset>/run/` (logs, checkpoints, plots, samples), `FINAL/<dataset>/eval/` (three-seed JSONs), `FINAL/<dataset>/comparison/` (side-by-side sheet). Estimated cost 1.8 + 3.3 A100 GPU-hours.

## Clean-up and run index (2026-10-17)

With the user's approval, the large merged model copies (`<run>_Full/`, 1.4 GB each) of 17 superseded runs were deleted (about 22 GB): all 14 test runs in `controlnet_testing/step4–6`, `outputs/checkpoints/1693046_Full`, `1698835_Full`, and `test/outputs/checkpoints/1587490_Full`. Nothing else was deleted: every run keeps its logs, metrics CSV, config, final-evaluation JSON, plots, samples and its LoRA file `<run>_LoRA.pt` (9–56 MB), which rebuilds the merged model. Run 1698836 in `outputs/` (newest pre-audit run) is kept complete for reference.

`RUN_INDEX.md` (generated by `reporting/index_runs.py`, re-run after every round) lists every training run with dataset, name, folder, settings, epochs, best KID, final metrics, kept checkpoint files and known issues (old pipeline, RNG bug), plus all evaluation folders and downstream results.

---

## Step 12 — code review fixes (2026-10-17)

A code review (`/code-review`) reported 10 findings. All queued jobs (finals 1843084–91, screen 1843072–78) were cancelled while still pending (no GPU time lost), all 10 were fixed, and everything was resubmitted. Originals of edited files: `_backup_before_review_fixes/`. Tests: `step12_review_fixes/test_review_fixes.py` 16/16; earlier suites re-run: step 0 7/7 (plot expectation updated to the new two-figure set), Phase-2 pieces 5/5, SANPO audit 21/21, dataset checks 12/12 and 7/7.

| # | Finding | Fix | Did it affect results so far? |
|---|---|---|---|
| 1 | `generate_prompts.py` ignored the data root's `LABEL_SPACE`; the documented SANPO command without `--label-space sanpo` would read SANPO ids as project ids (wrong object words) | label space read from `<data-root>/LABEL_SPACE`; a contradicting `--label-space` stops with an error | No: the SANPO prompt files were made with `--label-space sanpo`; regenerating without the flag gives identical files |
| 2 | segmentor target (`mask_seg`) built before mask augmentation, conditioning after it | augmentation now runs first on the raw ids; both are derived from the augmented mask | No: mask augmentation has been off in every run |
| 3 | unknown `CHECKPOINT_METRIC` (e.g. `kid_mean`) silently fell back to validation loss | only `kid`, `val_loss`, `last` accepted; anything else stops at start-up | No: all runs used valid values |
| 4 | final evaluation regenerated the whole val set for the checkpoint already evaluated during training | metrics of the selected epoch are stored and reused (same weights, images, seeds and settings) | No effect on numbers; saves one full evaluation per run |
| 5 | runs started from `INIT_LORA_CHECKPOINT` called epoch 0 the "untrained baseline" | epoch 0 is now "start of run" (pretrained model or the given checkpoint) in logs, prints and plots | Wording only: Phase 2 tables already described epoch 0 as the Phase 1 start |
| 6 | new plotting misread pre-audit CSVs (0-based trained epoch shown as "before training") | old format detected and shifted by one, no reference line | Plots only |
| 7 | with the Phase 2 loss, two copies of the frozen segmentor were loaded | the evaluator reuses the model's segmentor | No effect on numbers; less GPU memory |
| 8 | segmentor re-run on the real val images at every evaluation although the result never changes | computed once per val set and cached | No effect on numbers; faster evaluations |
| 9 | dead stubs `load_segmentor` / `evaluate_miou` (returning 0.0) still in the evaluator | removed; class docstring updated | No |
| 10 | diagnostics plot removed reference lines after drawing them | `_panel(..., reference=False)` | Plots only |

**Resubmitted** (job lists in `FINAL/submitted_jobs.txt` and `step11_hparam_screen/submitted_jobs.txt`):
- FINAL sidewalk: train 1843106, three-seed eval 1843107, pretrained three-seed 1843108, sample sheet 1843109.
- FINAL SANPO: train 1843110 (run id `sanpo_1843110`), three-seed eval 1843111, pretrained three-seed 1843112, sample sheet 1843113.
- Screen: reference 1843114, rank32 1843115, attn_ff 1843116, hflip 1843117, guidance 4.0 / 6.0 / 7.5 sweeps 1843118–1843120.
- 2026-10-17: to save about 2.4 GPU-hours, the four screen training runs (1843114–1843117: reference, rank32, attn_ff, hflip) were cancelled before starting, with the user's agreement. The finals run with the proven recipe; only the inference-time guidance × control-strength sweep (1843118–1843120) is kept, since its result can be applied to the final models without retraining. Remaining queue ≈ 6 GPU-hours.
- 2026-10-17: correction — the user wanted the screen kept ("6 hours is okay" meant the full queue). The four screen training runs were resubmitted: reference 1843131;rank32 1843132;attn_ff 1843133;hflip 1843134;. Queue back to ≈ 8.4 GPU-hours.
- 2026-10-17: FINAL Phase 2 on dataset 1 queued (`FINAL/submit_final_phase2_sidewalk.sh`, starts after Phase 1 job 1843106): control and segmentation-loss weight 0.01, 10 epochs at LR 5e-5 from the final Phase 1 LoRA, last epoch kept, each with a three-seed evaluation. Jobs: control train 1843135;control eval x3 1843136;segloss_w0.01 train 1843137;segloss_w0.01 eval x3 1843138; ≈ 1.8 GPU-hours. SANPO Phase 2 is not queued: the frozen segmentor is too weak on SANPO (mIoU ≈ 0.43 on real photos).
- 2026-10-17: FINAL sidewalk Phase 2 jobs 1843135–1843138 cancelled before starting (user: check Phase 1 results first). Script kept: `FINAL/submit_final_phase2_sidewalk.sh` (set P1 to the final Phase 1 job id before reuse).

---

## 2026-10-18 — A100 quota full, move to V100, and a third data-path bug

**Queue blocked.** All 15 jobs waited about 14 hours with reason `AssocGrpGRES`: the group's A100 allowance was used up by other jobs. Two 2-minute probe jobs started immediately on V100 and RTX 3080, so only A100s were blocked. RTX 3080 (10 GB) is too small for training (peak ≈ 17 GB); V100 (32 GB, Tesla V100-PCIE-32GB) is used instead.

**V100 support.** V100 has no native bf16. `src/train.py` now detects GPUs below compute capability 8.0 and switches to fp16 with gradient scaling (logged as `[INFO] … using fp16`). `MIXED_PRECISION` is now a setting (`DDPM_MIXED_PRECISION`). Submission scripts request `GPU=${GPU:-v100}` with doubled time limits (V100 ≈ half the speed of A100). Results from now on are trained in fp16 instead of bf16; this is a numerical-precision change, not a method change.

**Bug: conda activation reset the data root to dataset 1.** On V100 the two SANPO jobs failed within a minute: the `DDPM_EXPECT_DATASET` guard stopped them because the data root was `data/`. Cause: the environment's activation script `control2/etc/conda/activate.d/env_vars.sh` exports `DDPM_DATA_ROOT=$DDPM_ROOT/data` on every `conda activate`, overwriting the value passed by the submitter. Without the guard these jobs would have trained and evaluated on dataset 1 under SANPO names. Earlier completed runs all used dataset 1 on purpose, so none is affected. Fix (the user's conda environment is left unchanged): every job script stores the requested `DDPM_DATA_ROOT` before `conda activate` and restores it afterwards. Verified by running the real `train.sh` in a fresh login environment in a new dry-run mode (`DRY_RUN=1`): SANPO → `data_sanpo`, 31 classes; default → `data`, 21 classes.

**All jobs cancelled; canary first** (user request, to avoid filling the V100 quota with jobs that may all fail): `FINAL/canary/submit_canary.sh` — SANPO, 1 epoch, 20-image evaluation subset, V100 fp16; train → checkpoint evaluation → sample sheet. Jobs 1843490 (train, run id `sanpo_1843490`), 1843491 (eval), 1843492 (sheet). The 15 final/screen jobs are released only after its logs are checked.

**Canary passed (2026-10-18).** Train job 1843490 (V100, 19 min): data root `data_sanpo`, fp16 fallback active, peak 16.7 GB of 32; epoch-0 and epoch-1 evaluations with mIoU and real-photo reference; KID-based selection; merged model saved; final evaluation reused the stored metrics; plots and samples named `sanpo_1843490_*`. Evaluation job 1843491 and sheet job 1843492 were moved to RTX 3080 (generation only) and passed: peak 9.5 GB and 9.2 GB of 10 GB. The standalone evaluation reproduced the in-training numbers on a different GPU (KID 0.0150 vs 0.0152, mIoU 0.279 vs 0.277). One cosmetic fix: final sample folders are now labelled with the epoch of the selected checkpoint (was the 0-based index of the last epoch). Canary outputs kept in `FINAL/canary/`.

**Released**: training on V100, evaluation/baseline/sheet/guidance jobs on RTX 3080 (`EVAL_GPU`, close to the 10 GB limit; fall back to V100 if one runs out of memory).
- sidewalk train 1843532 (run id 1843532)
- sidewalk eval final x3 1843533
- sidewalk eval pretrained x3 1843534
- sidewalk sample sheet 1843535
- sanpo train 1843536 (run id sanpo_1843536)
- sanpo eval final x3 1843537
- sanpo eval pretrained x3 1843538
- sanpo sample sheet 1843539
- screen reference 1843540
- screen rank32 1843541
- screen attn_ff 1843542
- screen hflip 1843543
- screen guidance 4.0 1843544
- screen guidance 6.0 1843545
- screen guidance 7.5 1843546

---

## FINAL runs and screen — results (jobs ran 2026-10-18; checked 2026-10-23 after a 2–3 day HPC outage)

All 15 jobs ran on 2026-10-18, before the outage. 12 completed; the 3 guidance-sweep jobs failed (see below). No errors in the final or screen logs; V100 training peaked at 17.0 GB (dataset 1) and 17.6 GB (SANPO).

**FINAL, three generation seeds** (`FINAL/<dataset>/eval/`). Dataset 1: all 50 test images; SANPO: fixed 150-image subset.

| Dataset | Model | KID ↓ | FID ↓ | LPIPS ↓ | CLIP image ↑ | mIoU ↑ | mIoU real photos |
|---|---|---|---|---|---|---|---|
| sidewalk | pretrained | 0.085 ± 0.006 | 188 ± 8 | 0.647 | 0.687 | 0.363 ± 0.027 | 0.802 |
| sidewalk | **final** (job 1843532, epoch 20) | **0.014 ± 0.002** | **130 ± 3** | **0.588** | **0.795** | **0.436 ± 0.007** | 0.802 |
| SANPO | pretrained | 0.031 ± 0.001 | 154 ± 1 | 0.710 | 0.753 | 0.389 ± 0.005 | 0.444 |
| SANPO | **final** (job 1843536, epoch 8) | **0.009 ± 0.000** | **136 ± 1** | **0.668** | **0.802** | 0.394 ± 0.005 | 0.444 |

- Dataset 1: KID 6× lower, FID −58, mIoU +0.07 over the pretrained model.
- SANPO: KID 3.4× lower, FID −18; mIoU unchanged, but the frozen segmentor only reaches 0.444 on the real SANPO photos, so it cannot measure mask adherence there.
- SANPO quality levels off early: KID 0.011 after 2 epochs, 0.009 after 8.
- Dataset-1 note: the earlier 30-epoch objects-prompt run (1836710, bf16, RNG bug) measured KID 0.008 ± 0.002 with three seeds; the final run is 0.014 ± 0.002. Differences: 20 instead of 30 epochs (shorter cosine schedule), fp16 instead of bf16, RNG bug fixed. The final run was still improving at epoch 20 (KID 0.017 → 0.013 between epochs 12 and 20).

**Screen, dataset 1, 10 epochs, one seed** (`step11_hparam_screen/`):

| Run | KID ↓ ep 5 / ep 10 | LPIPS ↓ ep 10 | mIoU ↑ ep 10 |
|---|---|---|---|
| reference | 0.026 / 0.018 | 0.596 | 0.434 |
| rank 32 | 0.020 / 0.017 | 0.595 | 0.415 |
| **attn + ff** | **0.016 / 0.014** | **0.583** | **0.452** |
| flip | 0.023 / 0.019 | 0.595 | 0.424 |

LoRA on the transformer projections and feed-forward layers as well (attn + ff) is best on every metric at both evaluations; after 10 epochs it matches the 20-epoch final (KID 0.014 vs 0.013). Rank 32 and flip are within noise of the reference.

**Guidance sweep failed: out of memory on RTX 3080.** Each job evaluated three control strengths in one process; every `evaluate()` call created new Inception/LPIPS metric models and kept the old ones, so memory grew until the third call failed (9.9 of 10 GB). No result files were written. Fix in `src/evaluation.py`: metric models are freed at the end of every evaluation. To be rerun (≈ 0.6 GPU-hours).

---

## Step 13 — simpler prompts, a SANPO judge, generation settings (2026-10-23)

**Why.** SANPO final images often look hazy/washed out and thin objects (bicycles, people) are distorted, while KID/FID look good. Causes found from the samples: (1) the SANPO camera's own look (hazy, low contrast, flare, motion blur) is learned; (2) objects without a SANPO class (e.g. bicycles) are white void in the mask, so the model invents them; (3) thin/small objects are hard for SD 1.5 at 512 px; (4) limited adapter capacity. FID/KID compare whole image sets and hardly see local distortions. The user also wants higher mIoU on dataset 2, but the lab's segmentor reaches only mIoU ≈ 0.44 on real SANPO photos, so it can neither measure nor drive mask adherence there.

**Prompt variants** (`data_prep/generate_prompts.py`, both datasets; old files kept). Current `prompts_meta.csv` names objects in a fixed priority order regardless of size, with stiff wording (mean 2 objects, 26 tokens; e.g. "street-level photo of a sidewalk, urban street, sunny day, pedestrian, car, street pole, and fence visible").
- `prompts_simple.csv`: "a photo of a sidewalk" + recorded weather, e.g. "a photo of a sidewalk, sunny day". All layout comes from the mask.
- `prompts_natural.csv`: up to 3 LARGEST objects (≥ 0.5 % of the crop) in plain words + recorded weather, e.g. "a photo of a sidewalk with a fence, stairs and people, sunny day".

**Negative prompt** setting `DDPM_NEGATIVE_PROMPT` (evaluation, sampling, comparison sheets; recorded in evaluation JSONs). Default empty = unchanged behaviour.

**SANPO judge** (`src/segmentor.py`, `DDPM_SEG_JUDGE=sanpo`; training script `step13_sanpo_judge/train_judge.py`). U-Net with ResNet-50 encoder, trained on SANPO's real training photos (same 512 crop as the ControlNet, joint flip, 12,000 steps, batch 8), tested on all 522 real test photos (its ceiling). 13 classes: road (+ railway), sidewalk (+ curb, paved trail, other walkable surface), crosswalk, building (+ door), fence (+ wall/fence, guard rail, hand rail, gate, bike rack), person (+ rider), vehicle, pole, sign (+ traffic light), vegetation (tree + vegetation), terrain, sky, stairs; ignored: unlabeled, animal, water, inaccessible surface, obstacle, bus stop. Usable for mIoU and for a later Phase-2 loss on dataset 2. The dataset refuses `DDPM_SEG_JUDGE=sanpo` on dataset 1. CPU checks: judge target equals the mapping applied to the label; model forward gives 13 channels.

**Jobs** (V100): 1848170 judge training (≈ 1.5 GPU-h); 1848171 inference-only test on the SANPO final model after the judge exists (≈ 1.2 GPU-h), all on the 150-image subset, seed 42, mIoU from the new judge:

| Tag | Model | Prompts | Negative prompt | Guidance |
|---|---|---|---|---|
| pretrained_meta_noneg_g6 | pretrained | meta | – | 6 |
| final_meta_noneg_g6 | SANPO final | meta | – | 6 |
| final_meta_neg_g6 | SANPO final | meta | yes | 6 |
| final_simple_neg_g6 | SANPO final | simple | yes | 6 |
| final_natural_neg_g6 | SANPO final | natural | yes | 6 |
| final_meta_neg_g4 / _g7.5 | SANPO final | meta | yes | 4 / 7.5 |

Negative prompt used: "blurry, distorted, deformed, hazy, low contrast, washed out, overexposed". Note: the final model was trained with `prompts_meta`; the simple/natural rows only test them at generation time. Training with a new prompt set is part of the planned v2 run.

**Step 13 withdrawn (2026-10-23, user decision).** Jobs 1848170/1848171 cancelled before starting; `prompts_simple.csv` / `prompts_natural.csv` (both datasets) and `step13_sanpo_judge/` deleted; prompt-variant and judge code reverted to `_backup_before_prompts_judge/` (the negative-prompt setting stays, default off). Reasons:
- Prompts: "simpler" meant prompts close to the real conditions of each image *with* the object cues (car, street pole, fence, …), not stripped text. The SANPO `prompts_meta.csv` already is exactly that (recorded weather/environment + objects) and stays.
- Judge: a segmentor trained on SANPO's real photos favours real images by construction, so comparing real vs generated with it is biased. Instead, use a stronger general-purpose pretrained model that saw neither dataset. Candidate: Mask2Former Swin-L trained on Mapillary Vistas (`facebook/mask2former-swin-large-mapillary-vistas-semantic`, 65 street-level classes incl. sidewalk, curb, crosswalk, bike lane, person, rider, bicycle, car, pole, sign, fence, vegetation, terrain, sky). Being cached in `$WORK/huggingface_cache` for a ceiling test on the real test photos of both datasets.

## Clean-up, prompt checks and plot fixes (2026-10-23, CPU only)

**Rule from now on:** every SLURM job is submitted only after the user's explicit OK.

**CLIP token limit.** Every prompt file was tokenized with the SD 1.5 CLIP tokenizer (limit 77 tokens including start/end): longest prompt 50 tokens (dataset 1 `prompts.csv`), SANPO `prompts_meta.csv` max 38; **no prompt is truncated** in any file.

**Weather variety in SANPO prompts.** SANPO's own session labels say "sunny" for 99.6 % of frames (sunny 2,477; sunny + snowy 1,116; sunny + cloudy 86; cloudy 15), so "sunny day" appears in 100 % of training prompts and "snow on the ground" in 31 %. The labels are per recording session, not per frame.

**Deleted (noise from cancelled or bug-failed attempts; each has a successful rerun):**
- logs of cancelled/failed jobs: FINAL 1843452, 1843454, 1843456, 1843458; screen 1843460–1843464 and the out-of-memory guidance jobs 1843544–1843546; header-only metrics files and configs of 1843452, 1843460, 1843461, 1843462;
- `step10_sanpo/` (draft scripts of cancelled jobs), empty `FINAL/sidewalk_phase2/` (the Phase-2 script stays in `FINAL/`), empty `step11_hparam_screen/inference_sweep/`;
- large merged model copies of the four screen runs and the canary (LoRA files kept);
- SANPO `prompts.csv` and `prompts_objects.csv` (never used for training); SANPO now defaults to `prompts_meta.csv`.
- Code backups grouped in `controlnet_testing/_code_backups/`.

**Plot fixes** (`src/plotting.py`), final and screen plots redrawn: whole-number epoch ticks (were 2.5, 7.5, …); the train-loss panel is labelled as noisy by nature, with a note that the validation loss is the convergence signal. Remaining reading caveat for the report: panels use zoomed y-axes, so changes inside the noise (e.g. SANPO mIoU 0.383 → 0.397) look larger than they are.

---

## Step 14 — new judge, prompt augmentation, FINAL v2 (2026-10-23)

**Prompt decision** (delegated by the user): keep every prompt truthful (recorded conditions + object cues, never a condition the image does not show), but vary the wording during training. New setting `DDPM_TRAIN_PROMPT_KEEP_PROB` (`src/data/dataset.py: augment_prompt`): each condition phrase and each object of the true prompt is kept with this probability and the order is shuffled; evaluation always uses the full prompt. Check: all 4,122 prompts of both datasets split and rebuild exactly. Example for one SANPO frame: "street-level photo of a sidewalk, sunny day, urban street, car and pedestrian visible" / "…, snow on the ground, road junction, sunny day, fence and car visible".

**New judge: Mask2Former Swin-L trained on Mapillary Vistas** (`DDPM_SEG_JUDGE=mapillary`, `src/segmentor.py: MapillaryJudge`), never trained on either dataset. Its 65 classes and both datasets' labels are mapped to 12 shared classes + "other": road (incl. crosswalk, bike lane, parking, rail track), sidewalk (incl. curb, paved trail, pedestrian area), building, fence/wall, person, vehicle, bicycle, pole, sign/light, vegetation, terrain, sky. Not scored: void/unlabeled, stairs, bridge, obstacle, water, animal, bus stop, bike rack. Mapillary predictions without a counterpart (snow, bench, trash can, …) go to "other" and count as wrong. Input upscaled 512 → 1024 px (closer to Mapillary's training size); judged one image at a time to limit GPU memory.
- CPU check on 6 real test photos per dataset: SANPO mIoU 0.73, pixel acc 0.95 (old judge 0.44 on SANPO); dataset 1 mIoU 0.65, pixel acc 0.94. Crosswalk scored 0.00 on both because Mapillary labels only the painted stripes as crosswalk while our masks mark the whole crossing, so crosswalk is merged into road.
- Test suites re-run: review fixes 16/16 (call-count expectation updated for per-image judging), step 0 7/7.

**Submitted (user approved; ≈ 9 V100 GPU-hours)** — `FINAL_v2/submit_v2.sh`, job list `FINAL_v2/submitted_jobs.txt`:
- stage1 pretrained sidewalk 1848214
- stage1 final sidewalk 1848215
- stage1 pretrained sanpo 1848216
- stage1 final sanpo 1848217
- stage2 sidewalk train 1848218 (run id 1848218)
- stage2 sidewalk eval x3 1848219
- stage2 sidewalk sheet 1848220
- stage2 sanpo train 1848221 (run id sanpo_1848221)
- stage2 sanpo eval x3 1848222
- stage2 sanpo sheet 1848223
- Stage 1 re-scores the pretrained model and the v1 finals with the Mapillary judge (three seeds) into `FINAL/<dataset>/eval_mapillary/`; stage 2 starts only if both final-model re-scores succeed (canary for the new judge code).
- v2 recipe: v1 recipe + LoRA on transformer projections and feed-forward layers (`DDPM_LORA_TARGETS=attn_ff`) + prompt augmentation (keep probability 0.7) + Mapillary judge during training; same epochs (20 / 8), LR, rank, prompt files and evaluation sets as v1. Outputs in `FINAL_v2/<dataset>/`; the sample sheets compare v1 and v2 side by side.

## Step 14 — results (re-scoring finished 2026-10-23, v2 finished 2026-10-24)

All 10 jobs completed without errors (v2 training peak 17.8 GB on V100). Three seeds each; Mapillary judge; dataset 1 on all 50 test images, SANPO on the fixed 150-image subset.

| Dataset | Model | KID ↓ | FID ↓ | LPIPS ↓ | CLIP image ↑ | mIoU ↑ | mIoU real photos |
|---|---|---|---|---|---|---|---|
| sidewalk | pretrained | 0.085 ± 0.006 | 188 ± 8 | 0.647 | 0.688 | 0.402 ± 0.016 | 0.795 |
| sidewalk | v1 final (1843532, ep 20) | **0.014 ± 0.002** | 130 ± 3 | 0.588 | 0.795 | 0.584 ± 0.015 | 0.795 |
| sidewalk | v2 (1848218, ep 12) | 0.018 ± 0.001 | 131 ± 3 | **0.578** | **0.817** | **0.607 ± 0.029** | 0.795 |
| SANPO | pretrained | 0.031 ± 0.001 | 154 ± 1 | 0.710 | 0.753 | 0.458 ± 0.002 | 0.627 |
| SANPO | v1 final (1843536, ep 8) | **0.009 ± 0.000** | 136 ± 1 | 0.668 | 0.802 | 0.489 ± 0.003 | 0.627 |
| SANPO | v2 (1848221, ep 8) | 0.010 ± 0.001 | 136 ± 1 | **0.662** | **0.810** | **0.507 ± 0.007** | 0.627 |

- **New judge:** on real photos mIoU 0.795 (dataset 1) and 0.627 (SANPO; the old judge reached 0.44 there). Fine-tuning raises mask adherence clearly on dataset 1 (0.40 → 0.58) and modestly on SANPO (0.46 → 0.49; sidewalk class 0.39 → 0.61).
- **v2 vs v1:** v2 is better on the per-image measures (mIoU +0.02 on both datasets, LPIPS and CLIP image better). On SANPO the mIoU gain is outside the seed spread; on dataset 1 it is within it. KID is slightly worse (dataset 1 0.014 → 0.018; SANPO 0.009 → 0.010). Sample sheets show v1 and v2 very close.
- v2 changed two things at once (LoRA targets and prompt augmentation), so the two effects cannot be separated from these runs.
- Dataset-1 v2 single-seed curve: mIoU peaks early (epoch 4: 0.603) and drifts down to 0.567 by epoch 20 while KID keeps improving; the KID-selected checkpoint is epoch 12.
- SANPO road stays low for every model (≈ 0.23), also the pretrained one.

---

## Step 15 — SANPO with the segmentation loss (2026-10-24)

**15a. Check before training** (job 1848478, `step15_phase2_sanpo/check_segloss.py`, V100, SANPO v2 LoRA, no weights changed).
- Mapillary judge on the 150 real SANPO evaluation photos: input 512 px → mIoU 0.625, pixel acc 0.836; input 1024 px → mIoU 0.627, pixel acc 0.834. Enlarging gives no gain.
- Training with the loss on: at 512 px peak GPU memory 28.8 GB of 32 (forward + two separate backward passes); at 1024 px out of memory (the job therefore ended with an error after the 512 px measurement, as a result in itself).
- Strength (512 px, 30 batches; loss applied on 26): diffusion loss 0.117, segmentation loss 1.03; gradient norm on the LoRA weights 0.0058 (diffusion) vs 1.93 (segmentation at weight 1) → the segmentation term pushes ≈ 335× harder per unit weight. Weight 0.003 ≈ equal push, 0.006 ≈ 2×, 0.01 ≈ 3×. Consistent with dataset 1/old judge, where 0.01 (≈ 4×) gave the best mIoU and ≥ 0.1 damaged image quality.

**Project document** (`DDPM/ControlNet_Segmentor_Project.docx`): one fine-tuning with the joint loss L = L_diffusion + λ·L_seg, λ started low and ramped up, and an ablation "with/without semantic loss". Its numbers (λ 0.1 → 0.5, max 1.0) assume a different loss scale; on our setup λ = 0.1 would be ≈ 33× the diffusion push, so the principle (low start, ramp) is kept with the measured scale. The document's S is the lab's segmentor; on SANPO it is replaced by the Mapillary judge (lab segmentor mIoU 0.44 on real SANPO photos).

**Decision (user):** train from the **pretrained** model, not by continuing fine-tuned B — continuing B on the same images for more epochs pushes towards memorising them, while a run from pretrained sees the images exactly as long as B did. Weight 0.006 chosen by the user.

**15b. Run** (`FINAL_v2_segloss/submit_segloss_sanpo.sh`): identical to SANPO v2 / "fine-tuned B" (job 1848221: attn+ff LoRA rank 16, LR 1e-4, 8 epochs, prompts_meta + prompt variation 0.7, KID-based checkpoint selection, 150-image evaluation subset) plus the segmentation loss: λ ramps 0 → 0.006 over the first epoch (793 steps), applied to samples with timestep < 400 (max 2 per batch), judge input 512 px during training. B therefore is the "without semantic loss" side of the ablation. Then a three-seed evaluation scored exactly like B (judge at 1024 px) and a sample sheet pretrained | B | B + seg loss.
- Jobs: train 1848504 (run id `sanpo_1848504`), evaluation 1848505, sheet 1848506. ≈ 6 V100 GPU-hours.

**15b. Results** (all three jobs completed 2026-10-24; training 4 h 22 min; peak GPU memory 32.5 GB of the 32 GB V100, i.e. at the limit). Three seeds, Mapillary judge at 1024 px, 150-image SANPO subset, scored exactly like B.

| Model | KID ↓ | FID ↓ | LPIPS ↓ | CLIP prompt ↑ | CLIP image ↑ | mIoU ↑ | Pixel acc ↑ |
|---|---|---|---|---|---|---|---|
| pretrained | 0.031 ± 0.001 | 154 ± 1 | 0.710 | – | 0.753 | 0.458 ± 0.002 | 0.532 |
| B = v2, no seg loss (1848221) | **0.0098 ± 0.0006** | 135.7 | 0.662 | 0.308 | 0.810 | 0.507 ± 0.007 | 0.705 |
| B + seg loss (sanpo_1848504, ep 8) | 0.0107 ± 0.0008 | **133.5 ± 1.2** | 0.663 | 0.307 | 0.808 | **0.544 ± 0.008** | **0.748** |
| real photos (ceiling) | | | | | | 0.627 | |

Per class (B → B + seg): road 0.24 → 0.30, sidewalk 0.66 → 0.71, building 0.60 → 0.62, fence/wall 0.49 → 0.56, pole 0.31 → 0.37, vegetation 0.71 → 0.75, terrain 0.50 → 0.61, sky 0.56 → 0.57; person 0.81 and vehicle 0.59 unchanged; sign/light 0.10 → 0.09.

- **Ablation with/without semantic loss:** mIoU +0.037 (≈ 5× the seed spread), pixel accuracy +0.04; realism essentially unchanged (KID +0.0009 within ≈ 1.5 seed spreads, FID slightly better, LPIPS/CLIP equal). The loss closes ≈ 30 % of the gap between B and the judge's ceiling on real photos (0.507 → 0.544 of 0.627).
- Training curve (single seed, judge 512 px): mIoU ep 0 0.448, ep 2 0.507, ep 4 0.543, ep 6 0.541, ep 8 0.551; KID 0.0097 / 0.0125 / 0.0103 / 0.0096. Segmentation loss 1.15 → ≈ 1.0. KID-selected checkpoint: epoch 8.
- One logged gradient norm is nan (step 1700): an fp16 overflow step, skipped by the GradScaler as designed; training continued normally.
- Sample sheet `FINAL_v2_segloss/sanpo/comparison/comparison_sheet.jpg` (pretrained | B | B + seg loss): B and B + seg loss look alike in style; B + seg loss follows the mask layout more closely in places (e.g. lane marking / fence placement).
- Caveat: the judge used for the loss is also the judge used for scoring, so part of the gain may be judge-specific; a downstream check with another segmentor would test this.

---

## Step 16 — SANPO, strong slowly rising segmentation-loss weight (submitted 2026-10-24)

**Why:** with λ 0.006, image quality held but mIoU (0.544) is still well below the judge on real photos (0.627). The user asked for a much stronger weight, rising slowly, close to the project document's 0.1 → 0.5, in one run (not separate runs). Dataset 1 (950 images) was damaged by λ ≥ 0.1, but SANPO is 3× larger and the judge is much better now (old lab segmentor there), so it is tested again here.
**Control scale:** stays 1.0. 1.2 / 1.5 are postponed until this result is known (user decision).

**Code changes** (backups `_code_backups/*.before_lambda_ramp`):
- `DDPM_SEGMENTOR_LOSS_WEIGHT_START` (default 0): λ ramps linearly start → `SEGMENTOR_LOSS_WEIGHT` over `SEGMENTOR_LOSS_WARMUP_STEPS`, then stays constant. With the default it reproduces the earlier runs exactly.
- λ is printed on every step-log line and at the end of each epoch, saved as a new `seg_weight` CSV column and plotted in the diagnostics figure.
- `DDPM_SAVE_EVAL_CHECKPOINTS=true` keeps the LoRA of every evaluated epoch (`<run>_ep<N>_LoRA.pt`), so any point of the λ curve can get a three-seed evaluation later without retraining.
- Every job log (train / evaluation / sheet) now starts with a header: date, job, what the job tests (`RUN_NOTE`, set by the submit script), dataset, all `DDPM_*` settings.
- Tests: `step16_lambda_ramp/test_lambda_ramp_cpu.py` 8/8 (λ 0.05 at step 0, 0.225 at the midpoint, 0.4 at the end, monotone, default unchanged, plot with λ column). Step 0 (7/7, CSV now 15 columns) and step 12 (16/16) still pass.

**Run** (`FINAL_v2_segloss/submit_segloss_ramp_sanpo.sh`, outputs in `FINAL_v2_segloss/sanpo_ramp/`). Identical to the λ 0.006 run (1848504) except:
- λ 0.05 → 0.4 linearly over all 8 epochs (6,344 steps); end of epoch 1–8: 0.094, 0.138, 0.181, 0.225, 0.269, 0.312, 0.356, 0.400. In "diffusion-push" units (×335): ≈ 17× at the start, ≈ 134× at the end.
- Checkpoint = last epoch (λ 0.4 reached) instead of best KID, since KID would favour the early low-λ epochs. LoRA of epochs 2/4/6/8 kept.
- Jobs: train 1848904 (run id `sanpo_1848904`), evaluation ×3 1848905 (tag `v2_segloss_ramp_sanpo`), sheet 1848906 (pretrained | B | B + seg 0.006 | B + seg 0.05→0.4). ≈ 6 V100 GPU-hours.
- To read the result: in-training curve (single seed, every 2 epochs) gives mIoU and KID against λ; the three-seed evaluation gives the end point at λ 0.4.

**Step 16 results** (2026-10-25; train 4 h 45 min, evaluation 1 h 07 min, sheet 5 min, all completed). Three seeds, Mapillary judge at 1024 px, 150-image SANPO subset, control scale 1.0.

| Model | KID ↓ | FID ↓ | LPIPS ↓ | CLIP prompt ↑ | CLIP image ↑ | mIoU ↑ | Pixel acc ↑ |
|---|---|---|---|---|---|---|---|
| pretrained | 0.031 | 154 | 0.710 | 0.296 | 0.753 | 0.458 | 0.532 |
| B (no seg loss) | **0.0098** | 135.7 | **0.662** | **0.308** | **0.810** | 0.507 | 0.705 |
| B + seg λ 0.006 | 0.0107 | **133.5** | 0.663 | 0.307 | 0.808 | 0.544 | 0.748 |
| B + seg λ 0.05 → 0.4 (sanpo_1848904, ep 8) | 0.0451 ± 0.0030 | 152.3 ± 2.4 | 0.667 | 0.286 | 0.755 | **0.705 ± 0.008** | **0.912** |
| real photos (judge) | | | | | | 0.627 | 0.834 |

Per class (λ 0.006 → λ 0.05→0.4): road 0.30 → 0.58, sidewalk 0.71 → 0.91, building 0.62 → 0.87, fence/wall 0.56 → 0.83, person 0.81 → 0.86, vehicle 0.59 → 0.65, pole 0.37 → 0.51, vegetation 0.75 → 0.89, terrain 0.61 → 0.82, sky 0.57 → 0.74; sign/light stays 0.10.

Training curve (single seed, judge 512 px; λ = value at the end of the epoch):

| Epoch | λ | KID | FID | mIoU |
|---|---|---|---|---|
| 0 (pretrained) | – | 0.031 | 154 | 0.448 |
| 2 | 0.14 | 0.025 | 140 | 0.603 |
| 4 | 0.23 | 0.055 | 158 | 0.667 |
| 6 | 0.31 | 0.047 | 152 | 0.715 |
| 8 | 0.40 | 0.042 | 150 | 0.719 |

- **Mask adherence:** mIoU 0.705 is above the judge on the real photos (0.627): the judge segments the generated images more easily than the real ones. Every class except sign/light gains strongly.
- **Realism against SANPO:** KID 4× worse than B (0.045 vs 0.010), FID back at the pretrained level (152 vs 154), CLIP image similarity to the real photo back to pretrained (0.755). Most of the loss happens by epoch 4 (λ ≈ 0.2); afterwards KID recovers slightly while the learning rate decays.
- **Sample sheet** (`FINAL_v2_segloss/sanpo_ramp/comparison/comparison_sheet.jpg`): the images are not noisy or broken. They are sharp, bright, high-contrast "ideal" street scenes with clearly drawn objects in the mask positions (cars, fences, people, paving). They have lost the SANPO camera look (washed-out, hazy, chest-mounted camera, lens flare), which is what KID/FID measure. So the trade-off is dataset look vs clear, mask-faithful content, not image breakage.
- Caveat as before: the loss and the score use the same judge. That mIoU exceeds the real-photo value fits "images made easy for this judge". A second, independent segmentor is needed to show that the gain is general.
- fp16: 6 of 126 logged step windows contain an overflow step (grad norm nan, skipped by the GradScaler); scaler scale stayed at 4096–16384, training stable.
- Kept: `sanpo_1848904_ep{2,4,6,8}_LoRA.pt` (epoch 2 = λ 0.14 point: single-seed KID 0.025, mIoU 0.603).

**Step 16 discussion with the user (2026-10-25):** the user accepts training and scoring with the same Mapillary judge for now (that is the research setup); an independent unseen segmentor will be added as a second evaluation after the final run. Gallery of the λ 0.05 → 0.4 generations (4 scenes × 4 seeds): `FINAL_v2_segloss/sanpo_ramp/comparison/ramp_gallery_epoch8.jpg`. Seen: strict mask adherence, bright "stock-photo" look instead of SANPO's hazy camera look (winter scenes turned spring), little variety between seeds.

---

## Step 17 — SANPO, middle weight λ 0.02 (intended final run, submitted 2026-10-25)

**Goal (user):** more realistic images, mIoU somewhat lower than 0.705 but close to the real photos (0.627), and KID/FID close to B.
**Choice of λ:** per-epoch curves (single seed):

| Run | λ | KID ep 2 | CLIP image ep 2 | KID ep 8 | mIoU ep 8 |
|---|---|---|---|---|---|
| B | 0 | 0.015 | 0.805 | 0.009 | 0.505 |
| λ 0.006 | 0.006 | 0.010 | 0.805 | 0.010 | 0.551 |
| ramp | 0.05 → 0.14 (ep 2) → 0.4 | 0.025 | 0.762 | 0.042 | 0.719 |

Already at λ 0.05–0.14 the model stopped learning the SANPO look (CLIP image stayed at the pretrained level). λ 0.02 ≈ geometric middle of 0.006 and 0.05 (≈ 7× the diffusion push).
**Run** (`FINAL_v2_segloss/submit_segloss_l002_sanpo.sh`, outputs `FINAL_v2_segloss/sanpo_l0.02/`): identical to the λ 0.006 run except λ = 0.02 (0 → 0.02 over epoch 1, then constant); checkpoint = best KID; LoRA of epochs 2/4/6/8 kept.
- Jobs: train 1850042 (run id `sanpo_1850042`), evaluation ×3 1850043 (tag `v2_segloss_l002_sanpo`), sheet 1850044 (pretrained | B | 0.006 | 0.02 | 0.05→0.4). ≈ 6 V100 GPU-hours.
- Baselines for the report (three seeds, same protocol): pretrained KID 0.031 / FID 154 / mIoU 0.458; B KID 0.0098 / FID 135.7 / mIoU 0.507; real photos mIoU 0.627.

**White (void) areas check (2026-10-25, CPU only).** User noticed a bench and snow piles turned into cars. Trace: SANPO labels them id 20 "obstacle" (snow piles, benches, bins, barriers in one class) → void/white (ADE20K has no such class) → ignored by the seg loss. Mapping verified correct. Void = 6.4 % of SANPO pixels (obstacle 5.5 %, unlabeled 0.9 %); obstacle in 2,693 of 3,172 training images (> 2 % of the frame in 1,613).
Gallery of all already-generated scenes with white areas, outlined in red, all models: `FINAL_v2_segloss/white_regions_gallery.jpg` (5 scenes). Result: v1, B and λ 0.006 draw SANPO clutter there (benches/chairs, snow piles, barriers, scaffolding), learned through the diffusion loss; only λ 0.05 → 0.4 turns white into cars (4/4 generations in the bench scene), because its seg loss dominated the diffusion loss and the prompt ("car") took over. Not a mapping error; supports a middle λ. Larger LoRA not pursued (rank 32 gave no gain in step 11; the ambiguity is in the label, not in model capacity).

**Dataset options for fewer voids (2026-10-25).** Searched for an alternative to SANPO with fewer voids and labelled cyclists: no dataset combines SANPO's look with both. Candidates for the report's future work: Mapillary Vistas (bicyclist, bench, trash can, snow classes; but the judge was trained on it), SENSATION-DS (FAU, chest-height, 9 classes with a catch-all "background"; partly from Mapillary; on request), Cityscapes (car view). In the cyclist frame the riders are SANPO "unlabeled" (annotation gap), not a mapping issue; SANPO's rider class is only 0.02 % of pixels.
User decision: keep λ 0.02; afterwards try SANPO with fewer white areas.
**Low-void re-selection analysis** (`data_prep/sanpo/analyze_void_reselection.py`, CPU; masks of all 11,964 human-annotated chest frames cached in `datasets/sanpo_raw/masks_all/`, per-frame numbers in `data_prep/sanpo/void_per_frame.csv`): same sessions and frames per session, lowest-void frames at least 12 (train) / 18 (test) frame numbers apart.
- train 3,172 frames: void mean 6.4 % → 3.7 %, median 2.6 % → 1.2 %, frames > 20 % void 9 % → 3 %; frames with riders 87 → 93. Weather mix unchanged (per-session counts kept). test: 7.9 % → 4.2 %.
- Plan: build "SANPO low-void" training set, keep the current test set (fair comparison with B and the λ runs), train with the best λ after the λ 0.02 result (needs user OK).

---

## Step 18 — second, independent judge (prepared 2026-10-25, CPU only)

**Why:** the seg loss and the mIoU score both use the Mapillary judge. A second general segmentor that was used neither in training nor trained on our data checks that the mIoU gain is not specific to the Mapillary judge. The user's plan: once the final (λ 0.02) images are ready, first inspect them, then compare them with the real SANPO photos under the new judge.
**Code** (backups `_code_backups/*.before_cityscapes_judge`): `src/segmentor.py` gains SegFormer judges (`DDPM_SEG_JUDGE=ade20k` / `cityscapes`), mapped to the same 12 shared classes + "other", so the existing targets are reused (`src/data/dataset.py`). `DDPM_EVAL_SAVE_DIR` makes the evaluation save every generated image under the real photo's file name.
**Candidates, tested on 30 real SANPO evaluation photos** (`step18_second_judge/test_second_judge_cpu.py`):

| Judge | Input | mIoU real | Pixel acc | road | sidewalk | Verdict |
|---|---|---|---|---|---|---|
| SegFormer-B5 Cityscapes | 1024 px | 0.478 | 0.414 | 0.08 | 0.13 | rejected: car-view model, calls 83 % of SANPO sidewalk "road" |
| SegFormer-B5 ADE20K | 512 px | 0.529 | 0.746 | 0.40 | 0.73 | |
| **SegFormer-B5 ADE20K** | **640 px** | **0.538** | **0.755** | 0.42 | 0.74 | **chosen** (640 = its training size); sidewalk pixels 76 % correct |
| (Mapillary, for reference, 150 photos) | 1024 px | 0.627 | 0.834 | 0.51 | 0.83 | training judge |

Note for the report: ADE20K is also the class space the seg ControlNet's conditioning colours come from; the judge is still independent of our data and of the training judge.
**Job (drafted, not submitted):** `step18_second_judge/submit_second_judge.sh <λ 0.02 run id>`: pretrained, B, λ 0.006, λ 0.02, λ 0.05→0.4; seed 42; 150-image subset; RTX3080; generated images saved to `step18_second_judge/images/<model>/`.

**Step 18 jobs submitted 2026-10-25 (user OK).** RTX3080, seed 42, ADE20K judge at 640 px, generated images saved to `step18_second_judge/images/<model>/`:
- pretrained 1850330, B 1850331, λ 0.006 1850332, λ 0.02 (sanpo_1850042) 1850334, λ 0.05→0.4 1850335.
- **18b, split by white** (job 1850336, after all five; `split_by_white.py`, user's idea): the 150 test images split at the median white share of their mask into two halves, copied for viewing into `step18_second_judge/images_by_white/{low_white,high_white}/{real,mask,<model>}/` plus `*_sheet.jpg`. Per half and model: KID, LPIPS, CLIP, mIoU (Mapillary 512 px and ADE20K 640 px), real-photo ceiling. Question: is λ 0.05→0.4 as realistic as B where the mask has little white? If yes, strong λ + low-void SANPO is the candidate final run.

**Step 18 results — second judge (2026-10-25).** SegFormer-B5 ADE20K at 640 px, seed 42, 150-image SANPO subset (all 5 jobs ≈ 19 min each, RTX3080). Real photos under this judge: mIoU 0.523.

| Model | KID ↓ | FID ↓ | LPIPS ↓ | CLIP image ↑ | mIoU ADE20K ↑ | Pixel acc ↑ |
|---|---|---|---|---|---|---|
| pretrained | 0.0307 | 153.9 | 0.710 | 0.756 | 0.422 | 0.475 |
| B (no seg loss) | **0.0094** | 134.8 | **0.659** | **0.811** | 0.421 | 0.604 |
| λ 0.006 | 0.0096 | **132.8** | 0.660 | **0.811** | 0.463 | 0.670 |
| **λ 0.02** (sanpo_1850042) | 0.0135 | 134.0 | 0.661 | 0.803 | **0.502** | 0.702 |
| λ 0.05 → 0.4 | 0.0418 | 149.5 | 0.666 | 0.757 | 0.681 | 0.888 |
| real photos | | | | | 0.523 | |

- The seg-loss gain holds under the independent judge and grows with λ (0.421 → 0.463 → 0.502 → 0.681): it is not specific to the Mapillary judge used in training.
- Under this judge B is no better than pretrained (0.421 vs 0.422; better sidewalk, worse fence/sky); the mask-adherence gain comes from the seg loss.
- λ 0.02 reaches 96 % of the real-photo score (0.502 / 0.523) with realism close to B (KID 0.0135 vs 0.0094, FID 134.0 vs 134.8, CLIP image 0.803 vs 0.811).
- λ 0.05 → 0.4 again scores far above the real photos (0.681 vs 0.523) with KID 4× B.
- Generated images of every model: `step18_second_judge/images/<model>/` (file name = real photo).

**Step 18b results — low-white vs high-white halves** (job 1850336, 4 min; median split of the mask's white share: low half mean 0.8 %, high half mean 13.7 %; 75 images each, seed 42; images in `step18_second_judge/images_by_white/`, numbers in `split_by_white.json`):

| Model | KID low | KID high | CLIP img low / high | mIoU Mapillary low / high | mIoU ADE20K low / high |
|---|---|---|---|---|---|
| pretrained | 0.0288 | 0.0377 | 0.739 / 0.773 | 0.465 / 0.414 | 0.455 / 0.362 |
| B | 0.0095 | 0.0095 | 0.798 / 0.824 | 0.513 / 0.495 | 0.449 / 0.366 |
| λ 0.006 | **0.0084** | 0.0125 | 0.798 / 0.824 | 0.561 / 0.528 | 0.487 / 0.417 |
| **λ 0.02** | 0.0095 | 0.0212 | 0.789 / 0.815 | 0.598 / 0.539 | **0.538** / 0.437 |
| λ 0.05 → 0.4 | 0.0407 | 0.0526 | 0.742 / 0.770 | 0.722 / 0.689 | 0.682 / 0.653 |
| real photos | | | | 0.638 / 0.591 | 0.539 / 0.480 |

- **λ 0.02 on low-white masks: KID equal to B (0.0095) and ADE20K mIoU equal to the real photos (0.538 vs 0.539).** Its realism cost comes entirely from masks with much white (KID 0.0212 there vs 0.0095 for B). The white (void) areas are the weak spot of the seg-loss model.
- λ 0.05 → 0.4 is unrealistic on both halves (KID 0.041 / 0.053): its style shift is general, not caused by the white areas.
- Consequence: a low-void SANPO training set with λ 0.02 is the most promising remaining improvement (not run).

**Step 17 results — λ 0.02, three seeds** (train 1850042 4 h 31 min, KID-selected checkpoint = epoch 8; evaluation 1850043; sheet 1850044; all completed 2026-10-25). Mapillary judge 1024 px, 150-image SANPO subset.

| Model | KID ↓ | FID ↓ | LPIPS ↓ | CLIP prompt ↑ | CLIP image ↑ | mIoU ↑ | Pixel acc ↑ |
|---|---|---|---|---|---|---|---|
| pretrained | 0.031 | 154 | 0.710 | 0.296 | 0.753 | 0.458 | 0.532 |
| B (no seg loss) | **0.0098 ± 0.0006** | 135.7 | **0.662** | **0.308** | **0.810** | 0.507 ± 0.007 | 0.705 |
| λ 0.006 | 0.0107 ± 0.0008 | **133.5** | 0.663 | 0.307 | 0.808 | 0.544 ± 0.008 | 0.748 |
| **λ 0.02** (sanpo_1850042) | 0.0158 ± 0.0024 | 135.5 ± 1.6 | 0.664 | 0.307 | 0.799 | **0.565 ± 0.010** | **0.784** |
| λ 0.05 → 0.4 | 0.0451 | 152.3 | 0.667 | 0.286 | 0.755 | 0.705 | 0.912 |
| real photos | | | | | | 0.627 | 0.834 |

Per class λ 0.02: road 0.33, sidewalk 0.76, building 0.66, fence/wall 0.60, person 0.81, vehicle 0.64, pole 0.40, sign/light 0.09, vegetation 0.74, terrain 0.68, sky 0.51.
- λ 0.02 = 90 % of the real-photo score under the training judge (0.565 / 0.627) and 96 % under the independent ADE20K judge (0.502 / 0.523), with FID equal to B and KID +0.006. On low-white masks its KID equals B (step 18b).
- White areas (gallery `FINAL_v2_segloss/white_regions_gallery.jpg`, now with λ 0.02, script `reporting/make_white_gallery.py`): λ 0.02 draws benches/bins and snow in the bench scene and a barrier at the kerb, like B; only λ 0.05 → 0.4 draws cars.
- Chosen final SANPO model: **λ 0.02 (sanpo_1850042)**. RESULTS.docx (new section: second judge + low/high-white split + gallery) and RUN_INDEX rebuilt.

---

## Step 19 — judge class mapping check (2026-10-27 to 2026-10-31)

**Diagnosis** (`step19_judge_mapping/diagnose_judge_classes.py`, CPU, 50 real SANPO photos, `raw_classes_*.json`): which raw judge classes are predicted per target class. Definition mismatches that fell into "other": ADE20K floor (8.4 % of sidewalk), pier (1.6 % of sidewalk), column (7.5 % of pole); Mapillary Snow (21 % of terrain). Left unchanged as real errors or ambiguous: pole/sign predicted as building or tree, sky/vegetation at branches, Mapillary Tunnel on scaffolding. White pixels are excluded from every mIoU, so they do not cause these losses.
**Mapping v2** (`DDPM_JUDGE_MAPPING=v2`, `src/segmentor.py`; v1 stays the default, backup `_code_backups/segmentor.py.before_mapping_v2`): Mapillary Snow → terrain; ADE20K floor, pier → sidewalk, column → pole.
**Re-score:** job 1858017 (`rescore_mappings.py`, RTX3080), saved seed-42 images, v1 and v2 side by side, Mapillary 512 px, ADE20K 640 px, all / low-white / high-white.

**Step 19 results** (job 1858017, 6.5 min, RTX3080; `step19_judge_mapping/rescore_mappings.json`). mIoU, seed 42, 150 images, v1 → v2:

| Model | Mapillary 512 px | ADE20K 640 px |
|---|---|---|
| pretrained | 0.449 → 0.408 | 0.422 → 0.424 |
| B | 0.507 → 0.486 | 0.421 → 0.427 |
| λ 0.006 | 0.551 → 0.521 | 0.463 → 0.467 |
| λ 0.02 | 0.578 → 0.561 | 0.502 → 0.503 |
| λ 0.05 → 0.4 | 0.718 → 0.717 | 0.681 → 0.685 |
| real photos | 0.625 → 0.622 | 0.523 → 0.531 |

- The expected gain did not appear: the real-photo mIoU barely moves. Definition mismatches are not what keeps the real-photo scores low; the hard classes are. Mapillary Snow → terrain even lowers the models' scores, because snow on non-terrain pixels now counts against terrain.
- Decision: keep mapping v1 as the reported mapping and do not use v2 in training. The model ranking is unchanged under v2.

---

## Step 20 — downstream low-data test on SANPO (prepared 2026-10-31, nothing run on GPU yet)

**Question (user's design):** with few real photos, does adding generated images improve a segmentation model? Three settings, fine-tuned segmentation model, tested on all 522 real SANPO test photos, 3 seeds each: A limited real; B limited real + base ControlNet images; C limited real + seg-loss ControlNet images.
**Scripts** (`step20_downstream_sanpo/`): `make_splits.py` (`splits.json`: 300 real frames from 115 sessions, 1,000 other masks from 116 sessions, no overlap), `generate_from_masks.py`, `train_downstream.py` (SegFormer-B5 from ADE20K weights with a new 12-class head; shared classes, void ignored; 50 % synthetic sampling; 3,000 steps, batch 4, LR 6e-5). CPU smoke test passed. Step 8's scripts were not reused because `--real-limit` takes the first N frames (few sessions) and only class 0 is ignored on SANPO.
**Limitation:** B vs C cannot separate the effect of fine-tuning from that of the seg loss.

**White handling verified (2026-10-31).** Void is excluded per pixel, never per image. 87 test images: white 6.0 %, not scored 6.6 % (white + a few classes without a shared class), white pixels scored 0, images fully excluded 0. Bench frame: 9.2 % white, 90.4 % scored; cyclist frame: 34.1 % white, 65.9 % scored. `confusion_matrix`, cross-entropy and the Dice term in `seg_consistency_loss` (`src/losses.py`) all mask these pixels. Consequence: the seg loss cannot correct what is drawn inside white areas.

## Step 21 — last λ check, SANPO: λ 0.025 → 0.045 (submitted 2026-10-31)

B recipe + seg loss, λ rising linearly 0.025 → 0.045 over all 8 epochs (6,344 steps), checkpoint = last epoch, LoRA of epochs 2/4/6/8 kept, control scale 1.0. Replaces λ 0.02 as final only if Mapillary mIoU gets closer to the real photos (0.627), KID stays clearly below 0.045 and the images look good. Script: `FINAL_v2_segloss/submit_segloss_l0025_0045_sanpo.sh`, outputs `FINAL_v2_segloss/sanpo_l0.025-0.045/`. Jobs: training 1858162 (run sanpo_1858162), three-seed evaluation 1858163, sample sheet 1858164, ADE20K judge 1858165 (images in `step18_second_judge/images/seg0.025-0.045/`). About 6 GPU-hours. The winner's λ is then used for the step 20 downstream test. Results to be added.

### Step 21 results (2026-11-01)

All four jobs completed. Three seeds, Mapillary judge 1024 px, 150-image subset, last epoch (λ 0.045); ADE20K judge seed 42, 640 px.

| Model | KID | FID | CLIP img | mIoU Mapillary | Pixel acc | mIoU ADE20K |
|---|---|---|---|---|---|---|
| pretrained | 0.031 | 154 | 0.753 | 0.458 | 0.532 | 0.422 |
| B | 0.0098 | 135.7 | 0.810 | 0.507 | 0.705 | 0.421 |
| λ 0.02 (1850042) | 0.0158 ± 0.0024 | 135.5 | 0.799 | 0.565 ± 0.010 | 0.784 | 0.502 |
| λ 0.025 → 0.045 (1858162) | 0.0239 ± 0.0013 | 137.1 ± 3.0 | 0.790 | 0.604 ± 0.007 | 0.835 | 0.551 |
| λ 0.05 → 0.4 (1848904) | 0.0451 | 152.3 | 0.755 | 0.705 | 0.912 | 0.681 |
| real photos | – | – | – | 0.627 | 0.834 | 0.523 |

- Mapillary: 0.023 below real, pixel accuracy equal to real. ADE20K: 0.028 above real (λ 0.02 was 0.021 below).
- Realism: KID +0.008 over λ 0.02 and 2.4× B, still better than pretrained; FID within seed spread of λ 0.02.
- Training-time epoch 6 reached mIoU 0.613 on seed 42; the eval checkpoint is epoch 8.

## Step 20 — final experiment: does generated data help when real photos are limited? (submitted 2026-11-01)

Selected λ after step 21: 0.025 → 0.045 (sanpo_1858162). Setups: A 300 real photos; B 300 real + 1000 base ControlNet images; C 300 real + 1000 images of sanpo_1858162. SegFormer-B5 from ADE20K weights, 12 classes, 3000 steps, batch 4, LR 6e-5, seeds 0/1/2, 50 % synthetic sampling, white/void ignored per pixel, tested on all 522 real test photos.

User's decision: use the existing generator (Option 2). It was trained on all 3,172 photos, including the 300 and the photos behind the 1000 masks, so C is a result "with a well-trained generator", not a pure low-data result; state this in the report.

Scripts: `step20_downstream_sanpo/submit_downstream.sh`, `step20_job.sh`. Jobs: generate base 1859200, generate ours 1859201, train A 1859202, B 1859203, C 1859204. Results to be added.

### Step 20 results (2026-11-02)

All five jobs completed (1859200-04). SegFormer-B5, 3000 steps, seeds 0/1/2, tested on all 522 real SANPO photos.

| Setup | mIoU per seed | Mean ± SD (n-1) | Pixel acc |
|---|---|---|---|
| A: 300 real | 0.6226, 0.6245, 0.6155 | 0.621 ± 0.005 | 0.883 |
| B: 300 real + 1000 base | 0.6145, 0.6064, 0.6095 | 0.610 ± 0.004 | 0.877 |
| C: 300 real + 1000 ours | 0.6304, 0.6240, 0.6260 | 0.627 ± 0.003 | 0.885 |

Same-seed differences: C-B +0.016/+0.018/+0.017; C-A +0.008/-0.001/+0.011; B-A -0.008/-0.018/-0.006.

Per class, mean IoU A / B / C: road 0.667/0.598/0.653, sidewalk 0.906/0.893/0.906, building 0.694/0.703/0.696, fence/wall 0.544/0.566/0.575, person 0.786/0.763/0.783, vehicle 0.659/0.661/0.710, pole 0.360/0.326/0.347, sign/light 0/0/0, vegetation 0.815/0.811/0.818, terrain 0.635/0.632/0.645, sky 0.764/0.759/0.761. C beats A on all 3 seeds for fence/wall, vehicle, vegetation and terrain; A beats C on all 3 seeds for road and person. Sign/light is 0 in all setups.

- C beats B on every seed: seg-loss images are more useful than base images, and base images lower the score below real-only.
- C vs A is not shown: +0.006 mean, one seed lower, with a generator trained on all 3,172 photos.
- 300 photos are already strong (0.621), so this limit leaves little room to gain.

## Step 22 — 100 real photos, fakes from the same masks (submitted 2026-11-02)

User's design: A 100 real; B 100 real + 100 base ControlNet images; C 100 real + 100 images of sanpo_1858162, all fakes made from the SAME 100 masks. `limited_100` in `step20_downstream_sanpo/splits.json` = 100 evenly spaced frames of the 300. Otherwise as step 20. Script: `step20_downstream_sanpo/submit_downstream_100.sh`. The generator saw all 3,172 photos, including these 100. Results to be added.

Step 22 cancelled by the user on 2026-11-02 (jobs 1859575-79); no results.

## FULL FINAL FLOW (step 23, submitted 2026-11-02)

**Question:** with only 200 real photos and their masks, can generated images give a better segmentation model? Real-life rule set by the user: nothing outside those 200 photos and masks is used, except pretrained models anyone can download. No other masks and no generator trained on more photos.

| Step | What happens | Settings | Output | Job |
|---|---|---|---|---|
| 1. Pick the data | 200 real SANPO photos with their masks, evenly spaced over the 300 of step 20 (114 sessions) | `make_data_200.py`, `limited_200` in `splits.json`, `DDPM/data_sanpo_200/` (symlinks; `data_sanpo/` unchanged) | `data_sanpo_200/` | – |
| 2. Fine-tune ControlNet | New ControlNet on only the 200 photos | B recipe, LoRA rank 16, LR 1e-4, seg loss λ 0.025 → 0.045 over 3000 steps (15 epochs of 200 steps; the 200 photos are listed 4 times per epoch), Mapillary judge, last epoch, control scale 1.0 | `step20_downstream_sanpo/generator_200/` | 1859598 (run sanpo_1859598) |
| 3a. Generate, base | 4 fakes per mask with the base ControlNet, 800 images | DDIM 35, guidance 6.0, control 1.0, seed 1234, `prompts_meta.csv` | `synthetic/base_200x4/` | 1859599 |
| 3b. Generate, ours | 4 fakes per mask with the new ControlNet, same masks, prompts and noise, 800 images | as 3a | `synthetic/ours_200x4/` | 1859600 |
| 4. Train segmentation | A: 200 real. B: 200 real + 800 base. C: 200 real + 800 ours | SegFormer-B5 from ADE20K weights, 12 classes, 3000 steps, batch 4, LR 6e-5, seeds 0/1/2, 50 % fake samples in B and C, white/void ignored per pixel | `results/real_only_200.json`, `real_plus_base_200.json`, `real_plus_ours_200.json` | 1859601, 1859602, 1859603 |
| 5. Test | All 522 real SANPO test photos, never used in any training step | mIoU, pixel accuracy, per class, per seed | same files | – |

**How to read it:** a clear gain means C beats A on all 3 seeds, by more than the spread between seeds. C vs B shows whether the seg loss makes fakes more useful. A vs step 20's 0.621 (300 photos) shows how much the smaller photo count costs.

**Difference from step 20:** the generator there saw all 3,172 photos and the fakes used 1,000 other masks. Here both are limited to the 200, so this is the real-life result.

**Scripts:** `step20_downstream_sanpo/make_data_200.py`, `submit_final_flow.sh`, `generate_from_masks.py`, `train_downstream.py`, `step20_job.sh`. Results to be added.

### Full final flow, status 2026-11-03

- Generator on 200 photos (1859598) completed, 3 h 10 min V100. Seed 42, 150-image subset, Mapillary judge, last epoch: KID 0.030, FID 140.2, CLIP img 0.792, mIoU 0.594 (real photos 0.625). The same λ on all 3,172 photos (1858162, seed 42) reached KID 0.025, mIoU 0.606.
- A, 200 real (1859601): 0.6270 / 0.6218 / 0.6170, mean 0.622 ± 0.005 (300 real in step 20: 0.621).
- B, 200 real + 800 base (1859602): 0.6054 / 0.5944 / 0.5903, mean 0.597 ± 0.008.
- Generation with the new generator (1859600) crashed with a segmentation fault on node tg082 after 200 of 800 images; the base generation, same code, completed. C (1859603) could not start and was cancelled. Rerun needs the user's OK.
- Rerun submitted 2026-11-03 (`resubmit_final_ours.sh`, node tg082 excluded): generation 1862019, C 1862020.

### Full final flow, results (2026-11-05)

Rerun completed: generation 1862019 (800 images), C 1862020. Tested on all 522 real SANPO photos.

| Setup | mIoU per seed | Mean ± SD (n-1) | Pixel acc |
|---|---|---|---|
| A: 200 real | 0.6270, 0.6218, 0.6170 | 0.622 ± 0.005 | 0.883 |
| B: 200 real + 800 base | 0.6054, 0.5944, 0.5903 | 0.597 ± 0.008 | 0.870 |
| C: 200 real + 800 ours (generator on the 200 only) | 0.6145, 0.6012, 0.5969 | 0.604 ± 0.009 | 0.872 |

Same-seed differences: C-A -0.013/-0.021/-0.020; C-B +0.009/+0.007/+0.007.

Per class, mean IoU A / B / C: road 0.681/0.526/0.540, sidewalk 0.908/0.888/0.887, building 0.685/0.694/0.692, fence/wall 0.529/0.530/0.558, person 0.805/0.763/0.773, vehicle 0.680/0.655/0.655, pole 0.352/0.311/0.324, sign/light 0/0/0, vegetation 0.817/0.811/0.814, terrain 0.616/0.621/0.642, sky 0.767/0.761/0.761 (sky B 0.765).

- With only the 200 photos and their masks, fakes lower the score: C is below A on every seed. Most of the loss is road (-0.14), in both B and C.
- C beats B on every seed: seg-loss fakes are less harmful than base fakes. C beats A on all seeds only for fence/wall, terrain and building.
- 200 real photos score the same as 300 (0.622 vs 0.621), so the point where real photos run short was not reached.

## Step 25 — final flow with an ImageNet-start segmentation model (submitted 2026-11-08)

Same 200 photos and 800 fakes per generator as the full final flow, but SegFormer-B5 starts from ImageNet weights (`nvidia/mit-b5`, `train_downstream.py --init imagenet`), so it has no segmentation or street-scene pretraining. Question: do fakes help a model that cannot lean on ADE20K? Script: `step20_downstream_sanpo/submit_imagenet_200.sh`. Jobs: 1868314 in_imagenet_real_only_200 1868315 in_imagenet_real_plus_base_200 1868316 in_imagenet_real_plus_ours_200 

## Clean-up for the report (2026-11-08, user's request: no invalid old results, no buggy results)

- Moved, not deleted, to `DDPM/_archive_not_in_report/` (README there lists why): pre-audit runs (`outputs/`, `test/outputs/`), step 2 and 3 (old UNet++ judge and its baselines), step 4 smoke, step 5 LR sweep (original random-weather prompts, old judge; 1836710 RNG bug), step 6 Phase 2 and step 7 (RNG bug), step 8 dataset-1 downstream (generators with the RNG bug, saw all photos), `FINAL/*/eval` (v1 scored with the old judge; Mapillary rescores kept), `FINAL/canary`, old comparison sheets. The previous RESULTS.docx (2026-10-26) is kept there too.
- RESULTS.docx rebuilt (`reporting/build_results_doc.py`) from valid runs only: protocol, hyperparameter screen (KID only), v1, v2/B, four seg-loss λ runs, the 200-photo generator, three-seed Mapillary table, ADE20K judge and white split, mapping check, downstream tests (300 photos, final flow, ImageNet start). Training-time mIoU of runs scored by the old judge is hidden. '±' is now the sample standard deviation (n−1) everywhere.
- RUN_INDEX.md skips the archive and lists the step 20+ downstream results. `jobs/baseline_eval.sh` writes scratch output to `controlnet_testing/scratch_outputs/` (was the archived `step3_baseline/`).
- This log keeps the full history, including archived results.

### Step 25 results (2026-11-08)

All three jobs completed (1868314-16). ImageNet-start SegFormer-B5, 200 real photos, tested on all 522 real test photos.

| Setup | mIoU per seed | Mean ± SD (n-1) | Pixel acc |
|---|---|---|---|
| A: 200 real | 0.6201, 0.6209, 0.6273 | 0.623 ± 0.004 | 0.877 |
| B: + 800 base | 0.6116, 0.6165, 0.6096 | 0.613 ± 0.004 | 0.871 |
| C: + 800 ours | 0.6234, 0.6259, 0.6245 | 0.625 ± 0.001 | 0.876 |

Same-seed differences: C-A +0.003/+0.005/-0.003; C-B +0.012/+0.009/+0.015. Per class A / B / C: road 0.631/0.559/0.568, fence/wall 0.523/0.544/0.569, vehicle 0.672/0.664/0.693, terrain 0.599/0.603/0.630, sign/light 0.136/0.117/0.120.

- Without street-scene pretraining, 200 real photos still reach 0.623, the same as the ADE20K start (0.622).
- Ours beats base on every seed again; ours vs real only is equal (+0.002, one seed lower). Road drops with any fakes; fence/wall, vehicle and terrain improve with ours.
