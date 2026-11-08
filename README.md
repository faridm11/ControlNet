# ControlNet with a segmentation loss: generating sidewalk images that a segmentation model can learn from

This project fine-tunes a **ControlNet** (Stable Diffusion 1.5) to turn a **semantic segmentation mask + text prompt** into a
realistic sidewalk photo. During fine-tuning, a frozen, pretrained segmentation model (the **judge**) checks each generated
image against its mask, and an extra **segmentation loss** pushes the generator to draw what the mask says. The goal is
synthetic image/mask pairs good enough to enlarge the training data of a sidewalk segmentation model.

```
mask ──► ControlNet (+ LoRA) ──► Stable Diffusion 1.5 ──► generated photo
 │                                                          │
 │                                     frozen judge (Mask2Former, Mapillary Vistas)
 │                                                          │
 └──────────── segmentation loss = CE + Dice(judge(photo), mask) ◄──┘
total loss = diffusion loss + λ · segmentation loss
```

---

## Main results (SANPO, details in `controlnet_testing/RESULTS.docx`)

| Generator | KID ↓ | mIoU, Mapillary judge ↑ | mIoU, ADE20K judge ↑ |
|---|---|---|---|
| Pretrained ControlNet (no fine-tuning) | 0.031 | 0.458 | 0.422 |
| Fine-tuned, no segmentation loss ("B") | 0.010 | 0.507 | 0.421 |
| **Fine-tuned + segmentation loss, λ 0.025 → 0.045** | 0.024 | **0.603** | **0.551** |
| Real photos (judge ceiling) | – | 0.627 | 0.523 |

The generator scores use three seeds and the 150-image test subset; the ADE20K judge uses one seed. KID: lower = more
realistic. mIoU: how well the judge's reading of the image matches the input mask. The ADE20K judge was never used in training.

- The segmentation loss brings mask adherence close to real photos, at a moderate cost in realism. λ sets the trade-off.
- Segmentation models trained on 200 or 300 real photos plus our generated images beat the same models trained with
  pretrained-ControlNet images on every seed (0.627 vs 0.610 with 300 photos, 0.604 vs 0.597 with 200).
- They did not beat training on the real photos alone (300 photos: 0.627 vs 0.621, within noise; 200 photos: 0.604 vs 0.622).
  200–300 real photos were already enough for an ADE20K-pretrained SegFormer.

---

## Repository layout

```
DDPM/
├── src/                          project package
│   ├── config.py                 every setting; each can be overridden with an environment variable DDPM_<NAME>
│   ├── train.py                  training loop (Trainer): diffusion loss + optional segmentation loss, evaluation, checkpoints
│   ├── losses.py                 diffusion MSE loss; segmentation-consistency loss (CE + soft Dice, white/void pixels ignored)
│   ├── segmentor.py              judges: Mask2Former Swin-L Mapillary (training + scoring), SegFormer-B5 ADE20K (independent),
│   │                             class mapping of every judge to 12 shared classes
│   ├── evaluation.py             KID, FID, LPIPS, CLIP image/prompt, judge mIoU in one generation pass
│   ├── models/diffusion_controlnet.py   SD 1.5 + ControlNet + LoRA (PEFT), checkpoint save/load/merge
│   ├── data/dataset.py           image + mask + prompt dataset, joint flip, prompt augmentation, judge targets
│   ├── plotting.py, sampling.py  training curves, sample strips
│   └── utils.py                  paths for local / HPC runs
├── evaluate.py                   evaluate a LoRA checkpoint (or the pretrained ControlNet) on the test set
├── inference.py                  generate images from one mask + prompt
├── data_prep/                    class colour mappings and prompt generation
│   └── sanpo/                    download and conversion of SANPO (own README.md)
├── data/                         dataset 1: segments/sidewalk-semantic, 950 train / 50 test (images not in this folder's git)
├── data_sanpo/                   dataset 2: SANPO-Real chest camera, 3,172 train / 522 test
├── controlnet_testing/           all experiments: job scripts, run outputs, logs, results document (see below)
└── _archive_not_in_report/       superseded and bug-affected runs, kept for history (README.md inside)
```

### Where the experiments are (`controlnet_testing/`)

| Path | Contents |
|---|---|
| `RESULTS.docx` | all results used in the report: tables, curves, sample images (rebuilt by `reporting/build_results_doc.py`) |
| `EXPERIMENT_LOG.md` | every step in order: what, why, settings, job ids, results. The end has the **FULL FINAL FLOW** section |
| `RUN_INDEX.md` | one row per training run: folder, settings, best numbers, kept files (`reporting/index_runs.py`) |
| `jobs/` | generic SLURM scripts: `train.sh`, `baseline_eval.sh` (three-seed evaluation), `comparison_grid.sh` (sample sheets) |
| `FINAL/`, `FINAL_v2/` | v1 and v2 ("B") models of both datasets |
| `FINAL_v2_segloss/` | SANPO models with the segmentation loss: `sanpo` (λ 0.006), `sanpo_l0.02`, `sanpo_l0.025-0.045` (selected), `sanpo_ramp` (λ 0.05 → 0.4) |
| `step18_second_judge/` | independent ADE20K judge, every generated test image, low-white / high-white split |
| `step19_judge_mapping/` | check of the judge class mapping |
| `step20_downstream_sanpo/` | downstream test: segmentation models trained with real and generated images |
| `step0_fixes/`, `step1_cpu_checks/`, `step9_sanpo_audit/`, `step12_review_fixes/`, `step15_phase2_sanpo/`, `step16_lambda_ramp/` | CPU tests of the code fixes and new features |

Every output carries its run id: `metrics_<run>.csv`, `<run>_overview.png`, `<run>_LoRA.pt`, `<run>_Full/`.
SANPO runs start with `sanpo_` (for example `sanpo_1858162`).

---

## Setup

```bash
module load python/3.12-conda cuda/12.4.1        # FAU NHR tinygpu
conda activate control2                          # or: conda env create -f environment.yml
export HF_HOME=$WORK/huggingface_cache            # models are cached here
export HF_HUB_OFFLINE=1                           # compute nodes have no internet; download on the login node first
export https_proxy=http://proxy.nhr.fau.de:80     # login node only, for downloads
```

Versions in use: Python 3.12, torch 2.6 (CUDA 12.4), diffusers 0.35, transformers 5.5, peft 0.18, torchmetrics 1.8.

Models downloaded from Hugging Face: `runwayml/stable-diffusion-v1-5`, `lllyasviel/control_v11p_sd15_seg`,
`facebook/mask2former-swin-large-mapillary-vistas-semantic` (judge 1), `nvidia/segformer-b5-finetuned-ade-640-640` (judge 2),
`openai/clip-vit-base-patch16` (CLIP scores), `nvidia/mit-b5` (ImageNet start for one downstream test).

---

## Data

| | Dataset 1 | Dataset 2 (main) |
|---|---|---|
| Source | segments/sidewalk-semantic | SANPO-Real, chest camera, human-annotated frames (CC BY 4.0) |
| Train / test | 950 / 50 | 3,172 / 522 (official session split; evaluation uses a fixed 150-image subset) |
| Folder | `data/` | `data_sanpo/` (contains `LABEL_SPACE` = `sanpo`) |
| Labels | project class ids | SANPO's own 31 class ids |
| Prompts | `prompts_objects.csv` | `prompts_meta.csv`: recorded conditions (weather, environment) + objects in the crop |

Each folder has `{train,test}/{images,labels,prompts}`. **The data root selects the dataset**: set
`DDPM_DATA_ROOT=$PWD/data_sanpo` and every script uses the SANPO classes, colours and prompts.

**Conditioning:** each class is drawn in its ADE20K colour, which the pretrained ControlNet already understands. Classes
with no ADE20K equivalent (SANPO "unlabeled", "obstacle", "inaccessible surface", "bus stop") are white.
**White pixels are ignored pixel by pixel** in the segmentation loss, every mIoU and the downstream training. No image
is dropped because of them.

SANPO preparation (download, frame selection, label conversion, checks): `data_prep/sanpo/README.md`.
Prompts: `python data_prep/generate_prompts.py --data-root data_sanpo --label-space sanpo --crop 512 --variant meta --out-name prompts_meta.csv`.

---

## Training

Only a **LoRA on the ControlNet** is trained (rank 16, attention + feed-forward layers + conditioning input). Stable Diffusion's
UNet, VAE and text encoder stay frozen. Settings live in `src/config.py`; any of them can be set per job as `DDPM_<NAME>`.

| Setting | Value used | Variable |
|---|---|---|
| LoRA rank / alpha / dropout | 16 / 16 / 0 | `DDPM_CONTROLNET_LORA_RANK`, `_ALPHA`, `_DROPOUT` |
| LoRA targets | attention + feed-forward + conditioning conv | `DDPM_LORA_TARGETS=attn_ff` |
| Learning rate, schedule | 1e-4, cosine with 100 warm-up steps | `DDPM_CONTROLNET_LORA_LR` |
| Batch | 4 (no accumulation), fp16 on V100 | `DDPM_BATCH_SIZE`, `DDPM_GRADIENT_ACCUMULATION_STEPS=1` |
| Epochs | SANPO 8 (6,344 steps), dataset 1 20 | `DDPM_NUM_EPOCHS` |
| Prompt augmentation | each true phrase kept with p = 0.7, order shuffled | `DDPM_TRAIN_PROMPT_KEEP_PROB=0.7` |
| Segmentation loss | on/off, λ start → target, ramp length | `DDPM_USE_SEGMENTOR_LOSS`, `DDPM_SEGMENTOR_LOSS_WEIGHT_START`, `DDPM_SEGMENTOR_LOSS_WEIGHT`, `DDPM_SEGMENTOR_LOSS_WARMUP_STEPS` |
| Seg-loss details | applied for timesteps < 400, at most 2 samples per batch, on the decoded x0 estimate | `DDPM_SEGMENTOR_LOSS_MAX_TIMESTEP`, `_MAX_SAMPLES` |
| Judge | `mapillary` (training), `ade20k` (independent scoring) | `DDPM_SEG_JUDGE` |
| Evaluation during training | every N epochs on a fixed subset; checkpoint by KID or last epoch | `DDPM_EVAL_EVERY_N_EPOCHS`, `DDPM_VAL_SUBSET_SIZE`, `DDPM_CHECKPOINT_METRIC` |

**Train the selected SANPO model** (λ rising 0.025 → 0.045 over all steps; full script: `controlnet_testing/FINAL_v2_segloss/submit_segloss_l0025_0045_sanpo.sh`):

```bash
cd DDPM
export RUN_NOTE="what this job tests"            # printed in the job log header with the date, dataset and all settings
sbatch --partition=v100 --gres=gpu:v100:1 --cpus-per-task=8 --time=12:00:00 \
  --export=ALL,OUT_DIR=$PWD/my_run,DDPM_RUN_NAME=segloss_sanpo,\
DDPM_EXPECT_DATASET=sanpo,DDPM_DATA_ROOT=$PWD/data_sanpo,DDPM_PROMPTS_FILE_NAME=prompts_meta.csv,\
DDPM_VAL_SUBSET_SIZE=150,DDPM_EVAL_KID_SUBSET=50,\
DDPM_LORA_TARGETS=attn_ff,DDPM_CONTROLNET_LORA_RANK=16,DDPM_CONTROLNET_LORA_ALPHA=16,DDPM_CONTROLNET_LORA_DROPOUT=0.0,\
DDPM_CONTROLNET_LORA_LR=1e-4,DDPM_BATCH_SIZE=4,DDPM_GRADIENT_ACCUMULATION_STEPS=1,DDPM_TRAIN_PROMPT_KEEP_PROB=0.7,\
DDPM_SEG_JUDGE=mapillary,DDPM_USE_SEGMENTOR_LOSS=true,DDPM_SEGMENTOR_LOSS_WEIGHT_START=0.025,\
DDPM_SEGMENTOR_LOSS_WEIGHT=0.045,DDPM_SEGMENTOR_LOSS_WARMUP_STEPS=6344,DDPM_SEG_INPUT_SCALE=1,\
DDPM_NUM_EPOCHS=8,DDPM_EVAL_EVERY_N_EPOCHS=2,DDPM_CHECKPOINT_METRIC=last \
  controlnet_testing/jobs/train.sh
```

Leave out the `DDPM_*SEGMENTOR*` settings (or set `DDPM_USE_SEGMENTOR_LOSS=false`) to train model B. Training takes about
4.5 h on one V100 32 GB, and the segmentation loss uses nearly all of its memory.
`DDPM_EXPECT_DATASET` stops the job if the data root is not the expected dataset.

**Outputs in `OUT_DIR`:** `logs/metrics_<run>.csv` (every epoch, including λ and the seg loss), `logs/<run>_final_eval.json`,
`plots/<run>_overview.png` (KID, mIoU, LPIPS, CLIP, val loss) and `<run>_diagnostics.png` (losses, gradient norm, LR, λ),
`samples/`, `checkpoints/<run>_LoRA.pt` (LoRA only), `checkpoints/<run>_Full/` (merged ControlNet), and with
`DDPM_SAVE_EVAL_CHECKPOINTS=true`, `<run>_ep<N>_LoRA.pt` of every evaluated epoch.

---

## Evaluation and generation

```bash
# Three-seed evaluation of a checkpoint (KID, FID, LPIPS, CLIP, judge mIoU, per-class IoU); omit CKPT for the pretrained ControlNet
sbatch --partition=v100 --gres=gpu:v100:1 --export=ALL,TAG=my_eval,SEEDS="42 43 44",OUT=$PWD/my_eval,\
CKPT=$PWD/my_run/checkpoints/<run>_LoRA.pt,DDPM_DATA_ROOT=$PWD/data_sanpo,DDPM_VAL_SUBSET_SIZE=150,DDPM_EVAL_KID_SUBSET=50,\
DDPM_SEG_JUDGE=mapillary,DDPM_LORA_TARGETS=attn_ff,DDPM_CONTROLNET_LORA_RANK=16,DDPM_CONTROLNET_LORA_ALPHA=16 \
  controlnet_testing/jobs/baseline_eval.sh
#   independent judge: DDPM_SEG_JUDGE=ade20k,DDPM_SEG_INPUT_SCALE=1.25 ; save every image: DDPM_EVAL_SAVE_DIR=<dir>

# One mask -> images (class-id masks are coloured automatically; RGB masks are used as they are)
python inference.py --controlnet my_run/checkpoints/<run>_Full --mask data_sanpo/test/labels/<frame>.png \
  --prompt "street-level photo of a sidewalk, urban street, sunny day, pedestrian, car" --output out.png
```

Generation settings: DDIM 35 steps, guidance 6.0, control scale 1.0. Scoring uses a fixed 150-image SANPO test subset,
KID subset 50, generation seeds 42/43/44. KID is the main realism number; FID is unreliable at this sample size.

**Judges.** Both judges are general pretrained models, never trained on our data. Their classes are mapped to 12 shared
classes (road, sidewalk, building, fence/wall, person, vehicle, bicycle, pole, sign/light, vegetation, terrain, sky) in
`src/segmentor.py`. Running the same judge on the real photos gives the realistic ceiling, which is far below 1.0.

---

## Downstream test: do the generated images help a segmentation model?

`controlnet_testing/step20_downstream_sanpo/`: SegFormer-B5 trained on a small set of real SANPO photos with or without
generated images, then tested on all 522 real test photos.

| Script | Role |
|---|---|
| `make_splits.py`, `make_data_200.py` | which real photos are "available" (300 / 200), and a data root with only those 200 |
| `generate_from_masks.py` | `--controlnet pretrained` or a `<run>_Full`, `--split`, `--per-mask N`: images + their scored targets |
| `train_downstream.py` | `--name`, `--real-split`, `--synthetic <dir>`, `--init {ade20k,imagenet}`; 3,000 steps, 3 seeds, 50 % synthetic batches |
| `submit_final_flow.sh` | the full final flow: generator on only 200 photos → 4 fakes per mask → setups A / B / C |
| `step20_job.sh` | job wrapper that prints the standard log header |

**Full final flow:** only 200 real photos and their masks are used anywhere, apart from downloadable pretrained models.
Setups: (A) 200 real; (B) + 800 pretrained-ControlNet images; (C) + 800 images from a seg-loss ControlNet trained on those
200 photos. The steps are written out at the end of `controlnet_testing/EXPERIMENT_LOG.md`.

---

## Reporting

```bash
PYTHONPATH=/home/woody/iwi5/iwi5388h/software/private/reportdeps \
  python controlnet_testing/reporting/build_results_doc.py   # RESULTS.docx from the result files
python controlnet_testing/reporting/index_runs.py            # RUN_INDEX.md
```

Every job log starts with a header: date, job, what it tests (`RUN_NOTE`), dataset and every `DDPM_*` setting.

---

## Known limitations

- **White areas.** About 6 % of SANPO pixels (mostly "obstacle": benches, bins, snow piles) have no ADE20K colour and are
  white. The loss skips them, so the generator draws something plausible there, but sometimes wrong (for example a car).
  Images with large white areas are clearly less realistic.
- **Weather prompts** come from SANPO's per-session labels, not per frame.
- **Dataset 1** has no valid segmentation-loss run: its seg-loss runs had a since-fixed random-state bug and were archived.
- **Downstream leakage.** In the 300-photo downstream test the generator had seen all 3,172 photos. The final flow removes this.
- **Seeds.** Three seeds per result. Differences smaller than the seed spread are not meaningful.
