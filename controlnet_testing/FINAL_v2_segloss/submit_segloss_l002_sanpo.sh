#!/bin/bash
# SANPO: B recipe ("v2") + segmentation loss with a MIDDLE weight, λ 0.02 (step 17, 2026-10-25) — intended final run.
# Goal (user): mIoU close to the real photos (0.627) while KID/FID stay close to B (no strong realism loss).
# Why 0.02: λ 0.006 (job 1848504) kept realism (KID 0.011) but mIoU only 0.544; λ 0.05 -> 0.4 (job 1848904) gave
# mIoU 0.705 but KID 0.045, and already at λ 0.05-0.14 (epochs 1-2) the model stopped learning the SANPO look
# (CLIP image stayed 0.76 vs 0.805 for B). 0.02 ≈ geometric middle of 0.006 and 0.05 (≈ 7x the diffusion push).
# Identical to the λ 0.006 run otherwise: from PRETRAINED, λ ramps 0 -> 0.02 over the first epoch (793 steps) then
# constant, 8 epochs, checkpoint = best KID (as B and λ 0.006), LoRA of epochs 2/4/6/8 kept, control scale 1.0.
# Afterwards: three-seed evaluation (judge 1024 px, 150-image subset) + sample sheet. ≈ 6 V100 GPU-hours.
set -e
DDPM=/home/woody/iwi5/iwi5388h/SENSATION/DDPM
J=$DDPM/controlnet_testing/jobs
O=$DDPM/controlnet_testing/FINAL_v2_segloss/sanpo_l0.02
mkdir -p $O/logs
GPU_OPTS="--partition=v100 --gres=gpu:v100:1 --cpus-per-task=8"
L="--output=$O/logs/%x_%j.out --error=$O/logs/%x_%j.err"
SANPO="DDPM_EXPECT_DATASET=sanpo,DDPM_DATA_ROOT=$DDPM/data_sanpo,DDPM_PROMPTS_FILE_NAME=prompts_meta.csv,DDPM_VAL_SUBSET_SIZE=150,DDPM_EVAL_KID_SUBSET=50"
B="DDPM_LORA_TARGETS=attn_ff,DDPM_TRAIN_PROMPT_KEEP_PROB=0.7,DDPM_SEG_JUDGE=mapillary,DDPM_CONTROLNET_LORA_RANK=16,DDPM_CONTROLNET_LORA_ALPHA=16,DDPM_BATCH_SIZE=4,DDPM_CONTROLNET_LORA_LR=1e-4,DDPM_CONTROLNET_LORA_DROPOUT=0.0,DDPM_GRADIENT_ACCUMULATION_STEPS=1,DDPM_NUM_WORKERS=8"
SEG="DDPM_USE_SEGMENTOR_LOSS=true,DDPM_SEGMENTOR_LOSS_WEIGHT_START=0.0,DDPM_SEGMENTOR_LOSS_WEIGHT=0.02,DDPM_SEGMENTOR_LOSS_WARMUP_STEPS=793,DDPM_SEG_INPUT_SCALE=1"

export RUN_NOTE="TRAINING, SANPO (dataset 2). Step 17, intended final: B recipe + segmentation loss λ 0.02 (0 -> 0.02 over epoch 1, then constant). Between λ 0.006 (realistic, mIoU 0.544) and λ 0.05->0.4 (mIoU 0.705, KID 4x worse). Target: mIoU near real photos (0.627) with KID/FID near B. Checkpoint = best KID. Control scale 1.0."
TR=$(sbatch --parsable $GPU_OPTS --job-name=segloss_l002_sanpo_train --time=12:00:00 $L \
  --export=ALL,OUT_DIR=$O/run,DDPM_RUN_NAME=v2_segloss_l002_sanpo,$SANPO,$B,$SEG,DDPM_NUM_EPOCHS=8,DDPM_EVAL_EVERY_N_EPOCHS=2,DDPM_CHECKPOINT_METRIC=kid,DDPM_SAVE_EVAL_CHECKPOINTS=true $J/train.sh)
echo "train $TR (run id sanpo_$TR)"

export RUN_NOTE="EVALUATION x3 seeds, SANPO, of run sanpo_$TR (B + seg loss λ 0.02, best-KID epoch). Scored exactly like B (1848221), λ 0.006 (1848504) and λ 0.05->0.4 (1848904): Mapillary judge 1024 px, 150-image subset, control scale 1.0, guidance 6.0."
echo "eval x3 $(sbatch --parsable $GPU_OPTS --dependency=afterok:$TR --job-name=segloss_l002_sanpo_eval --time=05:00:00 $L \
  --export=ALL,TAG=v2_segloss_l002_sanpo,SEEDS="42 43 44",OUT=$O/eval,CKPT=$O/run/checkpoints/sanpo_${TR}_LoRA.pt,DDPM_SEG_JUDGE=mapillary,DDPM_LORA_TARGETS=attn_ff,DDPM_CONTROLNET_LORA_RANK=16,DDPM_CONTROLNET_LORA_ALPHA=16,DDPM_BATCH_SIZE=4,$SANPO $J/baseline_eval.sh)"

export RUN_NOTE="SAMPLE SHEET, SANPO: real | mask | pretrained | B (no seg loss) | λ 0.006 | λ 0.02 (run sanpo_$TR) | λ 0.05->0.4; same masks, prompts and noise per row."
echo "sheet $(sbatch --parsable $GPU_OPTS --dependency=afterok:$TR --job-name=segloss_l002_sanpo_sheet --time=01:30:00 $L --export=ALL,$SANPO \
  $J/comparison_grid.sh --out $O/comparison --models "B (v2)=$DDPM/controlnet_testing/FINAL_v2/sanpo/run/checkpoints/sanpo_1848221_Full" \
  "seg 0.006=$DDPM/controlnet_testing/FINAL_v2_segloss/sanpo/run/checkpoints/sanpo_1848504_Full" \
  "seg 0.02=$O/run/checkpoints/sanpo_${TR}_Full" \
  "seg 0.05-0.4=$DDPM/controlnet_testing/FINAL_v2_segloss/sanpo_ramp/run/checkpoints/sanpo_1848904_Full")"
