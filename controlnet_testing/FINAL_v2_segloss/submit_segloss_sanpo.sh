#!/bin/bash
# SANPO: fine-tuned B recipe ("v2") + segmentation loss, trained from the PRETRAINED ControlNet (2026-10-24).
# Identical to v2 (job 1848221) except the loss: L = L_diffusion + λ·L_seg(Mapillary judge), λ ramps linearly
# 0 -> 0.006 over the first epoch (793 steps), then stays. So "v2 vs this run" = with/without semantic loss
# (the ablation in ControlNet_Segmentor_Project.docx). Judge input 512 px during training (1024 px does not fit
# next to training on a 32 GB V100; accuracy on real photos 0.625 vs 0.627 — step 15a check).
# Afterwards: three-seed evaluation scored exactly like v2 (judge at 1024 px, 150-image subset) + sample sheet.
# Estimated cost ≈ 6 V100 GPU-hours.
set -e
DDPM=/home/woody/iwi5/iwi5388h/SENSATION/DDPM
J=$DDPM/controlnet_testing/jobs
O=$DDPM/controlnet_testing/FINAL_v2_segloss/sanpo
GPU_OPTS="--partition=v100 --gres=gpu:v100:1 --cpus-per-task=8"
L="--output=$O/logs/%x_%j.out --error=$O/logs/%x_%j.err"
SANPO="DDPM_EXPECT_DATASET=sanpo,DDPM_DATA_ROOT=$DDPM/data_sanpo,DDPM_PROMPTS_FILE_NAME=prompts_meta.csv,DDPM_VAL_SUBSET_SIZE=150,DDPM_EVAL_KID_SUBSET=50"
B="DDPM_LORA_TARGETS=attn_ff,DDPM_TRAIN_PROMPT_KEEP_PROB=0.7,DDPM_SEG_JUDGE=mapillary,DDPM_CONTROLNET_LORA_RANK=16,DDPM_CONTROLNET_LORA_ALPHA=16,DDPM_BATCH_SIZE=4,DDPM_CONTROLNET_LORA_LR=1e-4,DDPM_CONTROLNET_LORA_DROPOUT=0.0,DDPM_GRADIENT_ACCUMULATION_STEPS=1,DDPM_NUM_WORKERS=8"
SEG="DDPM_USE_SEGMENTOR_LOSS=true,DDPM_SEGMENTOR_LOSS_WEIGHT=0.006,DDPM_SEGMENTOR_LOSS_WARMUP_STEPS=793,DDPM_SEG_INPUT_SCALE=1"
TR=$(sbatch --parsable $GPU_OPTS --job-name=segloss_sanpo_train --time=12:00:00 $L \
  --export=ALL,OUT_DIR=$O/run,DDPM_RUN_NAME=v2_segloss_sanpo,$SANPO,$B,$SEG,DDPM_NUM_EPOCHS=8,DDPM_EVAL_EVERY_N_EPOCHS=2 $J/train.sh)
echo "train $TR (run id sanpo_$TR)"
echo "eval x3 $(sbatch --parsable $GPU_OPTS --dependency=afterok:$TR --job-name=segloss_sanpo_eval --time=05:00:00 $L \
  --export=ALL,TAG=v2_segloss_sanpo,SEEDS="42 43 44",OUT=$O/eval,CKPT=$O/run/checkpoints/sanpo_${TR}_LoRA.pt,DDPM_SEG_JUDGE=mapillary,DDPM_LORA_TARGETS=attn_ff,DDPM_CONTROLNET_LORA_RANK=16,DDPM_CONTROLNET_LORA_ALPHA=16,DDPM_BATCH_SIZE=4,$SANPO $J/baseline_eval.sh)"
echo "sheet $(sbatch --parsable $GPU_OPTS --dependency=afterok:$TR --job-name=segloss_sanpo_sheet --time=01:30:00 $L --export=ALL,$SANPO \
  $J/comparison_grid.sh --out $O/comparison --models "B (v2)=$DDPM/controlnet_testing/FINAL_v2/sanpo/run/checkpoints/sanpo_1848221_Full" "B + seg loss=$O/run/checkpoints/sanpo_${TR}_Full")"
