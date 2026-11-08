#!/bin/bash
# SANPO: B recipe ("v2") + segmentation loss with a SMALL rising weight, λ 0.025 -> 0.045 (step 21, 2026-10-31) — last λ check.
# Goal (user): mIoU closer to the real photos than λ 0.02 (0.565, real 0.627) while KID stays well below λ 0.05 -> 0.4 (0.045).
# Same as submit_segloss_l002_sanpo.sh (job 1850042) except:
#   - λ rises linearly 0.025 -> 0.045 over the WHOLE training (8 epochs x 793 steps = 6344 steps)
#   - checkpoint = LAST epoch (λ 0.045 reached), as the other whole-training ramp (job 1848904); LoRA of epochs 2/4/6/8 kept
# Afterwards: three-seed evaluation (Mapillary judge 1024 px, 150-image subset), sample sheet, and the independent
# ADE20K judge (seed 42, 640 px) with images saved next to the other models in step18_second_judge/images/.
# ≈ 6 V100 GPU-hours + 1 RTX3080 GPU-hour. Submit only with the user's OK.
set -e
DDPM=/home/woody/iwi5/iwi5388h/SENSATION/DDPM
T=$DDPM/controlnet_testing
J=$T/jobs
O=$T/FINAL_v2_segloss/sanpo_l0.025-0.045
S18=$T/step18_second_judge
mkdir -p $O/logs
GPU_OPTS="--partition=v100 --gres=gpu:v100:1 --cpus-per-task=8"
L="--output=$O/logs/%x_%j.out --error=$O/logs/%x_%j.err"
SANPO="DDPM_EXPECT_DATASET=sanpo,DDPM_DATA_ROOT=$DDPM/data_sanpo,DDPM_PROMPTS_FILE_NAME=prompts_meta.csv,DDPM_VAL_SUBSET_SIZE=150,DDPM_EVAL_KID_SUBSET=50"
B="DDPM_LORA_TARGETS=attn_ff,DDPM_TRAIN_PROMPT_KEEP_PROB=0.7,DDPM_SEG_JUDGE=mapillary,DDPM_CONTROLNET_LORA_RANK=16,DDPM_CONTROLNET_LORA_ALPHA=16,DDPM_BATCH_SIZE=4,DDPM_CONTROLNET_LORA_LR=1e-4,DDPM_CONTROLNET_LORA_DROPOUT=0.0,DDPM_GRADIENT_ACCUMULATION_STEPS=1,DDPM_NUM_WORKERS=8"
SEG="DDPM_USE_SEGMENTOR_LOSS=true,DDPM_SEGMENTOR_LOSS_WEIGHT_START=0.025,DDPM_SEGMENTOR_LOSS_WEIGHT=0.045,DDPM_SEGMENTOR_LOSS_WARMUP_STEPS=6344,DDPM_SEG_INPUT_SCALE=1"

export RUN_NOTE="TRAINING, SANPO (dataset 2). Step 21, last λ check: B recipe + segmentation loss λ rising linearly 0.025 -> 0.045 over all 8 epochs. Between λ 0.02 (mIoU 0.565, KID 0.016) and λ 0.05->0.4 (mIoU 0.705, KID 0.045). Does mIoU get closer to the real photos (0.627) without losing realism? Checkpoint = last epoch; epochs 2/4/6/8 LoRA kept. Control scale 1.0."
TR=$(sbatch --parsable $GPU_OPTS --job-name=segloss_l0025_0045_sanpo_train --time=12:00:00 $L \
  --export=ALL,OUT_DIR=$O/run,DDPM_RUN_NAME=v2_segloss_l0025_0045_sanpo,$SANPO,$B,$SEG,DDPM_NUM_EPOCHS=8,DDPM_EVAL_EVERY_N_EPOCHS=2,DDPM_CHECKPOINT_METRIC=last,DDPM_SAVE_EVAL_CHECKPOINTS=true $J/train.sh)
echo "train $TR (run id sanpo_$TR)"

export RUN_NOTE="EVALUATION x3 seeds, SANPO, of run sanpo_$TR (B + seg loss, λ 0.025 -> 0.045, last epoch). Scored exactly like B (1848221), λ 0.006 (1848504), λ 0.02 (1850042) and λ 0.05->0.4 (1848904): Mapillary judge 1024 px, 150-image subset, control scale 1.0, guidance 6.0."
echo "eval x3 $(sbatch --parsable $GPU_OPTS --dependency=afterok:$TR --job-name=segloss_l0025_0045_sanpo_eval --time=05:00:00 $L \
  --export=ALL,TAG=v2_segloss_l0025_0045_sanpo,SEEDS="42 43 44",OUT=$O/eval,CKPT=$O/run/checkpoints/sanpo_${TR}_LoRA.pt,DDPM_SEG_JUDGE=mapillary,DDPM_LORA_TARGETS=attn_ff,DDPM_CONTROLNET_LORA_RANK=16,DDPM_CONTROLNET_LORA_ALPHA=16,DDPM_BATCH_SIZE=4,$SANPO $J/baseline_eval.sh)"

export RUN_NOTE="SAMPLE SHEET, SANPO: real | mask | pretrained | B (no seg loss) | λ 0.02 | λ 0.025->0.045 (run sanpo_$TR) | λ 0.05->0.4; same masks, prompts and noise per row."
echo "sheet $(sbatch --parsable $GPU_OPTS --dependency=afterok:$TR --job-name=segloss_l0025_0045_sanpo_sheet --time=01:30:00 $L --export=ALL,$SANPO \
  $J/comparison_grid.sh --out $O/comparison --models "B (v2)=$T/FINAL_v2/sanpo/run/checkpoints/sanpo_1848221_Full" \
  "seg 0.02=$T/FINAL_v2_segloss/sanpo_l0.02/run/checkpoints/sanpo_1850042_Full" \
  "seg 0.025-0.045=$O/run/checkpoints/sanpo_${TR}_Full" \
  "seg 0.05-0.4=$T/FINAL_v2_segloss/sanpo_ramp/run/checkpoints/sanpo_1848904_Full")"

export RUN_NOTE="SECOND JUDGE (SegFormer-B5 ADE20K, 640 px, never trained on our data, not used in training), SANPO, model: λ 0.025 -> 0.045 (run sanpo_$TR). Seed 42, 150-image subset, control 1.0, guidance 6.0; generated images saved to step18_second_judge/images/seg0.025-0.045/."
echo "second judge $(sbatch --parsable --partition=rtx3080 --gres=gpu:rtx3080:1 --cpus-per-task=8 --dependency=afterok:$TR --job-name=judge2_seg0.025-0.045 --time=02:00:00 $L \
  --export=ALL,TAG=judge2_seg0.025-0.045,SEEDS=42,OUT=$S18/eval,CKPT=$O/run/checkpoints/sanpo_${TR}_LoRA.pt,DDPM_EVAL_SAVE_DIR=$S18/images/seg0.025-0.045,$SANPO,DDPM_LORA_TARGETS=attn_ff,DDPM_CONTROLNET_LORA_RANK=16,DDPM_CONTROLNET_LORA_ALPHA=16,DDPM_SEG_JUDGE=ade20k,DDPM_SEG_INPUT_SCALE=1.25,DDPM_BATCH_SIZE=4 $J/baseline_eval.sh)"
