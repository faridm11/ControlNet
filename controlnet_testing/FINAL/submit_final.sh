#!/bin/bash
# FINAL runs — one per dataset, with the best configuration found so far (see EXPERIMENT_LOG.md).
# Each final run is followed automatically by:
#   - a three-seed evaluation of the selected checkpoint (the numbers to quote)
#   - a three-seed evaluation of the pretrained ControlNet on the same images and prompts (the baseline)
#   - a side-by-side sample sheet: real photo | mask | pretrained | fine-tuned
# Everything lands in controlnet_testing/FINAL/<dataset>/.
# Estimated cost: sidewalk ≈ 1.8, SANPO ≈ 3.3 A100 GPU-hours (roughly double the wall time on V100).
set -e
# GPU type. The group's A100 quota was exhausted on 2026-10-18 (jobs stuck on AssocGrpGRES), V100s were free.
# V100 (32 GB, no bf16): training falls back to fp16 + gradient scaling automatically; ~2x slower than A100.
GPU=${GPU:-v100}
GPU_OPTS="--partition=$GPU --gres=gpu:$GPU:1 --cpus-per-task=8"
# Generation-only jobs (evaluation, sample sheets) fit on a 10 GB RTX 3080 and use that queue
EVAL_GPU=${EVAL_GPU:-rtx3080}
EVAL_GPU_OPTS="--partition=$EVAL_GPU --gres=gpu:$EVAL_GPU:1 --cpus-per-task=8"
DDPM=/home/woody/iwi5/iwi5388h/SENSATION/DDPM
F=$DDPM/controlnet_testing/FINAL
J=$DDPM/controlnet_testing/jobs
RECIPE="DDPM_CONTROLNET_LORA_LR=1e-4,DDPM_CONTROLNET_LORA_RANK=16,DDPM_CONTROLNET_LORA_ALPHA=16,DDPM_CONTROLNET_LORA_DROPOUT=0.0,\
DDPM_BATCH_SIZE=4,DDPM_GRADIENT_ACCUMULATION_STEPS=1,DDPM_NUM_WORKERS=8"
RANK="DDPM_CONTROLNET_LORA_RANK=16,DDPM_CONTROLNET_LORA_ALPHA=16,DDPM_BATCH_SIZE=4"

final() {  # dataset, data settings, epochs, eval every, run-id prefix, time limit
  local D=$1 DATA=$2 EP=$3 EV=$4 PREFIX=$5 TL=$6
  local O=$F/$D L="--output=$F/$D/logs/%x_%j.out --error=$F/$D/logs/%x_%j.err"
  local TR=$(sbatch --parsable $GPU_OPTS --job-name=final_${D}_train --time=$TL $L \
    --export=ALL,OUT_DIR=$O/run,DDPM_RUN_NAME=final_$D,$RECIPE,$DATA,DDPM_NUM_EPOCHS=$EP,DDPM_EVAL_EVERY_N_EPOCHS=$EV $J/train.sh)
  local RID=${PREFIX}${TR}
  echo "$D train $TR (run id $RID)"
  echo "$D eval final x3 $(sbatch --parsable $EVAL_GPU_OPTS --dependency=afterok:$TR --job-name=final_${D}_eval --time=04:00:00 $L \
    --export=ALL,TAG=final_${D},SEEDS="42 43 44",CKPT=$O/run/checkpoints/${RID}_LoRA.pt,OUT=$O/eval,$RANK,$DATA $J/baseline_eval.sh)"
  echo "$D eval pretrained x3 $(sbatch --parsable $EVAL_GPU_OPTS --job-name=final_${D}_baseline --time=04:00:00 $L \
    --export=ALL,TAG=pretrained_${D},SEEDS="42 43 44",OUT=$O/eval,$RANK,$DATA $J/baseline_eval.sh)"
  echo "$D sample sheet $(sbatch --parsable $EVAL_GPU_OPTS --dependency=afterok:$TR --job-name=final_${D}_sheet $L --export=ALL,$DATA \
    $J/comparison_grid.sh --out $O/comparison --models "fine-tuned (final)=$O/run/checkpoints/${RID}_Full")"
}

# Dataset 1 (sidewalk-semantic): 20 epochs = 4,760 steps (plateau at 15-20 epochs in the earlier runs)
final sidewalk "DDPM_EXPECT_DATASET=sidewalk,DDPM_PROMPTS_FILE_NAME=prompts_objects.csv" 20 4 "" 07:00:00
# Dataset 2 (SANPO): 8 epochs = 6,344 steps, evaluation on the fixed 150-image subset
final sanpo "DDPM_EXPECT_DATASET=sanpo,DDPM_DATA_ROOT=$DDPM/data_sanpo,DDPM_PROMPTS_FILE_NAME=prompts_meta.csv,DDPM_VAL_SUBSET_SIZE=150,DDPM_EVAL_KID_SUBSET=50" 8 2 "sanpo_" 10:00:00
