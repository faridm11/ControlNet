#!/bin/bash
# FINAL Phase 2 on dataset 1 (sidewalk): continue the FINAL Phase 1 LoRA (job 1843106) for 10 epochs at LR 5e-5,
# once with the segmentation-consistency loss (weight 0.01, best trade-off in step 6b) and once without (control).
# Same schedule for both, last epoch kept, so the difference is the effect of the loss. Each is followed by a
# three-seed evaluation. Starts automatically after the Phase 1 training job succeeds.
# Estimated cost: ≈ 1.8 A100 GPU-hours.
set -e
# GPU type. The group's A100 quota was exhausted on 2026-10-18 (jobs stuck on AssocGrpGRES), V100s were free.
# V100 (32 GB, no bf16): training falls back to fp16 + gradient scaling automatically; ~2x slower than A100.
GPU=${GPU:-v100}
GPU_OPTS="--partition=$GPU --gres=gpu:$GPU:1 --cpus-per-task=8"
# Generation-only jobs (evaluation, sample sheets) fit on a 10 GB RTX 3080 and use that queue
EVAL_GPU=${EVAL_GPU:-rtx3080}
EVAL_GPU_OPTS="--partition=$EVAL_GPU --gres=gpu:$EVAL_GPU:1 --cpus-per-task=8"
DDPM=/home/woody/iwi5/iwi5388h/SENSATION/DDPM
F=$DDPM/controlnet_testing/FINAL/sidewalk_phase2
J=$DDPM/controlnet_testing/jobs
P1=1843106
INIT=$DDPM/controlnet_testing/FINAL/sidewalk/run/checkpoints/${P1}_LoRA.pt
DATA="DDPM_EXPECT_DATASET=sidewalk,DDPM_PROMPTS_FILE_NAME=prompts_objects.csv"
RANK="DDPM_CONTROLNET_LORA_RANK=16,DDPM_CONTROLNET_LORA_ALPHA=16,DDPM_BATCH_SIZE=4"
COMMON="$DATA,$RANK,DDPM_CONTROLNET_LORA_DROPOUT=0.0,DDPM_GRADIENT_ACCUMULATION_STEPS=1,DDPM_NUM_WORKERS=8,\
DDPM_CONTROLNET_LORA_LR=5e-5,DDPM_NUM_EPOCHS=10,DDPM_EVAL_EVERY_N_EPOCHS=5,DDPM_CHECKPOINT_METRIC=last,DDPM_INIT_LORA_CHECKPOINT=$INIT"
L="--output=$F/logs/%x_%j.out --error=$F/logs/%x_%j.err"
for V in "control " "segloss_w0.01 DDPM_USE_SEGMENTOR_LOSS=true,DDPM_SEGMENTOR_LOSS_WEIGHT=0.01"; do
  set -- $V
  TR=$(sbatch --parsable $GPU_OPTS --dependency=afterok:$P1 --job-name=final_sidewalk_p2_$1 --time=04:00:00 $L \
    --export=ALL,OUT_DIR=$F/$1,DDPM_RUN_NAME=final_sidewalk_p2_$1,$COMMON${2:+,$2} $J/train.sh)
  echo "$1 train $TR"
  echo "$1 eval x3 $(sbatch --parsable $EVAL_GPU_OPTS --dependency=afterok:$TR --job-name=final_sidewalk_p2_${1}_eval --time=02:00:00 $L \
    --export=ALL,TAG=final_sidewalk_p2_$1,SEEDS="42 43 44",CKPT=$F/$1/checkpoints/${TR}_LoRA.pt,OUT=$F/eval,$RANK,$DATA $J/baseline_eval.sh)"
done
