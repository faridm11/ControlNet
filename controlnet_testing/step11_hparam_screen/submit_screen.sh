#!/bin/bash
# Step 11 — small hyperparameter screen on dataset 1 before the final runs (one seed each).
# All four training runs share one 10-epoch schedule (2,380 steps, cosine to 0), so they are
# directly comparable; the reference run uses the current best settings.
# Plus an inference-only sweep of guidance scale x control strength on the best existing
# checkpoint (objects prompts, job 1836710). Estimated cost: about 2.9 A100 GPU-hours.
set -e
# GPU type. The group's A100 quota was exhausted on 2026-10-18 (jobs stuck on AssocGrpGRES), V100s were free.
# V100 (32 GB, no bf16): training falls back to fp16 + gradient scaling automatically; ~2x slower than A100.
GPU=${GPU:-v100}
GPU_OPTS="--partition=$GPU --gres=gpu:$GPU:1 --cpus-per-task=8"
# Generation-only jobs (evaluation, sample sheets) fit on a 10 GB RTX 3080 and use that queue
EVAL_GPU=${EVAL_GPU:-rtx3080}
EVAL_GPU_OPTS="--partition=$EVAL_GPU --gres=gpu:$EVAL_GPU:1 --cpus-per-task=8"
DDPM=/home/woody/iwi5/iwi5388h/SENSATION/DDPM
T=$DDPM/controlnet_testing/step11_hparam_screen
J=$DDPM/controlnet_testing/jobs
BASE="DDPM_PROMPTS_FILE_NAME=prompts_objects.csv,DDPM_CONTROLNET_LORA_LR=1e-4,DDPM_CONTROLNET_LORA_RANK=16,DDPM_CONTROLNET_LORA_ALPHA=16,\
DDPM_CONTROLNET_LORA_DROPOUT=0.0,DDPM_BATCH_SIZE=4,DDPM_GRADIENT_ACCUMULATION_STEPS=1,DDPM_NUM_EPOCHS=10,DDPM_EVAL_EVERY_N_EPOCHS=5,DDPM_NUM_WORKERS=8"
run() {  # name, extra settings
  echo "$1 $(sbatch --parsable $GPU_OPTS --job-name=screen_$1 --time=04:00:00 --output=$T/logs/%x_%j.out --error=$T/logs/%x_%j.err \
    --export=ALL,OUT_DIR=$T/$1,DDPM_RUN_NAME=screen_$1,$BASE${2:+,$2} $J/train.sh)"
}
run reference ""
run rank32 "DDPM_CONTROLNET_LORA_RANK=32,DDPM_CONTROLNET_LORA_ALPHA=32"
run attn_ff "DDPM_LORA_TARGETS=attn_ff"
run hflip "DDPM_TRAIN_HFLIP=true"

# inference-only: guidance {4, 6, 7.5} x control strength {0.8, 1.0, 1.2}, seed 42, all 50 test images
CKPT=$DDPM/controlnet_testing/step5_phase1_sweep/lr1e-4_r16_objects/checkpoints/1836710_LoRA.pt
for G in 4.0 6.0 7.5; do
  echo "guidance $G $(sbatch --parsable $EVAL_GPU_OPTS --job-name=screen_g$G --time=02:00:00 --output=$T/logs/%x_%j.out --error=$T/logs/%x_%j.err \
    --export=ALL,TAG=guidance${G},SEEDS=42,CKPT=$CKPT,OUT=$T/inference_sweep,DDPM_PROMPTS_FILE_NAME=prompts_objects.csv,DDPM_BATCH_SIZE=4,\
DDPM_CONTROLNET_LORA_RANK=16,DDPM_CONTROLNET_LORA_ALPHA=16,DDPM_GUIDANCE_SCALE=$G,EXTRA_ARGS="--control-strengths 0.8 1.0 1.2" \
    $J/baseline_eval.sh)"
done
