#!/bin/bash
# Step 25 (2026-11-08): the final flow (200 real photos, 800 fakes per generator, 4 per mask) with a segmentation model that
# does NOT know street scenes: SegFormer-B5 starting from ImageNet weights (nvidia/mit-b5), new decoder, instead of ADE20K.
# Question: do the fakes help when the model cannot lean on street-scene pretraining? Same fakes, steps, seeds and test as
# the final flow. ≈ 2.5 RTX3080 GPU-hours, three jobs in parallel.
set -e
DDPM=/home/woody/iwi5/iwi5388h/SENSATION/DDPM
H=$DDPM/controlnet_testing/step20_downstream_sanpo
P=controlnet_testing/step20_downstream_sanpo
L="--output=$H/logs/%x_%j.out --error=$H/logs/%x_%j.err"
RTX="--partition=rtx3080 --gres=gpu:rtx3080:1 --cpus-per-task=8 --exclude=tg082"
ENVS="ALL,DDPM_EXPECT_DATASET=sanpo,DDPM_DATA_ROOT=$DDPM/data_sanpo,DDPM_PROMPTS_FILE_NAME=prompts_meta.csv,DDPM_SEG_JUDGE=ade20k"
train () {  # name, label, synthetic dir
  export RUN_NOTE="SEGMENTATION TRAINING, SANPO, step 25 (ImageNet start), setup $2. SegFormer-B5 from ImageNet weights (nvidia/mit-b5, no segmentation pretraining), 12 classes, 3000 steps, batch 4, LR 6e-5, seeds 0 1 2, 50 % of samples synthetic when present, white/void ignored per pixel. Test: all 522 real SANPO test photos."
  echo "train $1 $(sbatch --parsable $RTX --job-name=in_$1 --time=06:00:00 $L --export=$ENVS $H/step20_job.sh \
    python $P/train_downstream.py --name $1 --init imagenet --real-split limited_200 ${3:+--synthetic $3})"
}
train imagenet_real_only_200 "A: 200 real photos" ""
train imagenet_real_plus_base_200 "B: 200 real + 800 base fakes" $H/synthetic/base_200x4
train imagenet_real_plus_ours_200 "C: 200 real + 800 fakes of the 200-photo generator" $H/synthetic/ours_200x4
