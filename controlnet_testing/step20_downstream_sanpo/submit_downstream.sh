#!/bin/bash
# Step 20 (2026-11-01): does generated data help a segmentation model when real photos are limited? (user's final experiment)
#   A: 300 real SANPO photos | B: + 1000 base ControlNet images | C: + 1000 images of the λ 0.025 -> 0.045 model (sanpo_1858162)
# SegFormer-B5 from ADE20K weights, 12 classes, 3000 steps, 3 seeds, white/void ignored per pixel, tested on all 522 real test photos.
# User's choice: the existing generator, trained on all 3,172 photos, including the 300 and the photos of the 1000 masks.
# So C has seen more real data than A and B: report the result as "with a well-trained generator", not as a pure low-data result.
set -e
DDPM=/home/woody/iwi5/iwi5388h/SENSATION/DDPM
H=$DDPM/controlnet_testing/step20_downstream_sanpo
P=controlnet_testing/step20_downstream_sanpo
mkdir -p $H/logs
GPU_OPTS="--partition=rtx3080 --gres=gpu:rtx3080:1 --cpus-per-task=8"
L="--output=$H/logs/%x_%j.out --error=$H/logs/%x_%j.err"
ENVS="ALL,DDPM_EXPECT_DATASET=sanpo,DDPM_DATA_ROOT=$DDPM/data_sanpo,DDPM_PROMPTS_FILE_NAME=prompts_meta.csv,DDPM_SEG_JUDGE=ade20k"
OURS=$DDPM/controlnet_testing/FINAL_v2_segloss/sanpo_l0.025-0.045/run/checkpoints/sanpo_1858162_Full

export RUN_NOTE="GENERATION, SANPO, step 20: 1000 images with the BASE ControlNet (pretrained, not fine-tuned), one per mask in splits.json other_masks. Seed 1234, DDIM 35, guidance 6.0, control 1.0, prompts_meta."
G1=$(sbatch --parsable $GPU_OPTS --job-name=down_gen_base --time=05:00:00 $L --export=$ENVS $H/step20_job.sh \
  python $P/generate_from_masks.py --controlnet pretrained --out $H/synthetic/base)
echo "generate base $G1"
export RUN_NOTE="GENERATION, SANPO, step 20: 1000 images with OUR model (B + seg loss λ 0.025 -> 0.045, sanpo_1858162, trained on all 3172 photos), same masks, prompts and noise as the base images."
G2=$(sbatch --parsable $GPU_OPTS --job-name=down_gen_ours --time=05:00:00 $L --export=$ENVS $H/step20_job.sh \
  python $P/generate_from_masks.py --controlnet $OURS --out $H/synthetic/ours)
echo "generate ours $G2"

train () {  # name, label, dependency, synthetic dir
  export RUN_NOTE="SEGMENTATION TRAINING, SANPO, step 20, setup $2. SegFormer-B5 from ADE20K weights, 12 classes, 3000 steps, batch 4, LR 6e-5, seeds 0 1 2, 50 % of samples synthetic when present, white/void ignored per pixel. Test: all 522 real SANPO test photos."
  echo "train $1 $(sbatch --parsable $GPU_OPTS ${3:+--dependency=afterok:$3} --job-name=down_$1 --time=06:00:00 $L --export=$ENVS $H/step20_job.sh \
    python $P/train_downstream.py --name $1 ${4:+--synthetic $4})"
}
train real_only "A: 300 real photos" ""
train real_plus_base "B: 300 real + 1000 base ControlNet images" $G1 $H/synthetic/base
train real_plus_ours "C: 300 real + 1000 images of our λ 0.025 -> 0.045 model" $G2 $H/synthetic/ours
