#!/bin/bash
# Step 18 (2026-10-25): score the SANPO models with the SECOND, independent judge — SegFormer-B5 trained on
# ADE20K (DDPM_SEG_JUDGE=ade20k): different data and architecture than the Mapillary judge that drives the seg
# loss, never trained on our datasets. Shared 12 classes, same targets, judge input 640 px (its training size).
# (SegFormer Cityscapes was tested first and rejected: it calls 83 % of SANPO sidewalk "road".)
# Each model: seed 42 (the same generated images as seed 42 of the Mapillary evaluations: same seed, prompts,
# masks, control scale 1.0, guidance 6.0) on the 150-image SANPO subset. Every generated image is saved to
# images/<model>/ (file name = real photo) so the final images can be inspected next to the real photos.
# RTX3080 (judge is small: 85M params). ≈ 5 x 40 min. Submit only with the user's OK.
# Afterwards: split_by_white.py (analyze_job.sh) — low-white vs high-white halves of the test set, in two folders.
#   bash submit_second_judge.sh <λ 0.02 run id, e.g. sanpo_1850042> [training job id of that run to wait for]
set -e
L002=${1:?give the λ 0.02 run id, e.g. sanpo_1850042}
TRAIN_JOB=${2:-}   # only if that training job is still queued or running (a finished job is rejected as dependency)
DDPM=/home/woody/iwi5/iwi5388h/SENSATION/DDPM
T=$DDPM/controlnet_testing
O=$T/step18_second_judge
mkdir -p $O/logs $O/eval $O/images
GPU_OPTS="--partition=rtx3080 --gres=gpu:rtx3080:1 --cpus-per-task=8"
SANPO="DDPM_EXPECT_DATASET=sanpo,DDPM_DATA_ROOT=$DDPM/data_sanpo,DDPM_PROMPTS_FILE_NAME=prompts_meta.csv,DDPM_VAL_SUBSET_SIZE=150,DDPM_EVAL_KID_SUBSET=50"
LORA="DDPM_LORA_TARGETS=attn_ff,DDPM_CONTROLNET_LORA_RANK=16,DDPM_CONTROLNET_LORA_ALPHA=16"
JUDGE="DDPM_SEG_JUDGE=ade20k,DDPM_SEG_INPUT_SCALE=1.25,DDPM_BATCH_SIZE=4"

IDS=""
submit () {  # tag, label, checkpoint ("" = pretrained) [, dependency]
  export RUN_NOTE="SECOND JUDGE (SegFormer-B5 ADE20K, 640 px, never trained on our data, not used in training), SANPO, model: $2. Seed 42, 150-image subset, control 1.0, guidance 6.0; generated images saved to images/$1/."
  local id
  id=$(sbatch --parsable $GPU_OPTS ${4:+--dependency=$4} --job-name=judge2_$1 --time=02:00:00 --output=$O/logs/%x_%j.out --error=$O/logs/%x_%j.err \
    --export=ALL,TAG=judge2_$1,SEEDS=42,OUT=$O/eval,${3:+CKPT=$3,}DDPM_EVAL_SAVE_DIR=$O/images/$1,$SANPO,$LORA,$JUDGE $T/jobs/baseline_eval.sh)
  echo "$2: $id"; IDS="$IDS:$id"
}
submit pretrained   "pretrained"                ""
submit B            "B (no seg loss)"           $T/FINAL_v2/sanpo/run/checkpoints/sanpo_1848221_LoRA.pt
submit seg0.006     "B + seg λ 0.006"           $T/FINAL_v2_segloss/sanpo/run/checkpoints/sanpo_1848504_LoRA.pt
submit seg0.02      "B + seg λ 0.02 (final)"    $T/FINAL_v2_segloss/sanpo_l0.02/run/checkpoints/${L002}_LoRA.pt ${TRAIN_JOB:+afterok:$TRAIN_JOB}
submit seg0.05-0.4  "B + seg λ 0.05→0.4"        $T/FINAL_v2_segloss/sanpo_ramp/run/checkpoints/sanpo_1848904_LoRA.pt

export RUN_NOTE="ANALYSIS, SANPO: low-white vs high-white halves of the 150 test images (median split of the mask's white/void share). Per group: KID, LPIPS, CLIP, mIoU with Mapillary (512 px) and ADE20K (640 px) judges for pretrained, B, λ 0.006, λ 0.02, λ 0.05→0.4 and the real photos; images copied into images_by_white/{low_white,high_white}/."
echo "split by white: $(sbatch --parsable $GPU_OPTS --dependency=afterok$IDS --job-name=judge2_split_by_white --time=01:30:00 \
  --output=$O/logs/%x_%j.out --error=$O/logs/%x_%j.err --export=ALL,DDPM_SEG_JUDGE=ade20k,$SANPO $O/analyze_job.sh)"
