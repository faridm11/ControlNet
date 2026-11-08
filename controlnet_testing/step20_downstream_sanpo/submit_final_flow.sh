#!/bin/bash
# Step 23 (2026-11-02): FULL FINAL FLOW, SANPO (user's design). Only 200 real photos and their masks are used.
#   1. fine-tune a new ControlNet on the 200 photos only: B recipe + seg loss λ 0.025 -> 0.045 (the selected λ, step 21),
#      3000 steps (15 epochs of 200 steps), λ rising over all steps, checkpoint = last epoch
#   2. generate 4 images per mask (800) with the base ControlNet and with the new ControlNet; same masks, prompts, noise
#   3. segmentation: A 200 real | B 200 real + 800 base | C 200 real + 800 ours. SegFormer-B5, 3000 steps, 3 seeds,
#      white/void ignored per pixel, 50 % synthetic sampling, all 522 real test photos
# ≈ 2.5 V100 GPU-hours + 3 RTX3080 GPU-hours.
set -e
DDPM=/home/woody/iwi5/iwi5388h/SENSATION/DDPM
J=$DDPM/controlnet_testing/jobs
H=$DDPM/controlnet_testing/step20_downstream_sanpo
P=controlnet_testing/step20_downstream_sanpo
O=$H/generator_200
mkdir -p $O/logs
L="--output=$H/logs/%x_%j.out --error=$H/logs/%x_%j.err"
RTX="--partition=rtx3080 --gres=gpu:rtx3080:1 --cpus-per-task=8"
TRAIN_DATA="DDPM_EXPECT_DATASET=sanpo,DDPM_DATA_ROOT=$DDPM/data_sanpo_200,DDPM_PROMPTS_FILE_NAME=prompts_meta.csv,DDPM_VAL_SUBSET_SIZE=150,DDPM_EVAL_KID_SUBSET=50"
B="DDPM_LORA_TARGETS=attn_ff,DDPM_TRAIN_PROMPT_KEEP_PROB=0.7,DDPM_SEG_JUDGE=mapillary,DDPM_CONTROLNET_LORA_RANK=16,DDPM_CONTROLNET_LORA_ALPHA=16,DDPM_BATCH_SIZE=4,DDPM_CONTROLNET_LORA_LR=1e-4,DDPM_CONTROLNET_LORA_DROPOUT=0.0,DDPM_GRADIENT_ACCUMULATION_STEPS=1,DDPM_NUM_WORKERS=8"
SEG="DDPM_USE_SEGMENTOR_LOSS=true,DDPM_SEGMENTOR_LOSS_WEIGHT_START=0.025,DDPM_SEGMENTOR_LOSS_WEIGHT=0.045,DDPM_SEGMENTOR_LOSS_WARMUP_STEPS=3000,DDPM_SEG_INPUT_SCALE=1"
ENVS="ALL,DDPM_EXPECT_DATASET=sanpo,DDPM_DATA_ROOT=$DDPM/data_sanpo,DDPM_PROMPTS_FILE_NAME=prompts_meta.csv,DDPM_SEG_JUDGE=ade20k"

export RUN_NOTE="TRAINING, SANPO, final flow step 1: new ControlNet fine-tuned on ONLY 200 real photos (data_sanpo_200). B recipe + seg loss λ 0.025 -> 0.045 over 3000 steps (15 epochs of 200 steps), checkpoint = last epoch, control scale 1.0. Mapillary judge never trained on our data."
TR=$(sbatch --parsable --partition=v100 --gres=gpu:v100:1 --cpus-per-task=8 --job-name=final_gen200_train --time=06:00:00 --output=$O/logs/%x_%j.out --error=$O/logs/%x_%j.err \
  --export=ALL,OUT_DIR=$O/run,DDPM_RUN_NAME=final_gen200_sanpo,$TRAIN_DATA,$B,$SEG,DDPM_NUM_EPOCHS=15,DDPM_EVAL_EVERY_N_EPOCHS=5,DDPM_CHECKPOINT_METRIC=last,DDPM_SAVE_EVAL_CHECKPOINTS=true $J/train.sh)
echo "train generator $TR (run id sanpo_$TR)"

export RUN_NOTE="GENERATION, SANPO, final flow step 2: 800 images with the BASE ControlNet, 4 per mask of the 200 photos. Seed 1234, DDIM 35, guidance 6.0, control 1.0, prompts_meta."
G1=$(sbatch --parsable $RTX --job-name=final_gen_base --time=03:00:00 $L --export=$ENVS $H/step20_job.sh \
  python $P/generate_from_masks.py --controlnet pretrained --split limited_200 --per-mask 4 --out $H/synthetic/base_200x4)
echo "generate base $G1"
export RUN_NOTE="GENERATION, SANPO, final flow step 2: 800 images with the NEW ControlNet (run sanpo_$TR, trained on only the 200 photos), 4 per mask of the 200 photos; same masks, prompts and noise as the base images."
G2=$(sbatch --parsable $RTX --dependency=afterok:$TR --job-name=final_gen_ours --time=03:00:00 $L --export=$ENVS $H/step20_job.sh \
  python $P/generate_from_masks.py --controlnet $O/run/checkpoints/sanpo_${TR}_Full --split limited_200 --per-mask 4 --out $H/synthetic/ours_200x4)
echo "generate ours $G2"

train () {  # name, label, dependency, synthetic dir
  export RUN_NOTE="SEGMENTATION TRAINING, SANPO, final flow step 3, setup $2. SegFormer-B5 from ADE20K weights, 12 classes, 3000 steps, batch 4, LR 6e-5, seeds 0 1 2, 50 % of samples synthetic when present, white/void ignored per pixel. Test: all 522 real SANPO test photos."
  echo "train $1 $(sbatch --parsable $RTX ${3:+--dependency=afterok:$3} --job-name=final_$1 --time=06:00:00 $L --export=$ENVS $H/step20_job.sh \
    python $P/train_downstream.py --name $1 --real-split limited_200 ${4:+--synthetic $4})"
}
train real_only_200 "A: 200 real photos" ""
train real_plus_base_200 "B: 200 real + 800 base images, 4 per mask" $G1 $H/synthetic/base_200x4
train real_plus_ours_200 "C: 200 real + 800 images of the new ControlNet, 4 per mask" $G2 $H/synthetic/ours_200x4
