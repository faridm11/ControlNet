#!/bin/bash
# FINAL v2 (2026-10-23). Everything on V100. Two stages:
#
# Stage 1 — re-score with the new judge (Mask2Former Swin-L, Mapillary Vistas; DDPM_SEG_JUDGE=mapillary):
#   pretrained ControlNet and the v1 final model, both datasets, three seeds -> FINAL/<dataset>/eval_mapillary/
#   (these jobs also act as the canary for the new judge code: stage 2 starts only if the final-model ones succeed)
# Stage 2 — v2 training, both datasets, then three-seed evaluation + sample sheet -> FINAL_v2/<dataset>/
#   changes vs v1: LoRA also on transformer projections + feed-forward layers (best in the step-11 screen),
#   training-time prompt augmentation (each true condition/object phrase kept with p=0.7, order shuffled),
#   evaluation with the Mapillary judge. Same epochs, LR, rank, prompts files and evaluation sets as v1.
# Estimated cost ≈ 9 V100 GPU-hours (stage 1 ≈ 1.5, stage 2 ≈ 7.5).
set -e
DDPM=/home/woody/iwi5/iwi5388h/SENSATION/DDPM
J=$DDPM/controlnet_testing/jobs
GPU_OPTS="--partition=v100 --gres=gpu:v100:1 --cpus-per-task=8"
JUDGE="DDPM_SEG_JUDGE=mapillary"
RANK="DDPM_CONTROLNET_LORA_RANK=16,DDPM_CONTROLNET_LORA_ALPHA=16,DDPM_BATCH_SIZE=4"
SIDEWALK="DDPM_EXPECT_DATASET=sidewalk,DDPM_PROMPTS_FILE_NAME=prompts_objects.csv"
SANPO="DDPM_EXPECT_DATASET=sanpo,DDPM_DATA_ROOT=$DDPM/data_sanpo,DDPM_PROMPTS_FILE_NAME=prompts_meta.csv,DDPM_VAL_SUBSET_SIZE=150,DDPM_EVAL_KID_SUBSET=50"

ev() {  # name, out dir, data settings, checkpoint ("" = pretrained), extra settings, dependency
  local L="--output=$2/logs/%x_%j.out --error=$2/logs/%x_%j.err"; mkdir -p $2/logs
  sbatch --parsable $GPU_OPTS ${6:+--dependency=afterok:$6} --job-name=$1 --time=05:00:00 $L \
    --export=ALL,TAG=$1,SEEDS="42 43 44",OUT=$2,$RANK,$JUDGE,$3${4:+,CKPT=$4}${5:+,$5} $J/baseline_eval.sh
}

# ---- stage 1
F=$DDPM/controlnet_testing/FINAL
S1A=$(ev rescore_pretrained_sidewalk $F/sidewalk/eval_mapillary "$SIDEWALK" ""); echo "stage1 pretrained sidewalk $S1A"
S1B=$(ev rescore_final_sidewalk $F/sidewalk/eval_mapillary "$SIDEWALK" $F/sidewalk/run/checkpoints/1843532_LoRA.pt); echo "stage1 final sidewalk $S1B"
S1C=$(ev rescore_pretrained_sanpo $F/sanpo/eval_mapillary "$SANPO" ""); echo "stage1 pretrained sanpo $S1C"
S1D=$(ev rescore_final_sanpo $F/sanpo/eval_mapillary "$SANPO" $F/sanpo/run/checkpoints/sanpo_1843536_LoRA.pt); echo "stage1 final sanpo $S1D"

# ---- stage 2
V2=$DDPM/controlnet_testing/FINAL_v2
V2SET="DDPM_LORA_TARGETS=attn_ff,DDPM_TRAIN_PROMPT_KEEP_PROB=0.7,$JUDGE,$RANK,DDPM_CONTROLNET_LORA_LR=1e-4,DDPM_CONTROLNET_LORA_DROPOUT=0.0,DDPM_GRADIENT_ACCUMULATION_STEPS=1,DDPM_NUM_WORKERS=8"
v2() {  # dataset, data settings, epochs, eval every, run-id prefix, time limit
  local O=$V2/$1; mkdir -p $O/logs; local L="--output=$O/logs/%x_%j.out --error=$O/logs/%x_%j.err"
  local TR=$(sbatch --parsable $GPU_OPTS --dependency=afterok:$S1B:$S1D --job-name=v2_$1_train --time=$6 $L \
    --export=ALL,OUT_DIR=$O/run,DDPM_RUN_NAME=v2_$1,$V2SET,$2,DDPM_NUM_EPOCHS=$3,DDPM_EVAL_EVERY_N_EPOCHS=$4 $J/train.sh)
  echo "stage2 $1 train $TR (run id $5$TR)"
  echo "stage2 $1 eval x3 $(ev v2_$1 $O/eval "$2" $O/run/checkpoints/$5${TR}_LoRA.pt "DDPM_LORA_TARGETS=attn_ff" $TR)"
  echo "stage2 $1 sheet $(sbatch --parsable $GPU_OPTS --dependency=afterok:$TR --job-name=v2_$1_sheet --time=01:30:00 $L \
    --export=ALL,$2 $J/comparison_grid.sh --out $O/comparison --models "v1 final=$F/$1/run/checkpoints/$([ $1 = sanpo ] && echo sanpo_1843536 || echo 1843532)_Full" "v2=$O/run/checkpoints/$5${TR}_Full")"
}
v2 sidewalk "$SIDEWALK" 20 4 "" 08:00:00
v2 sanpo "$SANPO" 8 2 "sanpo_" 12:00:00
