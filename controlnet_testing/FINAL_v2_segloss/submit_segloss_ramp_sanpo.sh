#!/bin/bash
# SANPO: B recipe ("v2") + segmentation loss with a STRONG, slowly rising weight (step 16, 2026-10-24).
# Same as submit_segloss_sanpo.sh (λ 0 -> 0.006, job 1848504) except:
#   - λ rises linearly 0.05 -> 0.4 over the WHOLE training (8 epochs x 793 steps = 6344 steps), user's choice,
#     close to the project document's 0.1 -> 0.5. End of each epoch: 0.094 0.138 0.181 0.225 0.269 0.312 0.356 0.400
#   - checkpoint = LAST epoch (λ = 0.4 reached) instead of best KID: KID would favour the early, low-λ epochs,
#     which is not what this run tests. The LoRA of every evaluated epoch (2, 4, 6, 8) is kept as well
#     (<run>_ep<N>_LoRA.pt), so any point of the λ curve can get a three-seed evaluation later.
# Control scale stays 1.0 (1.2 / 1.5 postponed until the λ result is known).
# Afterwards: three-seed evaluation scored exactly like B and the λ 0.006 run (judge 1024 px, 150-image subset)
# + sample sheet B | B + seg λ 0.006 | B + seg λ 0.05->0.4. Estimated cost ≈ 6 V100 GPU-hours.
set -e
DDPM=/home/woody/iwi5/iwi5388h/SENSATION/DDPM
J=$DDPM/controlnet_testing/jobs
O=$DDPM/controlnet_testing/FINAL_v2_segloss/sanpo_ramp
mkdir -p $O/logs
GPU_OPTS="--partition=v100 --gres=gpu:v100:1 --cpus-per-task=8"
L="--output=$O/logs/%x_%j.out --error=$O/logs/%x_%j.err"
SANPO="DDPM_EXPECT_DATASET=sanpo,DDPM_DATA_ROOT=$DDPM/data_sanpo,DDPM_PROMPTS_FILE_NAME=prompts_meta.csv,DDPM_VAL_SUBSET_SIZE=150,DDPM_EVAL_KID_SUBSET=50"
B="DDPM_LORA_TARGETS=attn_ff,DDPM_TRAIN_PROMPT_KEEP_PROB=0.7,DDPM_SEG_JUDGE=mapillary,DDPM_CONTROLNET_LORA_RANK=16,DDPM_CONTROLNET_LORA_ALPHA=16,DDPM_BATCH_SIZE=4,DDPM_CONTROLNET_LORA_LR=1e-4,DDPM_CONTROLNET_LORA_DROPOUT=0.0,DDPM_GRADIENT_ACCUMULATION_STEPS=1,DDPM_NUM_WORKERS=8"
SEG="DDPM_USE_SEGMENTOR_LOSS=true,DDPM_SEGMENTOR_LOSS_WEIGHT_START=0.05,DDPM_SEGMENTOR_LOSS_WEIGHT=0.4,DDPM_SEGMENTOR_LOSS_WARMUP_STEPS=6344,DDPM_SEG_INPUT_SCALE=1"

export RUN_NOTE="TRAINING, SANPO (dataset 2). Ablation step 16: B recipe + segmentation loss with λ rising linearly 0.05 -> 0.4 over all 8 epochs (earlier run: λ 0 -> 0.006). Does a much stronger seg loss raise mIoU further, and what does it cost in realism (KID/FID)? Checkpoint = last epoch; epochs 2/4/6/8 LoRA kept. Control scale 1.0."
TR=$(sbatch --parsable $GPU_OPTS --job-name=segloss_ramp_sanpo_train --time=12:00:00 $L \
  --export=ALL,OUT_DIR=$O/run,DDPM_RUN_NAME=v2_segloss_ramp_sanpo,$SANPO,$B,$SEG,DDPM_NUM_EPOCHS=8,DDPM_EVAL_EVERY_N_EPOCHS=2,DDPM_CHECKPOINT_METRIC=last,DDPM_SAVE_EVAL_CHECKPOINTS=true $J/train.sh)
echo "train $TR (run id sanpo_$TR)"

export RUN_NOTE="EVALUATION x3 seeds, SANPO, of run sanpo_$TR (B + seg loss, λ 0.05 -> 0.4, last epoch). Scored exactly like B (1848221) and B + seg λ 0.006 (1848504): Mapillary judge 1024 px, 150-image subset, control scale 1.0, guidance 6.0."
echo "eval x3 $(sbatch --parsable $GPU_OPTS --dependency=afterok:$TR --job-name=segloss_ramp_sanpo_eval --time=05:00:00 $L \
  --export=ALL,TAG=v2_segloss_ramp_sanpo,SEEDS="42 43 44",OUT=$O/eval,CKPT=$O/run/checkpoints/sanpo_${TR}_LoRA.pt,DDPM_SEG_JUDGE=mapillary,DDPM_LORA_TARGETS=attn_ff,DDPM_CONTROLNET_LORA_RANK=16,DDPM_CONTROLNET_LORA_ALPHA=16,DDPM_BATCH_SIZE=4,$SANPO $J/baseline_eval.sh)"

export RUN_NOTE="SAMPLE SHEET, SANPO: real | mask | pretrained | B (no seg loss) | B + seg λ 0.006 | B + seg λ 0.05 -> 0.4 (run sanpo_$TR); same masks, prompts and noise per row."
echo "sheet $(sbatch --parsable $GPU_OPTS --dependency=afterok:$TR --job-name=segloss_ramp_sanpo_sheet --time=01:30:00 $L --export=ALL,$SANPO \
  $J/comparison_grid.sh --out $O/comparison --models "B (v2)=$DDPM/controlnet_testing/FINAL_v2/sanpo/run/checkpoints/sanpo_1848221_Full" \
  "B + seg 0.006=$DDPM/controlnet_testing/FINAL_v2_segloss/sanpo/run/checkpoints/sanpo_1848504_Full" \
  "B + seg 0.05-0.4=$O/run/checkpoints/sanpo_${TR}_Full")"
