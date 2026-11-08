#!/bin/bash
# Final flow, rerun of steps 3b + 4C (2026-11-03): job 1859600 crashed (segmentation fault on node tg082 after 200 of 800
# images), so C (1859603) never ran. Same commands as submit_final_flow.sh, node tg082 excluded.
set -e
DDPM=/home/woody/iwi5/iwi5388h/SENSATION/DDPM
H=$DDPM/controlnet_testing/step20_downstream_sanpo
P=controlnet_testing/step20_downstream_sanpo
L="--output=$H/logs/%x_%j.out --error=$H/logs/%x_%j.err"
RTX="--partition=rtx3080 --gres=gpu:rtx3080:1 --cpus-per-task=8 --exclude=tg082"
ENVS="ALL,DDPM_EXPECT_DATASET=sanpo,DDPM_DATA_ROOT=$DDPM/data_sanpo,DDPM_PROMPTS_FILE_NAME=prompts_meta.csv,DDPM_SEG_JUDGE=ade20k"
export RUN_NOTE="GENERATION (rerun of 1859600), SANPO, final flow step 2: 800 images with the NEW ControlNet (run sanpo_1859598, trained on only the 200 photos), 4 per mask of the 200 photos; same masks, prompts and noise as the base images."
G2=$(sbatch --parsable $RTX --job-name=final_gen_ours --time=03:00:00 $L --export=$ENVS $H/step20_job.sh \
  python $P/generate_from_masks.py --controlnet $H/generator_200/run/checkpoints/sanpo_1859598_Full --split limited_200 --per-mask 4 --out $H/synthetic/ours_200x4)
echo "generate ours $G2"
export RUN_NOTE="SEGMENTATION TRAINING, SANPO, final flow step 3, setup C: 200 real + 800 images of the new ControlNet, 4 per mask. SegFormer-B5 from ADE20K weights, 12 classes, 3000 steps, batch 4, LR 6e-5, seeds 0 1 2, 50 % of samples synthetic, white/void ignored per pixel. Test: all 522 real SANPO test photos."
echo "train real_plus_ours_200 $(sbatch --parsable $RTX --dependency=afterok:$G2 --job-name=final_real_plus_ours_200 --time=06:00:00 $L --export=$ENVS $H/step20_job.sh \
  python $P/train_downstream.py --name real_plus_ours_200 --real-split limited_200 --synthetic $H/synthetic/ours_200x4)"
