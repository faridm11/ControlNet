#!/bin/bash
# CPU, sequential: ADE20K then Mapillary, every 3rd of the 150 evaluation photos (50 images)
cd /home/woody/iwi5/iwi5388h/SENSATION/DDPM
for J in ade20k mapillary; do
  SC=$([ $J = ade20k ] && echo 1.25 || echo 1)
  DIAG_STEP=3 DDPM_SEG_JUDGE=$J DDPM_SEG_INPUT_SCALE=$SC DDPM_DATA_ROOT=$PWD/data_sanpo DDPM_PROMPTS_FILE_NAME=prompts_meta.csv \
  DDPM_VAL_SUBSET_SIZE=150 HF_HOME=$WORK/huggingface_cache HF_HUB_OFFLINE=1 \
  /home/woody/iwi5/iwi5388h/software/private/conda/envs/control2/bin/python -u controlnet_testing/step19_judge_mapping/diagnose_judge_classes.py \
    > controlnet_testing/step19_judge_mapping/diagnose_$J.log 2>&1
done
echo done > controlnet_testing/step19_judge_mapping/DONE
