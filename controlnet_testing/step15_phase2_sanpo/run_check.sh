#!/bin/bash
#SBATCH --job-name=sanpo_segloss_check
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
DDPM_ROOT=/home/woody/iwi5/iwi5388h/SENSATION/DDPM
module load python/3.12-conda
module load cuda/12.4.1
conda activate control2
# set after activation (control2's activation script resets DDPM_DATA_ROOT)
export DDPM_DATA_ROOT=$DDPM_ROOT/data_sanpo DDPM_EXPECT_DATASET=sanpo
export DDPM_SEG_JUDGE=mapillary DDPM_USE_SEGMENTOR_LOSS=true DDPM_LORA_TARGETS=attn_ff
export DDPM_CONTROLNET_LORA_RANK=16 DDPM_CONTROLNET_LORA_ALPHA=16 DDPM_CONTROLNET_LORA_DROPOUT=0.0
export DDPM_VAL_SUBSET_SIZE=150 DDPM_PROMPTS_FILE_NAME=prompts_meta.csv
export DDPM_OUTPUT_ROOT=$DDPM_ROOT/controlnet_testing/step15_phase2_sanpo/scratch_outputs
export HF_HOME=$WORK/huggingface_cache HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$DDPM_ROOT:$PYTHONPATH
cd "$DDPM_ROOT"
echo "DATA ROOT: $DDPM_DATA_ROOT"
python controlnet_testing/step15_phase2_sanpo/check_segloss.py
