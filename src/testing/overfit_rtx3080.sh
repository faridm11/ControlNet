#!/bin/bash
#SBATCH --job-name=controlnet_overfit8
#SBATCH --output=/home/woody/iwi5/iwi5388h/SENSATION/DDPM/test/outputs/logs/overfit8_%j.out
#SBATCH --error=/home/woody/iwi5/iwi5388h/SENSATION/DDPM/test/outputs/logs/overfit8_%j.err
#SBATCH --time=02:30:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16

# ============================================================================
# Overfit 8 samples on RTX 3080 (sanity check)
# Outputs go to DDPM/test/outputs
# ============================================================================

echo "========================================================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================================================"

# Proxy for HPC internet access
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

export PYTHONDONTWRITEBYTECODE=1

# Modules
module load python/3.12-conda
module load cuda/12.4.1

conda activate control2

# Paths
export DDPM_DATA_ROOT=/home/woody/iwi5/iwi5388h/SENSATION/DDPM/data
export DDPM_OUTPUT_ROOT=/home/woody/iwi5/iwi5388h/SENSATION/DDPM/test/outputs
export HF_HOME=$WORK/huggingface_cache
export HF_DATASETS_CACHE=$HF_HOME
export PYTHONPATH=/home/woody/iwi5/iwi5388h/SENSATION/DDPM:$PYTHONPATH

# Silence HuggingFace download bars in logs
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_DISABLE_PROGRESS_BARS=1

# Output dirs
mkdir -p /home/woody/iwi5/iwi5388h/SENSATION/DDPM/test/outputs/checkpoints
mkdir -p /home/woody/iwi5/iwi5388h/SENSATION/DDPM/test/outputs/logs
mkdir -p /home/woody/iwi5/iwi5388h/SENSATION/DDPM/test/outputs/samples
mkdir -p $HF_HOME

cd /home/woody/iwi5/iwi5388h/SENSATION/DDPM

# Clear stale bytecode
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

echo ""
echo "Python:  $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:    $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU:     $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
echo ""
echo "========================================================================"
echo "STARTING OVERFIT-8 RUN"
echo "========================================================================"
echo ""

python -m src.testing.overfit_8_samples --num-samples 2 --epochs 2000 --batch-size 2 --accum-steps 1

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "Job finished at: $(date) | Exit code: $EXIT_CODE"
echo "========================================================================"
exit $EXIT_CODE
