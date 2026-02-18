#!/bin/bash
#SBATCH --job-name=controlnet_test
#SBATCH --output=/home/woody/iwi5/iwi5388h/SENSATION/DDPM/outputs/logs/test_rtx3080_%j.out
#SBATCH --error=/home/woody/iwi5/iwi5388h/SENSATION/DDPM/outputs/logs/test_rtx3080_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8

# ============================================================================
# Quick Test Run on RTX 3080 (30 minutes, 1 epoch)
# ============================================================================

echo "========================================================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================================================"

# Set proxy for HPC
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

# Prevent Python from writing bytecode files
export PYTHONDONTWRITEBYTECODE=1

# Load required modules
module load python/3.12-conda
module load cuda/12.4.1

# Activate conda environment
conda activate controlnet

# Verify environment
echo ""
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
echo ""

# Set environment variables
export DDPM_DATA_ROOT=/home/woody/iwi5/iwi5388h/SENSATION/DDPM/data
export DDPM_OUTPUT_ROOT=/home/woody/iwi5/iwi5388h/SENSATION/DDPM/outputs
export HF_HOME=$WORK/huggingface_cache

# Disable progress bars in logs
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_DISABLE_PROGRESS_BARS=1

# Create output directories
mkdir -p $DDPM_OUTPUT_ROOT/checkpoints
mkdir -p $DDPM_OUTPUT_ROOT/logs
mkdir -p $DDPM_OUTPUT_ROOT/samples

# Change to project directory
cd /home/woody/iwi5/iwi5388h/SENSATION/DDPM

echo ""
echo "========================================================================"
echo "STARTING TEST TRAINING (1 EPOCH)"
echo "========================================================================"
echo ""

# Run training with config values
python -m src.train

echo ""
echo "========================================================================"
echo "Job finished at: $(date)"
echo "========================================================================"
