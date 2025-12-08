#!/bin/bash
#SBATCH --job-name=controlnet_phase1
#SBATCH --output=/home/woody/iwi5/iwi5388h/SENSATION/DDPM/logs/train_phase1_%j.out
#SBATCH --error=/home/woody/iwi5/iwi5388h/SENSATION/DDPM/logs/train_phase1_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# ============================================================================
# ControlNet Training - Phase 1: Diffusion Loss Only
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

# Activate conda environment (conda init already done in .bashrc)
conda activate controlnet

# Verify environment
echo ""
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
echo ""

# Set environment variables for paths
export DDPM_DATA_ROOT=/home/woody/iwi5/iwi5388h/SENSATION/DDPM/data
export DDPM_OUTPUT_ROOT=$WORK/controlnet_outputs
export HF_HOME=$WORK/huggingface_cache

# Set Hugging Face cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

# Set PYTHONPATH
export PYTHONPATH=/home/woody/iwi5/iwi5388h/SENSATION/DDPM:$PYTHONPATH

# Create output directories
mkdir -p $DDPM_OUTPUT_ROOT
mkdir -p $HF_HOME
mkdir -p /home/woody/iwi5/iwi5388h/SENSATION/DDPM/logs

# Print configuration
echo "========================================================================"
echo "Environment Configuration"
echo "========================================================================"
echo "DDPM_DATA_ROOT: $DDPM_DATA_ROOT"
echo "DDPM_OUTPUT_ROOT: $DDPM_OUTPUT_ROOT"
echo "HF_HOME: $HF_HOME"
echo "PYTHONPATH: $PYTHONPATH"
echo "========================================================================"

# Navigate to project directory
cd /home/woody/iwi5/iwi5388h/SENSATION/DDPM

# Clear Python cache to avoid import issues
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Run training
echo ""
echo "Starting training..."
echo "========================================================================"

python -m src.train

# Training finished
echo ""
echo "========================================================================"
echo "Job finished at: $(date)"
echo "========================================================================"
