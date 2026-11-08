#!/bin/bash
#SBATCH --job-name=controlnet_a100
#SBATCH --output=outputs/logs/train_phase1_%j.out
#SBATCH --error=outputs/logs/train_phase1_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16

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
conda activate control2

# Verify environment
echo ""
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
echo ""

# SLURM_SUBMIT_DIR is always set by SLURM to the directory where sbatch was called.
# Submit from DDPM/: sbatch jobs/train_phase1.sh
DDPM_ROOT="$SLURM_SUBMIT_DIR"

# Set environment variables for paths
export DDPM_DATA_ROOT=$DDPM_ROOT/data
export DDPM_OUTPUT_ROOT=$DDPM_ROOT/outputs
export HF_HOME=$WORK/huggingface_cache

# Set Hugging Face cache
export HF_DATASETS_CACHE=$HF_HOME

# All models are already in the cache — never depend on the Hub being reachable
export HF_HUB_OFFLINE=1

# Disable progress bars in logs
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_DISABLE_PROGRESS_BARS=1

# Set PYTHONPATH
export PYTHONPATH=$DDPM_ROOT:$PYTHONPATH

# Create output directories
mkdir -p $DDPM_OUTPUT_ROOT/checkpoints
mkdir -p $DDPM_OUTPUT_ROOT/logs
mkdir -p $DDPM_OUTPUT_ROOT/samples
mkdir -p $HF_HOME

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
cd "$DDPM_ROOT"

# Clear Python cache to avoid import issues
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Run training
echo ""
echo "Starting training..."
echo "========================================================================"

python -m src.train

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "Job finished at: $(date) | Exit code: $EXIT_CODE"
echo "========================================================================"
exit $EXIT_CODE
