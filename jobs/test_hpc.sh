#!/bin/bash
#SBATCH --job-name=test_controlnet
#SBATCH --partition=tinygpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err

echo "=========================================="
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Load modules
module load python/3.12-conda
module load cuda/12.4.1

# Activate conda environment
source activate controlnet

# Print environment info
echo ""
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")')"
echo ""

# ============================================================================
# CONFIGURE PATHS - Adjust these based on your data location
# ============================================================================
export DDPM_DATA_ROOT="$WORK/sidewalk_semantic"
export DDPM_OUTPUT_ROOT="$WORK/SENSATION/DDPM/outputs"
export HF_HOME="$WORK/.cache/huggingface"
export PYTHONPATH="$WORK/SENSATION:$PYTHONPATH"

echo "Paths configured:"
echo "  Data: $DDPM_DATA_ROOT"
echo "  Output: $DDPM_OUTPUT_ROOT"
echo "  Cache: $HF_HOME"
echo ""

# Run test
cd $WORK/SENSATION
echo "Testing ControlNet model initialization..."
python -m DDPM.src.models.diffusion_controlnet

echo ""
echo "=========================================="
echo "Job finished at $(date)"
echo "=========================================="
