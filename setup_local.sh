#!/bin/bash
# Setup script for local development environment
# Run this on your laptop with: bash setup_local.sh

set -e  # Exit on error

echo "========================================================================"
echo "ControlNet Local Environment Setup"
echo "========================================================================"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: Conda not found. Please install Anaconda or Miniconda first."
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo ""
echo "[1/4] Creating conda environment 'controlnet'..."
conda create -n controlnet python=3.12 -y

echo ""
echo "[2/4] Activating environment..."
eval "$(conda shell.bash hook)"
conda activate controlnet

echo ""
echo "[3/4] Installing PyTorch with CUDA 12.4 support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo ""
echo "[4/4] Installing other dependencies..."
pip install diffusers transformers accelerate pillow pandas tqdm xformers

echo ""
echo "========================================================================"
echo "✓ Environment setup complete!"
echo "========================================================================"
echo ""
echo "To activate the environment, run:"
echo "  conda activate controlnet"
echo ""
echo "To verify GPU is available:"
echo "  python -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}\")'"
echo ""
echo "Next steps:"
echo "  1. Copy a subset of data from HPC to your laptop"
echo "  2. Set DDPM_DATA_ROOT environment variable"
echo "  3. Run training with: python -m src.train"
echo ""
