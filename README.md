# ControlNet Training - Local Setup

Quick setup script for running ControlNet training on your local machine with RTX GPU.

## Prerequisites

- Anaconda or Miniconda installed
- NVIDIA GPU with CUDA support (RTX 4050 or better)
- At least 8GB GPU VRAM

## Setup Instructions

### 1. Clone the repository

```bash
git clone git@github.com:faridm11/ControlNet.git
cd ControlNet
```

### 2. Run setup script

```bash
bash setup_local.sh
```

This will:
- Create conda environment `controlnet` with Python 3.12
- Install PyTorch with CUDA 12.4 support
- Install all required dependencies (diffusers, transformers, etc.)

### 3. Prepare data

Copy a small subset of data from HPC for testing:

```bash
# Create data directory
mkdir -p data/train/{images,labels,prompts}
mkdir -p data/test/{images,labels,prompts}

# Copy from HPC using scp (adjust paths as needed)
scp username@hpc:/path/to/DDPM/data/train/images/image_{0..9}.png data/train/images/
scp username@hpc:/path/to/DDPM/data/train/labels/image_{0..9}.png data/train/labels/
scp username@hpc:/path/to/DDPM/data/train/prompts/prompts.csv data/train/prompts/

# Copy test data
scp username@hpc:/path/to/DDPM/data/test/images/image_{128..132}.png data/test/images/
scp username@hpc:/path/to/DDPM/data/test/labels/image_{128..132}.png data/test/labels/
scp username@hpc:/path/to/DDPM/data/test/prompts/prompts.csv data/test/prompts/
```

### 4. Verify GPU

```bash
conda activate controlnet
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
CUDA: True
GPU: NVIDIA GeForce RTX 4050
```

### 5. Run training

```bash
conda activate controlnet
export DDPM_DATA_ROOT=$(pwd)/data
python -m src.train
```

## Configuration

The code automatically detects local vs HPC environment. No code changes needed!

### Reduce memory usage for RTX 4050

Edit `src/config.py` if needed:

```python
BATCH_SIZE = 1  # Keep at 1
GRADIENT_ACCUMULATION_STEPS = 2  # Reduce from 4 to 2
NUM_EPOCHS = 5  # Reduce for quick testing
```

## Testing without full training

For quick tests, reduce epochs and samples in config or use debug mode:

```bash
export DEBUG_MODE=1
python -m src.train
```

## Troubleshooting

### Out of memory

- Reduce `GRADIENT_ACCUMULATION_STEPS` to 2
- Ensure no other programs are using GPU
- Close browser/other GPU applications

### CUDA not available

- Check NVIDIA drivers: `nvidia-smi`
- Reinstall PyTorch: `pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu124`

### Import errors

- Clear Python cache: `find . -type d -name __pycache__ -exec rm -rf {} +`
- Set: `export PYTHONDONTWRITEBYTECODE=1`

## Directory Structure

```
ControlNet/
├── data/                  # Your data (not in git)
│   ├── train/
│   │   ├── images/
│   │   ├── labels/
│   │   └── prompts/
│   └── test/
│       ├── images/
│       ├── labels/
│       └── prompts/
├── src/
│   ├── train.py           # Training script
│   ├── config.py          # Configuration
│   ├── models/
│   └── data/
├── jobs/                  # HPC job scripts
├── setup_local.sh         # This setup script
└── README.md
```

## Next Steps

After verifying training works locally, push changes and submit to HPC for full training:

```bash
git add .
git commit -m "Your changes"
git push

# On HPC
cd DDPM
git pull
sbatch jobs/train_phase1.sh
```
