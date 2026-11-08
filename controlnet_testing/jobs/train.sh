#!/bin/bash
#SBATCH --job-name=cn_train
#SBATCH --time=06:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#
# Generic training job. Everything that differs between experiments is passed in:
#   sbatch --job-name=<name> --output=<dir>/logs/%x_%j.out --error=<dir>/logs/%x_%j.err \
#          --export=ALL,OUT_DIR=<dir>/<name>,DDPM_RUN_NAME=<name>,DDPM_CONTROLNET_LORA_LR=1e-4,... \
#          controlnet_testing/jobs/train.sh
# OUT_DIR receives checkpoints/, logs/, plots/, samples/ of this run.

DDPM_ROOT=/home/woody/iwi5/iwi5388h/SENSATION/DDPM
: "${OUT_DIR:?OUT_DIR must be set}"

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export PYTHONDONTWRITEBYTECODE=1

module load python/3.12-conda
module load cuda/12.4.1
# The control2 activation script (etc/conda/activate.d/env_vars.sh) resets DDPM_DATA_ROOT to dataset 1.
# Keep the data root the submitter asked for, and restore it after activation.
REQUESTED_DATA_ROOT=${DDPM_DATA_ROOT:-}
conda activate control2
[ -n "$REQUESTED_DATA_ROOT" ] && export DDPM_DATA_ROOT=$REQUESTED_DATA_ROOT

export DDPM_DATA_ROOT=${DDPM_DATA_ROOT:-$DDPM_ROOT/data}   # dataset 1 unless the submitter chose another
echo "DATA ROOT: $DDPM_DATA_ROOT"
export DDPM_OUTPUT_ROOT=$OUT_DIR
export HF_HOME=$WORK/huggingface_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONPATH=$DDPM_ROOT:$PYTHONPATH

mkdir -p "$OUT_DIR"
cd "$DDPM_ROOT"

# Header: date, what is tested (RUN_NOTE, exported by the submit script), dataset, all settings
echo "======================================================================"
echo "Date:      $(date '+%Y-%m-%d %H:%M:%S')"
echo "Job:       $SLURM_JOB_ID ($SLURM_JOB_NAME) on $SLURM_NODELIST"
echo "Tests:     ${RUN_NOTE:-(no description given)}"
echo "Dataset:   $DDPM_DATA_ROOT"
echo "Settings:"
env | grep '^DDPM_' | sort | sed 's/^/  /'
echo "======================================================================"
echo "OUT_DIR=$OUT_DIR"

if [ -n "$DRY_RUN" ]; then
    python -c "from src import config; print('DRY RUN ->', config.DATASET_NAME, config.CLASS_VERSION, config.NUM_CLASSES, config.DATA_DIR, config.VAL_PROMPTS_FILE.name, 'val subset', config.VAL_SUBSET_SIZE)"
    exit $?
fi
python -m src.train
EXIT_CODE=$?
echo "Finished at $(date) | exit code $EXIT_CODE"
exit $EXIT_CODE
