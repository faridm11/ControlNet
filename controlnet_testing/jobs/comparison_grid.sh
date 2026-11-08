#!/bin/bash
#SBATCH --job-name=cmp_grid
#SBATCH --time=01:30:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#
# Side-by-side sample sheet. Arguments after the script name are passed on:
#   sbatch controlnet_testing/jobs/comparison_grid.sh --out <dir> --models "label=<..._Full>" ...
DDPM_ROOT=/home/woody/iwi5/iwi5388h/SENSATION/DDPM
module load python/3.12-conda
module load cuda/12.4.1
# The control2 activation script (etc/conda/activate.d/env_vars.sh) resets DDPM_DATA_ROOT to dataset 1.
# Keep the data root the submitter asked for, and restore it after activation.
REQUESTED_DATA_ROOT=${DDPM_DATA_ROOT:-}
conda activate control2
[ -n "$REQUESTED_DATA_ROOT" ] && export DDPM_DATA_ROOT=$REQUESTED_DATA_ROOT
export DDPM_DATA_ROOT=${DDPM_DATA_ROOT:-$DDPM_ROOT/data}   # dataset 1 unless the submitter chose another
echo "DATA ROOT: $DDPM_DATA_ROOT"
export DDPM_OUTPUT_ROOT=$DDPM_ROOT/controlnet_testing/reporting/scratch_outputs
export HF_HOME=$WORK/huggingface_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_VERBOSITY=error
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$DDPM_ROOT:$PYTHONPATH
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
echo "Arguments: $*"
python controlnet_testing/reporting/make_comparison_grid.py "$@"
