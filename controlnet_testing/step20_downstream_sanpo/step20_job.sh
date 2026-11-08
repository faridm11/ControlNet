#!/bin/bash
# Step 20 job: runs the given command from DDPM/ with the project environment and the standard log header.
#   sbatch ... step20_job.sh python controlnet_testing/step20_downstream_sanpo/<script>.py <args>
DDPM_ROOT=/home/woody/iwi5/iwi5388h/SENSATION/DDPM
export http_proxy=http://proxy.nhr.fau.de:80 https_proxy=http://proxy.nhr.fau.de:80 PYTHONDONTWRITEBYTECODE=1
module load python/3.12-conda
module load cuda/12.4.1
REQUESTED_DATA_ROOT=${DDPM_DATA_ROOT:-}
conda activate control2
[ -n "$REQUESTED_DATA_ROOT" ] && export DDPM_DATA_ROOT=$REQUESTED_DATA_ROOT
export HF_HOME=$WORK/huggingface_cache HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONPATH=$DDPM_ROOT:$PYTHONPATH
cd "$DDPM_ROOT"
echo "======================================================================"
echo "Date:      $(date '+%Y-%m-%d %H:%M:%S')"
echo "Job:       $SLURM_JOB_ID ($SLURM_JOB_NAME) on $SLURM_NODELIST"
echo "Tests:     ${RUN_NOTE:-(no description given)}"
echo "Dataset:   $DDPM_DATA_ROOT"
echo "Command:   $*"
echo "Settings:"
env | grep '^DDPM_' | sort | sed 's/^/  /'
echo "======================================================================"
"$@"
EXIT_CODE=$?
echo "Finished at $(date) | exit code $EXIT_CODE"
exit $EXIT_CODE
