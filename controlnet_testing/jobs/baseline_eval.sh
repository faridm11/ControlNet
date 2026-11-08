#!/bin/bash
#SBATCH --job-name=cn_baseline
#SBATCH --time=02:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#
# Evaluation with evaluate.py for one prompt file and class version, repeated over
# several generation seeds. Without CKPT it evaluates the pretrained ControlNet
# (zero-shot baseline); with CKPT=<..._LoRA.pt> it evaluates that checkpoint
# (the DDPM_CONTROLNET_LORA_RANK / _ALPHA of that run must be exported too).
#   sbatch --export=ALL,TAG=<tag>,DDPM_PROMPTS_FILE_NAME=<csv>,SEEDS="42 43 44"[,CKPT=<path>,OUT=<dir>] ...

DDPM_ROOT=/home/woody/iwi5/iwi5388h/SENSATION/DDPM
OUT=${OUT:-$DDPM_ROOT/controlnet_testing/scratch_outputs/eval}
: "${TAG:?TAG must be set}"
SEEDS=${SEEDS:-"42 43 44"}

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
export DDPM_OUTPUT_ROOT=$DDPM_ROOT/controlnet_testing/scratch_outputs
export HF_HOME=$WORK/huggingface_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONPATH=$DDPM_ROOT:$PYTHONPATH

mkdir -p "$OUT"
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
echo "TAG=$TAG | seeds: $SEEDS | checkpoint: ${CKPT:-none (pretrained)} | OUT=$OUT"

FAIL=0
for SEED in $SEEDS; do
    python evaluate.py --seed $SEED --tag ${TAG}_seed${SEED} --output-dir "$OUT" ${CKPT:+--checkpoint "$CKPT"} $EXTRA_ARGS || FAIL=1
done
echo "Finished at $(date) | fail=$FAIL"
exit $FAIL
