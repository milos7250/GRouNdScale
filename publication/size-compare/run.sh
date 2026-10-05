#!/bin/bash
#SBATCH --job-name=causalgan-size
#SBATCH --output=logs/causalgan-size/%A/%a.out
#SBATCH --error=logs/causalgan-size/%A/%a.err
#SBATCH --nodes=1
#SBATCH --nodelist=n23-64-512-aragog,n23-64-512-buckbeak,n23-64-512-crookshanks,n23-64-512-dobby,n23-64-512-fawkes,n23-64-512-nagini,n23-64-1024-hedwig,n24-64-384-angel,n24-64-384-anya,n24-64-384-darla,n24-64-384-drusilla,n24-64-384-lorne,n24-64-384-spike
#SBATCH --gpus=a100:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --array=33-50%3

set -exuo pipefail

export CONFIG="causal_gan.cfg";
export CODE_ROOT="${CODE_ROOT:-/mnt/shared/scratch/mmicik/private/Geneformer/simulation/GRouNdGAN.worktrees/dedup}"
export GROUNDSCALE_LOGLEVEL=DEBUG
export GROUNDSCALE_NO_TQDM=1
export PYTHON="apptainer exec --nv 'docker://milos7250/groundscale:latest' python"
python(){
    apptainer exec --nv 'docker://milos7250/groundscale:latest' python "$@"
}

# SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-999999999999}
# SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-5}


PID=""
on_exit() {
    status=$?
    if [[ -n "$PID" ]]; then
        kill -SIGINT "$PID" 2>/dev/null || true
    fi
    if (( status != 0 )); then
        scancel -t PENDING "$SLURM_ARRAY_JOB_ID" || true
    fi
    exit "$status"
}
trap on_exit EXIT

export SIZE=$((1000 + 500 * SLURM_ARRAY_TASK_ID))
export NUM_GENES=$SIZE

"$CODE_ROOT/scripts/preprocess.sh"
"$CODE_ROOT/scripts/create-grn.sh"

NUM_GENES="$(eval "$PYTHON" "$CODE_ROOT/../..//inspect_h5ad.py" "results/${SIZE}/train.h5ad" |grep -oE '\([0-9]+, [0-9]+\)' |cut -d',' -f2 | grep -oE '[0-9]+')"
export NUM_GENES

python "$CODE_ROOT/scripts/measure-gpu-usage.py" --output "results/${SIZE}/gpu-usage.csv" &
PID=$!

"$CODE_ROOT/scripts/train.sh"

kill -SIGINT "$PID"

python "$CODE_ROOT/scripts/count-parameters.py" --checkpoint "results/${SIZE}/checkpoints/step_20.pth" --out "results/${SIZE}/params-size.json"
