#!/bin/bash
#SBATCH --job-name=causalgan-mem-profile
#SBATCH --output=logs/causalgan-mem-profile/%A/%a.out
#SBATCH --error=logs/causalgan-mem-profile/%A/%a.err
#SBATCH --nodes=1
#SBATCH --nodelist=n23-64-512-aragog,n23-64-512-buckbeak,n23-64-512-crookshanks,n23-64-512-dobby,n23-64-512-fawkes,n23-64-512-nagini,n23-64-1024-hedwig,n24-64-384-angel,n24-64-384-anya,n24-64-384-darla,n24-64-384-drusilla,n24-64-384-lorne,n24-64-384-spike
#SBATCH --gpus=a100:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --array=2-2%3

set -euo pipefail

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

export SIZE=$(( SLURM_ARRAY_TASK_ID * 500))
export PYTHON="apptainer exec --nv 'docker://milos7250/groundscale:latest' python"
python(){
    apptainer exec --nv 'docker://milos7250/groundscale:latest' python "$@"
}

NUM_GENES="$(eval "$PYTHON" "../../scripts/inspect_h5ad.py" "../size-compare/results/${SIZE}/train.h5ad" |grep -oE '\([0-9]+, [0-9]+\)' |cut -d',' -f2 | grep -oE '[0-9]+')"
export NUM_GENES

python "../../scripts/measure-gpu-usage.py" --output "results/${SIZE}/gpu-usage.csv" &
PID=$!

python \
    profile_memory.py \
    "$SIZE-compile"

kill -SIGINT "$PID"
