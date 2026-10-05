#!/bin/bash
#SBATCH --job-name=causalgan-tf-k
#SBATCH --output=logs/causalgan-tf-k/%A/%a.out
#SBATCH --error=logs/causalgan-tf-k/%A/%a.err
#SBATCH --nodes=1
#SBATCH --nodelist=n23-64-512-aragog,n23-64-512-buckbeak,n23-64-512-crookshanks,n23-64-512-dobby,n23-64-512-fawkes,n23-64-512-nagini,n23-64-1024-hedwig,n24-64-384-angel,n24-64-384-anya,n24-64-384-darla,n24-64-384-drusilla,n24-64-384-lorne,n24-64-384-spike
#SBATCH --gpus=a100:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --array=1-15%1

set -exuo pipefail

export CONFIG="causal_gan.cfg";
export CODE_ROOT="${CODE_ROOT:-/mnt/shared/scratch/mmicik/private/Geneformer/simulation/GRouNdGAN.worktrees/dedup}"
export GROUNDSCALE_LOGLEVEL=DEBUG
export GROUNDSCALE_NO_TQDM=1
export PYTHON="apptainer exec --nv 'docker://milos7250/groundscale:latest' python"
python(){
    apptainer exec --nv 'docker://milos7250/groundscale:latest' python "$@"
}

SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-999999999999}
SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-1}


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

export TOP_K=$SLURM_ARRAY_TASK_ID
export NUM_GENES=$TOP_K
export OUT_DIR=$TOP_K

mkdir -p "results/${OUT_DIR}"
cp data/processed/inferred_grnboost2.csv data/processed/*.h5ad results/${OUT_DIR}/
"$CODE_ROOT/scripts/create-grn.sh"

NUM_GENES="$(eval "$PYTHON" "$CODE_ROOT/../..//inspect_h5ad.py" "results/${OUT_DIR}/train.h5ad" |grep -oE '\([0-9]+, [0-9]+\)' |cut -d',' -f2 | grep -oE '[0-9]+')"
export NUM_GENES

python "$CODE_ROOT/scripts/measure-gpu-usage.py" --output "results/${OUT_DIR}/gpu-usage.csv" &
PID=$!

"$CODE_ROOT/scripts/train.sh"

kill -SIGINT "$PID"

python "$CODE_ROOT/scripts/count-parameters.py" --checkpoint "results/${OUT_DIR}/checkpoints/step_20.pth" --out "results/${OUT_DIR}/params-size.json"
