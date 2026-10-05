#!/bin/bash
#SBATCH --job-name=causalgan-hyperopt
#SBATCH --output=logs/causalgan-hyperopt/%A/%a.out
#SBATCH --error=logs/causalgan-hyperopt/%A/%a.err
#SBATCH --nodes=1
#SBATCH --nodelist=n23-64-512-aragog,n23-64-512-buckbeak,n23-64-512-crookshanks,n23-64-512-dobby,n23-64-512-fawkes,n23-64-512-nagini,n23-64-1024-hedwig,n24-64-384-angel,n24-64-384-anya,n24-64-384-darla,n24-64-384-drusilla,n24-64-384-lorne,n24-64-384-spike
#SBATCH --gpus=a100:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --array=0-20%8

set -euo pipefail
CODE_ROOT="${CODE_ROOT:-/mnt/shared/scratch/mmicik/private/Geneformer/simulation/GRouNdScale.worktrees/dedup}"
source "$CODE_ROOT/scripts/common.sh"

# Sleep random amount of time to avoid race conditions when creating optuna study
sleep $(( RANDOM % 60 ))

python \
    "$CODE_ROOT/src/main.py" \
    --config "$CONFIG" \
    --optimize-hyperparameters

rm -rf "$MKTEMP"
