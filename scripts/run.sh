#!/bin/bash
#SBATCH --job-name=causalgan
#SBATCH --output=logs/causalgan/%A.out
#SBATCH --error=logs/causalgan/%A.err
#SBATCH --nodes=1
# SBATCH --nodelist=n24-64-384-angel,n24-64-384-anya,n23-64-512-aragog,n23-64-512-buckbeak,n23-64-512-crookshanks,n24-64-384-darla,n23-64-512-dobby,n24-64-384-drusilla,n23-64-512-fawkes,n23-64-1024-hedwig,n24-64-384-lorne,n23-64-512-nagini,n24-64-384-spike
#SBATCH --gpus=a100:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G

set -euo pipefail
CODE_ROOT="${CODE_ROOT:-/mnt/shared/scratch/mmicik/private/Geneformer/simulation/GRouNdScale}"
source "$CODE_ROOT/scripts/common.sh"

python \
    "$CODE_ROOT/src/main.py" \
    --config "$CONFIG" \
    "$@"

rm -rf "$MKTEMP"