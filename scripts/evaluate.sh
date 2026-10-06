#!/bin/bash
#SBATCH --job-name=causalgan-evaluate
#SBATCH --output=logs/causalgan-evaluate/%A.out
#SBATCH --error=logs/causalgan-evaluate/%A.err
#SBATCH --nodes=1
#SBATCH --nodelist=n23-64-512-aragog,n23-64-512-buckbeak,n23-64-512-crookshanks,n23-64-512-dobby,n23-64-512-fawkes,n23-64-512-nagini,n23-64-1024-hedwig,n24-64-384-angel,n24-64-384-anya,n24-64-384-darla,n24-64-384-drusilla,n24-64-384-lorne,n24-64-384-spike
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

set -euo pipefail
CODE_ROOT="${CODE_ROOT:-/mnt/shared/scratch/mmicik/private/Geneformer/simulation/GRouNdScale}"
source "$CODE_ROOT/scripts/common.sh"


python \
    "$CODE_ROOT/src/main.py" \
    --config "$CONFIG" \
    --evaluate
