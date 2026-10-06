#!/bin/bash
#SBATCH --job-name=gan-preprocess
#SBATCH --output=logs/gan-preprocess/%A.out
#SBATCH --partition=short
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G

set -euo pipefail
CODE_ROOT="${CODE_ROOT:-/mnt/shared/scratch/mmicik/private/Geneformer/simulation/GRouNdScale}"
source "$CODE_ROOT/scripts/common.sh"

python \
    "$CODE_ROOT/src/main.py" \
    --config "$CONFIG" \
    --preprocess

rm -rf "$MKTEMP"
