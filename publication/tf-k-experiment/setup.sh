#!/bin/bash
#SBATCH --job-name=causalgan-tf-num
#SBATCH --output=logs/causalgan-tf-num/%A.out
#SBATCH --error=logs/causalgan-tf-num/%A.err
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

set -exuo pipefail

export CONFIG="causal_gan.cfg";
export CODE_ROOT="${CODE_ROOT:-/mnt/shared/scratch/mmicik/private/Geneformer/simulation/GRouNdGAN.worktrees/dedup}"
export GROUNDGAN_LOGLEVEL=DEBUG
export GROUNDGAN_NO_TQDM=1
export PYTHON="apptainer exec --nv 'docker://milos7250/groundscale:latest' python"
python(){
    apptainer exec --nv 'docker://milos7250/groundscale:latest' python "$@"
}

export OUT_DIR="../data/processed"
export TOP_K=15
export NUM_GENES=3000

"$CODE_ROOT/scripts/preprocess.sh"
"$CODE_ROOT/scripts/create-grn.sh"
