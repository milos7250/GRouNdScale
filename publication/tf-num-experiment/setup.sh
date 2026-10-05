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
export GROUNDSCALE_LOGLEVEL=DEBUG
export GROUNDSCALE_NO_TQDM=1
export PYTHON="apptainer exec --nv 'docker://milos7250/groundscale:latest' python"
python(){
    apptainer exec --nv 'docker://milos7250/groundscale:latest' python "$@"
}

export SIZE="../data/processed"
export NUM_GENES=3000

"$CODE_ROOT/scripts/preprocess.sh"

sed -i "s|TFs = |; TFs = |g" causal_gan.cfg
"$CODE_ROOT/scripts/create-grn.sh"
sed -i "s|; TFs = |TFs = |g" causal_gan.cfg
