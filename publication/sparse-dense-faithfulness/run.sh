#!/bin/bash
#SBATCH --job-name=p2-sparse-dense
#SBATCH --output=logs/%A.out
#SBATCH --error=logs/%A.err
#SBATCH --nodes=1
#SBATCH --nodelist=n23-64-512-aragog,n23-64-512-buckbeak,n23-64-512-crookshanks,n23-64-512-dobby,n23-64-512-fawkes,n23-64-512-nagini,n23-64-1024-hedwig,n24-64-384-angel,n24-64-384-anya,n24-64-384-darla,n24-64-384-drusilla,n24-64-384-lorne,n24-64-384-spike
#SBATCH --gpus=a100:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Numerical faithfulness of GRouNdScale's sparse masked layers to GRouNdGAN's dense MaskedLinear, in FP32 and TF32.
# 32 log-spaced GRN sizes G = 10..2500 (F = 63, k = 15, depth 2, width 1, batch 1024), 4 random GRNs/weights/inputs
# each; see faithfulness.py. Sizes whose dense twin does not fit on the GPU are recorded as out of memory and skipped.
#
# Submit from this directory: sbatch run.sh [extra faithfulness.py run options]
# A resubmission resumes results/records.jsonl; pass --overwrite to start again.

set -exuo pipefail

export EXPERIMENT="sparse-dense-faithfulness"
export CODE_ROOT="${CODE_ROOT:-/mnt/shared/scratch/mmicik/private/Geneformer/simulation/GRouNdScale}"
export CONFIG="$CODE_ROOT/publication/causal_gan.cfg" # only needed by common.sh's existence check
source "$CODE_ROOT/scripts/common.sh" # sets the container python() and $MKTEMP

# The upstream checkout (GRouNdGAN at 2df087f) is a symlink to a directory next to CODE_ROOT, so bind it too.
UPSTREAM="$(readlink -f "$CODE_ROOT/upstream")"
export APPTAINER_BIND="${APPTAINER_BIND:+$APPTAINER_BIND,}$CODE_ROOT,$UPSTREAM"
export PYTHONUNBUFFERED=1

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

python faithfulness.py run --out results --device cuda --upstream "$UPSTREAM" "$@"
python faithfulness.py summarize --out results > /dev/null
echo "Summary written to $PWD/results/summary.json"
