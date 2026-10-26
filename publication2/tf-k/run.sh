#!/bin/bash
#SBATCH --job-name=p2-tf-k
#SBATCH --output=logs/p2-tf-k/%A/%a.out
#SBATCH --error=logs/p2-tf-k/%A/%a.err
#SBATCH --nodes=1
#SBATCH --nodelist=n23-64-512-aragog,n23-64-512-buckbeak,n23-64-512-crookshanks,n23-64-512-dobby,n23-64-512-fawkes,n23-64-512-nagini,n23-64-1024-hedwig,n24-64-384-angel,n24-64-384-anya,n24-64-384-darla,n24-64-384-drusilla,n24-64-384-lorne,n24-64-384-spike
#SBATCH --gpus=a100:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --array=1-40%4

# Tasks 1-40: k = 1..20, each compiled then un-compiled. F = 1000, G = 16789 and B, d, w, n are fixed.
# Run prepare-data.py once before. Submit from this directory: sbatch run.sh

set -exuo pipefail

export EXPERIMENT="tf-k"
export CODE_ROOT="${CODE_ROOT:-/mnt/shared/scratch/mmicik/private/Geneformer/simulation/GRouNdScale}"
export CONFIG="$CODE_ROOT/publication2/causal_gan.cfg"
source "$CODE_ROOT/publication2/common.sh"

K=$((RUN_INDEX + 1))
export DATA_DIR=data/base
export CAUSAL_GRAPH="data/graphs/k${K}.pkl"
RUN="k${K}"
run_profile "$RUN"
