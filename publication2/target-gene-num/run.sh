#!/bin/bash
#SBATCH --job-name=p2-target-genes
#SBATCH --output=logs/p2-target-genes/%A/%a.out
#SBATCH --error=logs/p2-target-genes/%A/%a.err
#SBATCH --nodes=1
#SBATCH --nodelist=n23-64-512-aragog,n23-64-512-buckbeak,n23-64-512-crookshanks,n23-64-512-dobby,n23-64-512-fawkes,n23-64-512-nagini,n23-64-1024-hedwig,n24-64-384-angel,n24-64-384-anya,n24-64-384-darla,n24-64-384-drusilla,n24-64-384-lorne,n24-64-384-spike
#SBATCH --gpus=a100:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --array=1-34%4

# Tasks 1-34: G = 1000..16000 in steps of 1000 and 16789, each compiled then un-compiled. F = 1000, k = 15.
# Run prepare-data.py once before. Submit from this directory: sbatch run.sh

set -exuo pipefail

export EXPERIMENT="target-gene-num"
export CODE_ROOT="${CODE_ROOT:-/mnt/shared/scratch/mmicik/private/Geneformer/simulation/GRouNdScale}"
export CONFIG="$CODE_ROOT/publication2/causal_gan.cfg"
source "$CODE_ROOT/publication2/common.sh"

TARGETS=(1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 16789)
G=${TARGETS[$RUN_INDEX]}
export DATA_DIR="data/processed/${G}"
export CAUSAL_GRAPH="data/processed/${G}/causal_graph.pkl"
RUN="g${G}"
run_profile "$RUN"
