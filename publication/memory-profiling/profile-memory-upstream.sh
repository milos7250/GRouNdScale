#!/bin/bash
#SBATCH --job-name=causalgan-size
#SBATCH --output=logs/causalgan-size/%A/%a.out
#SBATCH --error=logs/causalgan-size/%A/%a.err
#SBATCH --nodes=1
#SBATCH --nodelist=n23-64-512-aragog,n23-64-512-buckbeak,n23-64-512-crookshanks,n23-64-512-dobby,n23-64-512-fawkes,n23-64-512-nagini,n23-64-1024-hedwig,n24-64-384-angel,n24-64-384-anya,n24-64-384-darla,n24-64-384-drusilla,n24-64-384-lorne,n24-64-384-spike
#SBATCH --gpus=a100:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G

set -euo pipefail

export CONFIG=causal_gan_upstream.cfg
export SIZE=$1
export NUM_GENES=$SIZE

sed "s|\${SIZE}|$SIZE|g" "${CONFIG%.cfg}_template.cfg" > "${CONFIG%.cfg}.cfg"
sed -i "s|\${NUM_GENES}|$NUM_GENES|g" "${CONFIG%.cfg}.cfg"

apptainer exec --nv "docker://milos7250/groundscale:latest" \
    python \
    profile_memory_upstream.py
