#!/bin/bash
#SBATCH --job-name=p2-target-genes-upstream
#SBATCH --output=logs/p2-target-genes-upstream/%j.out
#SBATCH --error=logs/p2-target-genes-upstream/%j.err
#SBATCH --nodes=1
#SBATCH --nodelist=n23-64-512-aragog,n23-64-512-buckbeak,n23-64-512-crookshanks,n23-64-512-dobby,n23-64-512-fawkes,n23-64-512-nagini,n23-64-1024-hedwig,n24-64-384-angel,n24-64-384-anya,n24-64-384-darla,n24-64-384-drusilla,n24-64-384-lorne,n24-64-384-spike
#SBATCH --gpus=a100:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# The target-gene-num experiment on the original implementation (upstream/), on the data and graphs of ../target-gene-num.
# One job runs the sizes one after the other and stops after the first run that runs out of memory, as the larger ones
# would too. The original implementation runs in its own image, with Python 3.9.
# Submit from this directory: sbatch run.sh

set -exuo pipefail

export EXPERIMENT="target-gene-num-upstream"
export CODE_ROOT="${CODE_ROOT:-/mnt/shared/scratch/mmicik/private/Geneformer/simulation/GRouNdScale}"
export CONFIG="$CODE_ROOT/publication2/target-gene-num-upstream/causal_gan_upstream.cfg"
source "$CODE_ROOT/scripts/common.sh" # sets $MKTEMP and the environment of the other scripts, and enables set -euo pipefail

# The container only sees the working directory, so the repository and the real location of its upstream/ link are bound.
export APPTAINER_BIND="${APPTAINER_BIND:+$APPTAINER_BIND,}$CODE_ROOT,$(readlink -f "$CODE_ROOT/upstream")"
export PYTHONUNBUFFERED=1

upstream_python() {
    apptainer exec --nv 'docker://yazdanz/groundgan:GRouNdGAN-toolkit' python3.9 "$@"
}

trap 'rm -rf "$MKTEMP"' EXIT

SIZES=(1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 16789)

for G in "${SIZES[@]}"; do
    export DATA_DIR="../target-gene-num/data/processed/${G}"
    export CAUSAL_GRAPH="$DATA_DIR/causal_graph.pkl"
    export OUT_DIR="results/g${G}/upstream"

    mkdir -p "$OUT_DIR"
    rm -f "$OUT_DIR/summary.json" "$OUT_DIR/trace.json" "$OUT_DIR/nvidia-smi.csv"

    upstream_python "$CODE_ROOT/publication2/profile-memory-upstream.py" \
        --template "$CONFIG" \
        --experiment "$EXPERIMENT" \
        --run "g${G}"

    # Checkpoints, t-SNE plots and TensorBoard logs are not results, nor is the saved causal controller.
    # KEEP_OUTPUTS=1 keeps them.
    if [[ "${KEEP_OUTPUTS:-0}" != "1" ]]; then
        find "${OUT_DIR:?}" -mindepth 1 -maxdepth 1 \
            ! -name summary.json ! -name trace.json ! -name nvidia-smi.csv ! -name causal_gan.cfg \
            -exec rm -rf {} +
        rm -rf "${OUT_DIR:?}_CC"
    fi

    if grep -q '"status": "oom"' "$OUT_DIR/summary.json"; then
        echo "G=${G} ran out of memory, stopping: the larger sizes would too"
        break
    fi
done
