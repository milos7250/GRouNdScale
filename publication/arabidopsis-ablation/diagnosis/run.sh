#!/bin/bash
#SBATCH --job-name=p2-arabidopsis-ablation-diagnosis
#SBATCH --output=diagnosis/logs/%A_%a.out
#SBATCH --error=diagnosis/logs/%A_%a.err
#SBATCH --nodes=1
#SBATCH --nodelist=n23-64-512-aragog
#SBATCH --gpus=a100:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --array=1-2

# Memory diagnosis of one GAN step (diagnose_memory.py) for a2-upstream-newenv (task 1) and dense-eager-fp32 (task 2),
# with the settings of ../run.sh. Outputs go to diagnosis/<variant>/; the results of the experiment are not touched.
# Submit from the experiment directory: sbatch diagnosis/run.sh

set -exuo pipefail

export EXPERIMENT="arabidopsis-ablation-diagnosis"
export CODE_ROOT="${CODE_ROOT:-/mnt/shared/scratch/mmicik/private/Geneformer/simulation/GRouNdScale}"
HERE="$CODE_ROOT/publication/arabidopsis-ablation"
export CONFIG="$HERE/causal_gan.cfg"
source "$CODE_ROOT/scripts/common.sh"

GROUNDSCALE_IMAGE='docker://milos7250/groundscale@sha256:e5732466d05309290f846c1631222a29227cfbc1a6d129c4d520367cc0f49fe7'
export APPTAINER_BIND="${APPTAINER_BIND:+$APPTAINER_BIND,}$CODE_ROOT,$(readlink -f "$CODE_ROOT/upstream")"
export GROUNDSCALE_LOGLEVEL=INFO GROUNDSCALE_NO_TQDM=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export DATA_DIR="data" CAUSAL_GRAPH="data/causal_graph.pkl"

VARIANTS=(a2-upstream-newenv dense-eager-fp32)
VARIANT=${VARIANTS[$((${SLURM_ARRAY_TASK_ID:-1} - 1))]}
export OUT_DIR="diagnosis/${VARIANT}/run" DIAG_OUT="diagnosis/${VARIANT}" PROFILE_MODE="$VARIANT"
mkdir -p "$OUT_DIR"

case "$VARIANT" in
a2-upstream-newenv)
    PYTHONPATH="$HERE/a2-pydeps" apptainer exec --nv "$GROUNDSCALE_IMAGE" python "$HERE/diagnosis/diagnose_memory.py" \
        "$HERE/a2_launch.py" --template "$HERE/causal_gan_upstream.cfg" --experiment "$EXPERIMENT" --run "$VARIANT"
    ;;
dense-eager-fp32)
    COMPILE_MODULES=False TF32=0 GENERATOR_LAYERS=dense apptainer exec --nv "$GROUNDSCALE_IMAGE" python \
        "$HERE/diagnosis/diagnose_memory.py" \
        "$HERE/launch.py" --config "$CONFIG" --experiment "$EXPERIMENT" --run "$VARIANT"
    ;;
esac

# checkpoints and logs of the shortened run are not needed
rm -rf "${OUT_DIR:?}" "${OUT_DIR:?}_CC"
