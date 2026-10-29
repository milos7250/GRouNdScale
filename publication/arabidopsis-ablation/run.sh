#!/bin/bash
#SBATCH --job-name=p2-arabidopsis-ablation
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --nodes=1
#SBATCH --nodelist=n23-64-512-aragog
#SBATCH --gpus=a100:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --array=1-10%4

# Which change gives which speed-up on the paper's Arabidopsis configuration, see README.md. One task per variant, each
# measured once, all on one node (four tasks at a time, one GPU each).
# Submit from this directory: sbatch run.sh

set -exuo pipefail

export EXPERIMENT="arabidopsis-ablation"
export CODE_ROOT="${CODE_ROOT:-/mnt/shared/scratch/mmicik/private/Geneformer/simulation/GRouNdScale}"
export CONFIG="$CODE_ROOT/publication/arabidopsis-ablation/causal_gan.cfg"
UPSTREAM_CONFIG="$CODE_ROOT/publication/arabidopsis-ablation/causal_gan_upstream.cfg"
source "$CODE_ROOT/scripts/common.sh" # sets $MKTEMP and the cache settings, and enables set -euo pipefail

# The images are pinned by digest, so that every variant runs in the same environment.
GROUNDSCALE_IMAGE='docker://milos7250/groundscale@sha256:e5732466d05309290f846c1631222a29227cfbc1a6d129c4d520367cc0f49fe7'
UPSTREAM_IMAGE='docker://yazdanz/groundgan@sha256:6f219dff74b1c8cc920495f62cdf8aed706c33960bfdb047034642a695dab2e7'

# The container only sees the working directory, so the repository and the real location of its upstream/ link are
# bound. The data links in data/ resolve inside the repository.
export APPTAINER_BIND="${APPTAINER_BIND:+$APPTAINER_BIND,}$CODE_ROOT,$(readlink -f "$CODE_ROOT/upstream")"

export GROUNDSCALE_LOGLEVEL=DEBUG
export GROUNDSCALE_NO_TQDM=1
export PYTHONUNBUFFERED=1

SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-999999999999}
SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-1}

# a-upstream           the original implementation in its own image
# a2-upstream-newenv   the original implementation, unchanged, in the groundscale image (may fail)
# <layers>-<compile>-<matmuls>   the current code: dense (MaskedLinear) or sparse (SparseLinear) generator layers,
#                      un-compiled or compiled, full float32 or TF32 matmuls
VARIANTS=(
    a-upstream
    a2-upstream-newenv
    dense-eager-fp32
    dense-eager-tf32
    dense-compiled-fp32
    dense-compiled-tf32
    sparse-eager-fp32
    sparse-eager-tf32
    sparse-compiled-fp32
    sparse-compiled-tf32
)
# Variants that may not run at all; their failure is recorded and does not cancel the other tasks.
MAY_FAIL=(a2-upstream-newenv dense-compiled-fp32 dense-compiled-tf32)

if ((SLURM_ARRAY_TASK_ID < 1 || SLURM_ARRAY_TASK_ID > ${#VARIANTS[@]})); then
    echo "Array task id $SLURM_ARRAY_TASK_ID is outside 1-${#VARIANTS[@]}"
    exit 1
fi
VARIANT=${VARIANTS[$((SLURM_ARRAY_TASK_ID - 1))]}

# An error other than running out of memory cancels the tasks that have not started yet. The profiling scripts exit
# successfully when the model does not fit on the GPU and record status "oom".
on_exit() {
    status=$?
    rm -rf "$MKTEMP"
    if ((status != 0)); then
        scancel -t PENDING "$SLURM_ARRAY_JOB_ID" || true
    fi
    exit "$status"
}
trap on_exit EXIT

groundscale_python() {
    apptainer exec --nv "$GROUNDSCALE_IMAGE" python "$@"
}

upstream_python() {
    apptainer exec --nv "$UPSTREAM_IMAGE" python3.9 "$@"
}

export DATA_DIR="data"
export CAUSAL_GRAPH="$DATA_DIR/causal_graph.pkl"
export OUT_DIR="results/${VARIANT}"
export PROFILE_MODE="$VARIANT"

mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR/summary.json" "$OUT_DIR/trace.json" "$OUT_DIR/nvidia-smi.csv"

run_variant() {
    case "$VARIANT" in
    a-upstream)
        # anndata 0.8 of the original image cannot read uns/log1p/base (None); data-upstream/ is data/ without it
        export DATA_DIR="data-upstream"
        upstream_python "$CODE_ROOT/publication/profile-memory-upstream.py" \
            --template "$UPSTREAM_CONFIG" --experiment "$EXPERIMENT" --run "$VARIANT"
        ;;
    a2-upstream-newenv)
        # The image lacks tabulate and tensorflow (imported at start-up, not used in training): a2-pydeps holds them and
        # their missing dependencies, installed with --no-deps so that the image's numpy and h5py stay in use.
        # a2_launch.py loads Triton before tensorflow, which otherwise crashes the run (see there).
        PYTHONPATH="$CODE_ROOT/publication/arabidopsis-ablation/a2-pydeps" groundscale_python \
            "$CODE_ROOT/publication/arabidopsis-ablation/a2_launch.py" \
            --template "$UPSTREAM_CONFIG" --experiment "$EXPERIMENT" --run "$VARIANT"
        ;;
    *)
        local layers compile matmuls
        IFS=- read -r layers compile matmuls <<<"$VARIANT"
        case "$compile" in
        eager) export COMPILE_MODULES=False ;;
        compiled) export COMPILE_MODULES=True ;;
        esac
        case "$matmuls" in
        fp32) export TF32=0 ;;
        tf32) export TF32=1 ;;
        esac
        export GENERATOR_LAYERS="$layers"
        groundscale_python "$CODE_ROOT/publication/arabidopsis-ablation/launch.py" \
            --config "$CONFIG" --experiment "$EXPERIMENT" --run "$VARIANT"
        ;;
    esac
}

if [[ " ${MAY_FAIL[*]} " == *" $VARIANT "* ]]; then
    run_variant || echo "$VARIANT failed with exit status $?, see summary.json (if written) and the job log"
else
    run_variant
fi

# Checkpoints, UMAPs and TensorBoard logs of the 20 steps are not results, nor is the saved causal controller (in
# OUT_DIR/CC for the current code, in OUT_DIR_CC for the original one). KEEP_OUTPUTS=1 keeps them.
if [[ "${KEEP_OUTPUTS:-0}" != "1" ]]; then
    find "${OUT_DIR:?}" -mindepth 1 -maxdepth 1 \
        ! -name summary.json ! -name trace.json ! -name nvidia-smi.csv ! -name causal_gan.cfg \
        -exec rm -rf {} +
    rm -rf "${OUT_DIR:?}_CC"
fi

if [[ -f "$OUT_DIR/summary.json" ]] && grep -q '"status": "oom"' "$OUT_DIR/summary.json"; then
    echo "$VARIANT ran out of memory"
fi
