# Shared job body of the publication experiments, sourced by the run.sh of each experiment.
#
# Before sourcing, run.sh sets EXPERIMENT and CODE_ROOT and exports CONFIG, and it must run from its own directory.
# After sourcing, run.sh sets DATA_DIR and CAUSAL_GRAPH for the run it was assigned and calls `run_profile <run name>`.
#
# The array task id enumerates every (run, mode) pair, two tasks per run: odd tasks run the compiled sparse model, even
# tasks the un-compiled one. Each task has the GPU to itself.

source "$CODE_ROOT/scripts/common.sh" # sets the container python() and $MKTEMP, and enables set -euo pipefail

# The container only sees the working directory, which is below CODE_ROOT, but not src/ and publication/ next to it.
export APPTAINER_BIND="${APPTAINER_BIND:+$APPTAINER_BIND,}$CODE_ROOT"

export GROUNDSCALE_LOGLEVEL=DEBUG
export GROUNDSCALE_NO_TQDM=1
export PYTHONUNBUFFERED=1

SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-999999999999}
SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-1}

MODES=(compiled eager)
RUN_INDEX=$(((SLURM_ARRAY_TASK_ID - 1) / 2)) # 0-based index of the run within the experiment
MODE=${MODES[$(((SLURM_ARRAY_TASK_ID - 1) % 2))]}

on_exit() {
    status=$?
    rm -rf "$MKTEMP"
    if ((status != 0)); then
        scancel -t PENDING "$SLURM_ARRAY_JOB_ID" || true
    fi
    exit "$status"
}
trap on_exit EXIT

# Runs and profiles one (run, mode). The run's inputs are DATA_DIR (train/validation/test.h5ad) and CAUSAL_GRAPH.
# The profiling script exits successfully when the model does not fit on the GPU and records status "oom".
run_profile() {
    local run="$1"

    case "$MODE" in
    compiled) export COMPILE_MODULES=True ;;
    eager) export COMPILE_MODULES=False ;;
    esac

    export OUT_DIR="results/${run}/${MODE}"
    mkdir -p "$OUT_DIR"
    rm -f "$OUT_DIR/summary.json" "$OUT_DIR/trace.json" "$OUT_DIR/nvidia-smi.csv"

    python "$CODE_ROOT/publication/profile-memory.py" \
        --config "$CONFIG" \
        --experiment "$EXPERIMENT" \
        --run "$run"

    # Checkpoints, UMAPs and TensorBoard logs of the 20 steps are not results. KEEP_OUTPUTS=1 keeps them.
    if [[ "${KEEP_OUTPUTS:-0}" != "1" ]]; then
        find "$OUT_DIR" -mindepth 1 -maxdepth 1 \
            ! -name summary.json ! -name trace.json ! -name nvidia-smi.csv ! -name "$(basename "$CONFIG")" \
            -exec rm -rf {} +
    fi
}
