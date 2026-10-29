#!/bin/bash
#SBATCH --job-name=p2-compile-overhead
#SBATCH --output=logs/%A/%a.out
#SBATCH --error=logs/%A/%a.err
#SBATCH --nodes=1
#SBATCH --nodelist=n23-64-512-aragog,n23-64-512-buckbeak,n23-64-512-crookshanks,n23-64-512-dobby,n23-64-512-fawkes,n23-64-512-nagini,n23-64-1024-hedwig
#SBATCH --gpus=a100:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --array=1-12%6

# Where does the memory that nvidia-smi sees beyond torch's reserved memory come from in compiled runs, and does it
# limit the model size? Each task runs one compiled target-gene-num configuration,
# "<compile threads>:<free GiB>:<G>[:<options>]", where free GiB is the memory left free by a ballast process ("full"
# for no ballast) and options is a comma-separated list of: trim, to reset the CUDA stack limit after each warm-up step
# (see profile-lmem.py); aten, to autotune matrix multiplications over ATen (cuBLAS) only, without the Triton GEMM
# templates; expandable, for PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. Every task also logs the GPU memory of
# each process on the GPU every second to processes.csv.
#
# Submit from this directory: sbatch run.sh, or with other tasks:
#   TASKS="8:8:1000 8:8:2000:aten,trim ..." sbatch --array=1-2 run.sh  (not --export, which splits at commas)

set -exuo pipefail

export EXPERIMENT="compile-overhead"
export CODE_ROOT="${CODE_ROOT:-/mnt/shared/scratch/mmicik/private/Geneformer/simulation/GRouNdScale}"
export CONFIG="$CODE_ROOT/publication/causal_gan.cfg"
source "$CODE_ROOT/scripts/common.sh" # sets python(), $MKTEMP and TORCHINDUCTOR_COMPILE_THREADS=8
export APPTAINER_BIND="${APPTAINER_BIND:+$APPTAINER_BIND,}$CODE_ROOT"
export GROUNDSCALE_LOGLEVEL=DEBUG GROUNDSCALE_NO_TQDM=1 PYTHONUNBUFFERED=1

TASKS=(${TASKS:-8:8:1000 8:8:2000 8:8:3000 8:8:9000 8:8:10000 8:8:11000 8:8:1000:trim 8:8:2000:trim 8:8:3000:trim 8:8:9000:trim 8:8:10000:trim 8:8:11000:trim})
IFS=: read -r THREADS FREE G OPTIONS <<<"${TASKS[$((${SLURM_ARRAY_TASK_ID:-1} - 1))]}"

export TORCHINDUCTOR_COMPILE_THREADS="$THREADS"
export COMPILE_MODULES=True
export TRIM_STACK=$([[ ",${OPTIONS:-}," == *,trim,* ]] && echo 1 || echo 0)
if [[ ",${OPTIONS:-}," == *,aten,* ]]; then
    export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=ATEN
fi
if [[ ",${OPTIONS:-}," == *,expandable,* ]]; then
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi
VARIANT="threads${THREADS}-free${FREE}${OPTIONS:+-${OPTIONS//,/-}}"
export DATA_DIR="$CODE_ROOT/publication/target-gene-num/data/processed/${G}"
export CAUSAL_GRAPH="$DATA_DIR/causal_graph.pkl"
export OUT_DIR="results/${VARIANT}/g${G}/compiled"
mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR"/{summary.json,trace.json,nvidia-smi.csv,processes.csv,ballast.json}

BACKGROUND=()
on_exit() {
    status=$?
    ((${#BACKGROUND[@]})) && kill "${BACKGROUND[@]}" 2>/dev/null || true
    rm -rf "$MKTEMP"
    exit "$status"
}
trap on_exit EXIT

if [[ "$FREE" != "full" ]]; then
    python "$CODE_ROOT/publication/compile-overhead/ballast.py" --free-gib "$FREE" --ready-file "$OUT_DIR/ballast.json" &
    BACKGROUND+=($!)
    until [[ -s "$OUT_DIR/ballast.json" ]]; do
        kill -0 "${BACKGROUND[0]}" # stops the job if the ballast failed
        sleep 1
    done
    UUID=$(grep -oP '"uuid": "\K[^"]+' "$OUT_DIR/ballast.json")
else
    UUID=$(python -c "import torch; u = str(torch.cuda.get_device_properties(0).uuid); print(u if u.startswith('GPU-') else 'GPU-' + u)")
fi

# Memory of every process on the GPU, with its parent and command line, to tell the training from its workers
log_processes() {
    set +x # once a second, not worth tracing
    echo "timestamp,pid,ppid,used_memory_mib,command"
    while true; do
        now=$(date +%s.%N)
        nvidia-smi --id="$UUID" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits |
            while IFS=', ' read -r pid used; do
                [[ -n "$pid" ]] || continue
                echo "$now,$pid,$(ps -o ppid= -p "$pid" | tr -d ' '),$used,\"$(ps -o args= -p "$pid" | cut -c1-300 | tr -d '"')\""
            done
        sleep 1
    done
}
log_processes >"$OUT_DIR/processes.csv" &
BACKGROUND+=($!)

python "$CODE_ROOT/publication/compile-overhead/profile-lmem.py" \
    --config "$CONFIG" \
    --experiment "$EXPERIMENT" \
    --run "${VARIANT}-g${G}" || echo "profile-memory.py failed with status $?"

find "$OUT_DIR" -mindepth 1 -maxdepth 1 \
    ! -name summary.json ! -name trace.json ! -name nvidia-smi.csv ! -name processes.csv ! -name ballast.json \
    ! -name "$(basename "$CONFIG")" -exec rm -rf {} +
