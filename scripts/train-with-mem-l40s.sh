#!/bin/bash
#SBATCH --job-name=causalgan-train
#SBATCH --output=logs/causalgan-train/%A.out
#SBATCH --error=logs/causalgan-train/%A.err
#SBATCH --nodes=1
#SBATCH --nodelist=n24-64-384-angel,n24-64-384-anya,n23-64-512-aragog,n23-64-512-buckbeak,n23-64-512-crookshanks,n24-64-384-darla,n23-64-512-dobby,n24-64-384-drusilla,n23-64-512-fawkes,n23-64-1024-hedwig,n24-64-384-lorne,n23-64-512-nagini,n24-64-384-spike
#SBATCH --gpus=l40s:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G

set -euo pipefail
CODE_ROOT="${CODE_ROOT:-/mnt/shared/scratch/mmicik/private/Geneformer/simulation/GRouNdScale}"
source "$CODE_ROOT/scripts/common.sh"

export GROUNDSCALE_LOGLEVEL=DEBUG
export GROUNDSCALE_NO_TQDM=1

# Find original output directory from config file
RESULTS_DIR="$(sed -n 's/^output directory[[:space:]]*=[[:space:]]*//p' "$CONFIG")"
mkdir -p "$RESULTS_DIR"

PID=""
on_exit() {
    status=$?
    if [[ -n "$PID" ]]; then
        kill -SIGINT "$PID" 2>/dev/null || true
    fi
    exit "$status"
}
trap on_exit EXIT

python "$CODE_ROOT/scripts/measure-gpu-usage.py" --output "$RESULTS_DIR/gpu-usage.csv" &
PID=$!

"$CODE_ROOT/scripts/train.sh"

kill -SIGINT "$PID"

CHECKPOINT_PATH="$(find "$RESULTS_DIR/checkpoints/" -name 'step_*.pth' -printf '%T+ %p\n' |sort -r |head -n1 | cut -d ' ' -f2)"
python "$CODE_ROOT/scripts/count-parameters.py" --checkpoint "$CHECKPOINT_PATH" --out "$RESULTS_DIR/params-size.json"

rm -rf "$MKTEMP"
