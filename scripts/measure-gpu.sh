#!/bin/bash
#SBATCH --job-name=causalgan-size
#SBATCH --output=logs/causalgan-size/%A.out
#SBATCH --error=logs/causalgan-size/%A.err
#SBATCH --nodes=1
#SBATCH --nodelist=n23-64-512-aragog,n23-64-512-buckbeak,n23-64-512-crookshanks,n23-64-512-dobby,n23-64-512-fawkes,n23-64-512-nagini,n23-64-1024-hedwig,n24-64-384-angel,n24-64-384-anya,n24-64-384-darla,n24-64-384-drusilla,n24-64-384-lorne,n24-64-384-spike
#SBATCH --gpus=a100:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G

set -exuo pipefail

export CONFIG="${CONFIG:-causal_gan.cfg}"
export CODE_ROOT="${CODE_ROOT:-/mnt/shared/scratch/mmicik/private/Geneformer/simulation/GRouNdScale.worktrees/dedup}"
export GROUNDSCALE_LOGLEVEL=DEBUG
export GROUNDSCALE_NO_TQDM=1
export PYTHON="apptainer exec --nv 'docker://milos7250/groundscale:latest' python"
python(){
    apptainer exec --nv 'docker://milos7250/groundscale:latest' python "$@"
}
export RESULTS_DIR="results-measure-gpu"

PID=""
NEW_CONFIG=""
on_exit() {
    status=$?
    if [[ -n "$PID" ]]; then
        kill -SIGINT "$PID" 2>/dev/null || true
    fi
    if [[ -n "$NEW_CONFIG" ]]; then
        rm "$NEW_CONFIG"
    fi
    exit "$status"
}
trap on_exit EXIT

python "$CODE_ROOT/scripts/measure-gpu-usage.py" --output "$RESULTS_DIR/gpu-usage.csv" &
PID=$!

sed "s|maximum steps\s*=.*|maximum steps = 20|g" "$CONFIG" > "${CONFIG%.cfg}_measure.cfg"
sed -i "s|output directory\s*=.*|output directory = $RESULTS_DIR|g" "${CONFIG%.cfg}_measure.cfg"
export CONFIG="${CONFIG%.cfg}_measure.cfg"
export NEW_CONFIG="$CONFIG"
"$CODE_ROOT/scripts/train.sh"

kill -SIGINT "$PID"

python "$CODE_ROOT/scripts/count-parameters.py" --checkpoint "$RESULTS_DIR/checkpoints/step_20.pth" --out "$RESULTS_DIR/params-size.json"
