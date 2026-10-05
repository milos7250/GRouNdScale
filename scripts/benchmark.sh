#!/bin/bash
#SBATCH --job-name=causalgan-benchmark
#SBATCH --output=logs/causalgan-benchmark/%A.out
#SBATCH --error=logs/causalgan-benchmark/%A.err
#SBATCH --nodes=1
#SBATCH --nodelist=n23-64-512-aragog,n23-64-512-buckbeak,n23-64-512-crookshanks,n23-64-512-dobby,n23-64-512-fawkes,n23-64-512-nagini,n23-64-1024-hedwig,n24-64-384-angel,n24-64-384-anya,n24-64-384-darla,n24-64-384-drusilla,n24-64-384-lorne,n24-64-384-spike
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

set -euo pipefail
CODE_ROOT="${CODE_ROOT:-/mnt/shared/scratch/mmicik/private/Geneformer/simulation/GRouNdScale.worktrees/dedup}"
source "$CODE_ROOT/scripts/common.sh"

# Find original output directory from config file
TRIAL_PATH="$(sed -n 's/^output directory[[:space:]]*=[[:space:]]*//p' "$CONFIG")"

# Copy the config file to a unique temp file to avoid race conditions
CONFIG_COPY="$(mktemp -p "$TMPDIR" .config.XXXXXX.cfg)"
cp "$CONFIG" "$CONFIG_COPY"
CONFIG="$CONFIG_COPY"

# GRNBoost2 without TF list
OUT_DIR="$TRIAL_PATH/infer_grn_without_tfs"
mkdir -p "$OUT_DIR/generated"
cp "$TRIAL_PATH/generated/simulated.h5ad" "$OUT_DIR/generated/simulated.h5ad"

python "$CODE_ROOT/scripts/infer-grn.py" --cells "$OUT_DIR/generated/simulated.h5ad" --out "$OUT_DIR/generated/simulated-grn.csv"
sed -i -E "s|^(output directory)[[:space:]]*=.*|\1 = $OUT_DIR|" "$CONFIG"
python \
    "$CODE_ROOT/src/main.py" \
    --config "$CONFIG" \
    --benchmark-grn
sed -i -E "s|^(output directory)[[:space:]]*=.*|\1 = $TRIAL_PATH|" "$CONFIG"
cp "$OUT_DIR/generated/ground truth GRN.csv" "$TRIAL_PATH/generated/"

# GRNBoost2 with TF list
OUT_DIR="$TRIAL_PATH/infer_grn_with_tfs"
mkdir -p "$OUT_DIR/generated"
cp "$TRIAL_PATH/generated/simulated.h5ad" "$TRIAL_PATH/generated/ground truth GRN.csv" "$OUT_DIR/generated/"

echo "Symbol" > "$OUT_DIR/generated/ground truth TFs.csv"
tail -n+2 "$OUT_DIR/generated/ground truth GRN.csv" |cut -d',' -f1 |sort -u >> "$OUT_DIR/generated/ground truth TFs.csv"

python "$CODE_ROOT/scripts/infer-grn.py" --cells "$OUT_DIR/generated/simulated.h5ad" --out "$OUT_DIR/generated/simulated-grn.csv" --tfs "$OUT_DIR/generated/ground truth TFs.csv"
sed -i -E "s|^(output directory)[[:space:]]*=.*|\1 = $OUT_DIR|" "$CONFIG"
python \
    "$CODE_ROOT/src/main.py" \
    --config "$CONFIG" \
    --benchmark-grn
sed -i -E "s|^(output directory)[[:space:]]*=.*|\1 = $TRIAL_PATH|" "$CONFIG"

# PIDC
OUT_DIR="$TRIAL_PATH/infer_grn_with_PIDC"
mkdir -p "$OUT_DIR/generated"
cp "$TRIAL_PATH/generated/simulated.h5ad" "$TRIAL_PATH/generated/ground truth GRN.csv" "$OUT_DIR/generated/"

apptainer exec --writable-tmpfs "$CODE_ROOT/PIDC/PIDC.sif" \
    "$CODE_ROOT/PIDC/run_PIDC.sh" \
    "$OUT_DIR/generated/simulated.h5ad" \
    "$OUT_DIR/generated/simulated-grn.csv"

sed -i -E "s|^(output directory)[[:space:]]*=.*|\1 = $OUT_DIR|" "$CONFIG"
python \
    "$CODE_ROOT/src/main.py" \
    --config "$CONFIG" \
    --benchmark-grn
sed -i -E "s|^(output directory)[[:space:]]*=.*|\1 = $TRIAL_PATH|" "$CONFIG"

rm "$CONFIG"

echo "Benchmarking completed."
