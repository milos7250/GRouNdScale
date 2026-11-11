#!/bin/bash
#SBATCH --job-name=causalgan-benchmark
#SBATCH --output=logs/causalgan-benchmark/%A.out
#SBATCH --error=logs/causalgan-benchmark/%A.err
#SBATCH --nodes=1
#SBATCH --nodelist=n23-64-512-aragog,n23-64-512-buckbeak,n23-64-512-crookshanks,n23-64-512-dobby,n23-64-512-fawkes,n23-64-512-nagini,n23-64-1024-hedwig,n24-64-384-angel,n24-64-384-anya,n24-64-384-darla,n24-64-384-drusilla,n24-64-384-lorne,n24-64-384-spike
#SBATCH --gpus=a100:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

CODE_ROOT="/mnt/shared/scratch/mmicik/private/Geneformer/simulation/GRouNdScale"
ORIG_CODE_ROOT="/mnt/shared/scratch/mmicik/private/Geneformer/simulation/upstream/GRouNdGAN"
CONFIG="causal_gan.cfg"
if [ ! -f "$CONFIG" ]; then
    echo "Error: config file '$CONFIG' not found" >&2
    exit 1
fi

TRIAL_PATH="$(sed -n 's/^output directory[[:space:]]*=[[:space:]]*//p' "$CONFIG")"

COLUMNS="$(stty size 2> /dev/null |cut -d' ' -f2)"
COLUMNS=${COLUMNS:-200}
export COLUMNS
TMPDIR=${TMPDIR:-/tmp}
export TMPDIR

set -euo pipefail

echo "Running command on $HOSTNAME"

export OMP_NUM_THREADS=8
export TORCHINDUCTOR_COMPILE_THREADS=8

export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/mmicik_torchinductor_cache"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1 TORCHINDUCTOR_AUTOGRAD_CACHE=1 TRITON_CACHE_AUTOTUNING=1
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=0

export GROUNDGAN_LOGLEVEL=INFO
export GROUNDGAN_NO_TQDM=0
export LOGLEVEL=WARNING

# Copy the config file to a unique temp file to avoid race conditions
CONFIG_COPY="$(mktemp -p "$TMPDIR" .config.XXXXXX.cfg)"
cp "$CONFIG" "$CONFIG_COPY"
CONFIG="$CONFIG_COPY"

# Transfer gene names from dataset to generated data
python "$ORIG_CODE_ROOT/transfer-gene-names.py" \
    --reference "$(sed -n 's/^[[:space:]]*train[[:space:]]*=[[:space:]]*//p' "$CONFIG")" \
    --target "$TRIAL_PATH/generated/simulated.h5ad"

# GRNBoost2 without TF list
OUT_DIR="$TRIAL_PATH/infer_grn_without_tfs"
mkdir -p "$OUT_DIR/generated"
cp "$TRIAL_PATH/generated/simulated.h5ad" "$OUT_DIR/generated/simulated.h5ad"

python "$CODE_ROOT/infer_grn.py" --cells "$OUT_DIR/generated/simulated.h5ad" --out "$OUT_DIR/generated/simulated-grn.csv"
sed -i 's/,/\t/g' "$OUT_DIR/generated/simulated-grn.csv"
sed -i -E "/^checkpoint/! s|$TRIAL_PATH|$OUT_DIR|g" "$CONFIG"
apptainer exec --nv docker://yazdanz/groundgan:GRouNdGAN-toolkit \
    python3.9 \
    "$ORIG_CODE_ROOT/src/main.py" \
    --config "$CONFIG" \
    --benchmark_grn
sed -i -E "/^checkpoint/! s|$OUT_DIR|$TRIAL_PATH|g" "$CONFIG"
cp "$OUT_DIR/generated/ground truth GRN.csv" "$TRIAL_PATH/generated/"

# GRNBoost2 with TF list
OUT_DIR="$TRIAL_PATH/infer_grn_with_tfs"
mkdir -p "$OUT_DIR/generated"
cp "$TRIAL_PATH/generated/simulated.h5ad" "$TRIAL_PATH/generated/ground truth GRN.csv" "$OUT_DIR/generated/"

echo "Symbol" > "$OUT_DIR/generated/ground truth TFs.csv"
tail -n+2 "$OUT_DIR/generated/ground truth GRN.csv" |cut -d',' -f1 |sort -u >> "$OUT_DIR/generated/ground truth TFs.csv"

python "$CODE_ROOT/infer_grn.py" --cells "$OUT_DIR/generated/simulated.h5ad" --out "$OUT_DIR/generated/simulated-grn.csv" --tfs "$OUT_DIR/generated/ground truth TFs.csv"
sed -i 's/,/\t/g' "$OUT_DIR/generated/simulated-grn.csv"
sed -i -E "/^checkpoint/! s|$TRIAL_PATH|$OUT_DIR|g" "$CONFIG"
apptainer exec --nv docker://yazdanz/groundgan:GRouNdGAN-toolkit \
    python3.9 \
    "$ORIG_CODE_ROOT/src/main.py" \
    --config "$CONFIG" \
    --benchmark_grn
sed -i -E "/^checkpoint/! s|$OUT_DIR|$TRIAL_PATH|g" "$CONFIG"

# PIDC
OUT_DIR="$TRIAL_PATH/infer_grn_with_PIDC"
mkdir -p "$OUT_DIR/generated"
cp "$TRIAL_PATH/generated/simulated.h5ad" "$TRIAL_PATH/generated/ground truth GRN.csv" "$OUT_DIR/generated/"

apptainer exec --writable-tmpfs "$CODE_ROOT/PIDC/PIDC.sif" \
    "$CODE_ROOT/PIDC/run_PIDC.sh" \
    "$OUT_DIR/generated/simulated.h5ad" \
    "$OUT_DIR/generated/simulated-grn.csv"
sed -i 's/,/\t/g' "$OUT_DIR/generated/simulated-grn.csv"
sed -i -E "/^checkpoint/! s|$TRIAL_PATH|$OUT_DIR|g" "$CONFIG"
apptainer exec --nv docker://yazdanz/groundgan:GRouNdGAN-toolkit \
    python3.9 \
    "$ORIG_CODE_ROOT/src/main.py" \
    --config "$CONFIG" \
    --benchmark_grn
sed -i -E "/^checkpoint/! s|$OUT_DIR|$TRIAL_PATH|g" "$CONFIG"

rm "$CONFIG"

echo "Benchmarking completed."
