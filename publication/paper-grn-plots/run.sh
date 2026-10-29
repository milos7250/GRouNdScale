#!/bin/bash
#SBATCH --job-name=paper-grn-plots
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=short
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G

# PR curve and top TF precision of the simulated PBMC-All dataset published with the GRouNdGAN paper, made by the GRN
# benchmark of the original implementation (upstream/src/main.py --benchmark_grn) in its own image:
#   1. prepare.py: the causal graph and the TF list from the downloaded ground truth GRN (original-grn.csv),
#   2. infer-grn.py: GRNBoost2 on the downloaded simulated cells (generated.h5ad), regulators restricted to the TF list,
#   3. --benchmark_grn: precision at k (k = 1..15) and the PR curve of the inferred GRN against the ground truth.
# Everything runs on CPUs. Submit from this directory: sbatch run.sh
#
# N_CELLS=10000 sbatch run.sh infers the GRN from a random subset of 10000 of the 100000 cells (results-10000-cells/),
# the number of cells of the GRouNdGAN and GRouNdScale benchmarks in the paper. An inferred GRN that already exists in
# the output directory is reused; delete it to run GRNBoost2 again.

set -euxo pipefail

CODE_ROOT="${CODE_ROOT:-/mnt/shared/scratch/mmicik/private/Geneformer/simulation/GRouNdScale}"
UPSTREAM="$(readlink -f "$CODE_ROOT/upstream")"
IMAGE="docker://yazdanz/groundgan:GRouNdGAN-toolkit"
N_CELLS="${N_CELLS:-}"
OUT_DIR="results${N_CELLS:+-${N_CELLS}-cells}"

if [[ ! -f generated.h5ad || ! -f original-grn.csv ]]; then
    echo "Submit from publication/paper-grn-plots, next to generated.h5ad and original-grn.csv" >&2
    exit 1
fi

MKTEMP="$(mktemp -d)"
trap 'rm -rf "$MKTEMP"' EXIT
export TMPDIR="$MKTEMP" NUMBA_CACHE_DIR="$MKTEMP/numba" MPLCONFIGDIR="$MKTEMP/matplotlib" PYTHONUNBUFFERED=1

# The container only sees the working directory: bind the repository, the real location of its upstream/ link and the
# temporary directory (dask's worker space).
export APPTAINER_BIND="${APPTAINER_BIND:+$APPTAINER_BIND,}$CODE_ROOT,$UPSTREAM,$MKTEMP"

toolkit() {
    apptainer exec "$IMAGE" python3.9 "$@"
}

# 1. Causal graph and TF list
toolkit prepare.py \
    --cells generated.h5ad \
    --grn original-grn.csv \
    --causal-graph data/causal_graph.pkl \
    --tfs data/TFs.csv

# 2. GRNBoost2 with the TF list
mkdir -p "$OUT_DIR"
if [[ -s "$OUT_DIR/simulated-grn.tsv" ]]; then
    echo "Reusing the inferred GRN $OUT_DIR/simulated-grn.tsv"
else
    toolkit infer-grn.py \
        --cells generated.h5ad \
        --tfs data/TFs.csv \
        --out "$OUT_DIR/simulated-grn.tsv" \
        --workers "${SLURM_CPUS_PER_TASK:-$(nproc)}" \
        ${N_CELLS:+--n-cells "$N_CELLS"}
fi

# 3. Benchmark of the inferred GRN. main.py builds the GAN factory before anything else, which only reads [Model] type,
# and the benchmark reads the gene names of [Data] train to map the indices of the causal graph to names.
# The plots load their font from Atkinson_Hyperlegible/ relative to the working directory, i.e. the upstream root.
ln -sfn "$UPSTREAM/Atkinson_Hyperlegible" Atkinson_Hyperlegible
cat > "$OUT_DIR/benchmark.cfg" << EOF
[EXPERIMENT]
output directory = $OUT_DIR

[Model]
type = causal GAN

[Data]
train = generated.h5ad
causal graph = data/causal_graph.pkl

[GRN Benchmarking]
grn to benchmark = $OUT_DIR/simulated-grn.tsv
ground truth save path = $OUT_DIR/
plots save path = $OUT_DIR/
compute precision at k = True
k = 15
compute pr = True
EOF

toolkit "$UPSTREAM/src/main.py" --config "$OUT_DIR/benchmark.cfg" --benchmark_grn 2>&1 | tee "$OUT_DIR/benchmark.log"

# The ground truth that the benchmark reconstructed from the causal graph is the downloaded one.
cmp <(sort original-grn.csv) <(sort "$OUT_DIR/ground truth GRN.csv")

# File names of the figures in the paper
mv "$OUT_DIR/PR_curve.png" "$OUT_DIR/PR-curve.png"
mv "$OUT_DIR/Top_TF_precision.png" "$OUT_DIR/top-TF-precision.png"

echo "Figures: $OUT_DIR/PR-curve.png $OUT_DIR/top-TF-precision.png"
