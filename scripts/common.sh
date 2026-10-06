CODE_ROOT="${CODE_ROOT:-/mnt/shared/scratch/mmicik/private/Geneformer/simulation/GRouNdScale}"
CONFIG="${CONFIG:-causal_gan.cfg}"
POSTGRES_DIR="${POSTGRES_DIR:-$PWD/postgres}"

 if [[ ! -d "$CODE_ROOT" ]]; then
    echo "Error: CODE_ROOT directory does not exist: $CODE_ROOT"
    exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "Error: Config file does not exist: $CONFIG"
    exit 1
fi

COLUMNS="$(stty size 2> /dev/null |cut -d' ' -f2)" || COLUMNS=${COLUMNS:-200}
MKTEMP=$(mktemp -d)
trap 'rm -rf "$MKTEMP"' EXIT
TMPDIR="${TMPDIR:-$MKTEMP}"
export COLUMNS TMPDIR

set -euo pipefail

echo "Running command on $HOSTNAME"

export OMP_NUM_THREADS=8
export TORCHINDUCTOR_COMPILE_THREADS=8

export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torchinductor_cache"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1 TORCHINDUCTOR_AUTOGRAD_CACHE=1 TRITON_CACHE_AUTOTUNING=1
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=0

export GROUNDSCALE_LOGLEVEL=${GROUNDSCALE_LOGLEVEL:-INFO}
export GROUNDSCALE_NO_TQDM=${GROUNDSCALE_NO_TQDM:-0}
export LOGLEVEL=${LOGLEVEL:-WARNING}

export PYTHON="apptainer exec --nv 'docker://milos7250/groundscale:latest' python"
python(){
    apptainer exec --nv 'docker://milos7250/groundscale:latest' python "$@"
}

if [[ -f "$POSTGRES_DIR/.env" ]]; then
    set -a
    source "$POSTGRES_DIR/.env"
fi
