#!/bin/bash
#SBATCH --job-name=conda-env-create
#SBATCH --output=logs/conda-env-create/%A.out
#SBATCH --partition=short
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G

source ~/.bashrc # to ensure conda is available

set -euo pipefail

usage() {
    cat <<EOF
Usage: ./conda-env-create.sh [options]

Creates the 'groundscale' conda environment and installs dependencies.

Options:
  --cpu        Install the CPU build of torch (default: GPU build)
  --gpu        Install the CUDA build of torch (default)
  --dev        Install developer dependencies from requirements-dev.txt
  --no-dev     Skip developer dependencies (default)
  -h, --help   Show this help message

Run without options on an interactive terminal, the script asks which torch
build to install and whether to install developer dependencies. Passing the
flags above, or running without a terminal, creates the environment
non-interactively using the defaults for any option that is not given.
EOF
}

install_cpu=0
install_dev=0
torch_flag_given=0
dev_flag_given=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu) install_cpu=1; torch_flag_given=1; shift ;;
        --gpu) install_cpu=0; torch_flag_given=1; shift ;;
        --dev) install_dev=1; dev_flag_given=1; shift ;;
        --no-dev) install_dev=0; dev_flag_given=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

conda env create -f ./environment.yml
conda activate groundscale

# Dependencies need to be installed in two phases, as sparselinear needs to be installed after torch

# Ask which torch build to install, unless it was given on the command line
if [[ $torch_flag_given -eq 0 && -t 0 ]]; then
    read -p "Do you want to install the CPU version of torch? (y/N) " -n 1 -r
    echo    # move to a new line
    if [[ ${REPLY:-} =~ ^[Yy]$ ]]; then
        install_cpu=1
    fi
fi

if [[ $install_cpu -eq 1 ]]; then
    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu torch==2.14.0
else
    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu132 torch==2.14.0
fi
pip install -r requirements2.txt --no-build-isolation

patch -d "$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/arboreto" < ./arboreto.patch

# Ask about developer dependencies, unless it was given on the command line
if [[ $dev_flag_given -eq 0 && -t 0 ]]; then
    read -p "Do you want to install developer dependencies? (y/N) " -n 1 -r
    echo    # move to a new line
    if [[ ${REPLY:-} =~ ^[Yy]$ ]]; then
        install_dev=1
    fi
fi

if [[ $install_dev -eq 1 ]]; then
    pip install -r requirements-dev.txt
fi