#!/bin/bash

source ~/.bashrc # to ensure conda is available

set -euo pipefail

conda env create -f ./environment.yml
conda activate groundscale

# Dependencies need to be installed in two phases, as sparselinear needs to be installed after torch

## Ask user if they want to install CPU or GPU version of torch, default to GPU
read -p "Do you want to install the CPU version of torch? (y/N) " -n 1 -r
echo    # move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu torch==2.14.0
else
    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu132 torch==2.14.0
fi
pip install -r requirements2.txt --no-build-isolation

patch -d "$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/arboreto" < ./arboreto.patch

# Ask user if they want to install developer dependencies, default to no
read -p "Do you want to install developer dependencies? (y/N) " -n 1 -r
echo    # move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install -r requirements-dev.txt
fi
