.. code-block::

    .
    |-- .git
    |-- .gitattributes
    |-- .github
    |   `-- workflows
    |       |-- docker-build.yml
    |       `-- documentation.yaml
    |-- .gitignore
    |-- .gitmodules
    |-- Atkinson_Hyperlegible
    |-- Beeline
    |-- LICENSE
    |-- README.md
    |-- configs
    |   |-- causal_gan.cfg
    |   |-- conditional_gan.cfg
    |   `-- gan.cfg
    |-- data
    |   |-- generated
    |   |-- interim
    |   |-- processed
    |   |   |-- BoneMarrow
    |   |   `-- PBMC
    |   `-- raw
    |       |-- BoneMarrow
    |       |-- Homo_sapiens_TF.csv
    |       |-- Mus_musculus_TF.csv
    |       `-- PBMC
    |-- docker
    |   |-- Dockerfile
    |   |-- build-docker.sh
    |   |-- build-apptainer.sh
    |   `-- apptainer.def
    |-- docs
    |-- notebooks
    |-- conda-env-create.sh
    |-- environment.yml
    |-- requirements.txt
    |-- requirements2.txt
    |-- requirements-dev.txt
    |-- results
    |-- scDesign2
    |-- scGAN
    |-- scripts
    |   |-- benchmark.sh
    |   |-- common.sh
    |   |-- create-grn.sh
    |   |-- evaluate.sh
    |   |-- generate.sh
    |   |-- hyperopt.sh
    |   |-- preprocess.sh
    |   `-- train.sh
    |-- sparsim
    `-- src
