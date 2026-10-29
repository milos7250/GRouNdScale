# AGENTS.md

## Setup

- Run commands from the repository root; `pytest.ini` exposes `src` on `PYTHONPATH` and config paths are generally relative.
- Use `./conda-env-create.sh` for the supported environment. It creates Python 3.11, installs PyTorch 2.9.1 CUDA 13 dependencies, installs `requirements2.txt` only after PyTorch, and applies `arboreto.patch`.
- Install `requirements-dev.txt` when running tests, lint, or documentation builds.

## Structure

- `src/main.py` is the config-driven CLI; `src/preprocessing`, `src/training`, `src/gans`, `src/evaluation`, and `src/perturbation` implement the pipeline.
- `configs/*.cfg` are INI-style examples. The CLI requires `--config` and can chain preprocessing, GRN creation, training, generation, evaluation, GRN benchmarking, and perturbation.
- `tests/resources/processed/` contains required checked-in fixtures, including `.h5ad` datasets and a causal graph.

## Verification

- Run the fast non-GPU suite with `pytest -m "not gpu and not long"`.
- Run a focused test with `pytest tests/test_loggers.py` or a specific node such as `pytest tests/training/test_trainers.py::TestGANTrainer::test_short`.
- GPU tests are selected by `gpu`; long tests by `long`. DDP tests require multiple CUDA devices and NCCL.
- Run `ruff check .`, `ruff format --check .`, and `pyright` for static checks. Ruff uses preview mode, a 120-character limit, and fix-on-check configuration.
- Build documentation with `sphinx-build docs _build`; CI uses this exact command.

## CLI And Outputs

- Use hyphenated option names from `src/custom_parser.py`, including `--create-grn`, `--benchmark-grn`, and `--optimize-hyperparameters`; older docs may show underscores.
- Each CLI run creates the configured output directory and copies an interpolated config there. Do not assume an omitted generation path is required: it defaults to `simulated.h5ad` under that output directory.
- Preprocessing must run before training and produces library-size-normalized data. Causal GAN training additionally requires the pickled causal graph and its pretrained controller unless configured otherwise.

## HPC And Containers

- some `scripts/{preprocess,create-grn,train,benchmark}.sh` are Slurm/Apptainer workflows, not local wrappers. Override their machine-specific defaults with `CODE_ROOT` and `CONFIG`.
- `scripts/common.sh` sets runtime/cache environment variables and optionally sources `$POSTGRES_DIR/.env`; hyperparameter optimization may therefore require the PostgreSQL job setup.
- Build containers with `./docker/build-docker.sh` or `./docker/build-apptainer.sh`. The container build depends on `arboreto.patch` and the two-phase PyTorch/sparse dependency installation.
- Optional study/benchmark repositories are Git submodules; initialize them with `git submodule update --init --recursive` when working on those workflows.

## Known Non-Issues

- The Optuna PostgreSQL password is committed on purpose: in the `storage` URL of run configs (e.g. under `publication/paper-arabidopsis`), in `tests/resources/configs/*hyperopt*.cfg` and in git history. The database is internal and holds nothing sensitive. Do not flag it, redact it, or keep files out of git because of it.
