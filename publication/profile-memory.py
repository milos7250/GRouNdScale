#!/usr/bin/env python3
"""Train the causal GAN through main() and record the step time and GPU memory of the causal controller and of the
causal generator.

One call is one run: the causal controller is trained first, then the causal generator, exactly as `main.py --train`
does. The compilation of the modules (`[EXPERIMENT] compile modules`) selects the compiled or the un-compiled sparse
model. All settings come from the config and the environment, see causal_gan.cfg and common.sh.

Measurements. Both phases ("controller", "generator") are measured from the start of the model initialization to the
end of the last training step, which excludes the evaluation hooks that main() runs after the last step.
- Torch memory is read from the CUDA caching allocator. Peaks are tracked in nested windows, so a phase-wide peak and
  the peaks of every event, step and step stage can be taken from one allocator without resetting each other.
- nvidia-smi memory is the memory.used of the GPU, sampled by a background thread and at every event boundary. It
  includes the CUDA context and anything else on the GPU.
- Step time is the wall time of one training step (critic iterations, generator, labelers) with CUDA synchronized.

Outputs, written to `[EXPERIMENT] output directory`:
- summary.json     the hyperparameters and the requested measurements of the run
- trace.json       memory before/after the model and optimizer initialization and every training step
- nvidia-smi.csv   the raw nvidia-smi samples
"""

import functools
import logging
import os
import pickle
import statistics
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

try:
    import rich_click as click
except ImportError:
    import click

from memory_tools import SmiSampler, Tracker, git_state, module_sizes, write_results

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from custom_parser import get_configparser
from main import main

logger = logging.getLogger("profile-memory")

HYPERPARAMETER_DEFAULTS = {"BATCH_SIZE": "1024", "DEPTH": "2", "WIDTH": "1", "NOISE": "1"}
REQUIRED_ENVIRONMENT = ("OUT_DIR", "DATA_DIR", "CAUSAL_GRAPH", "COMPILE_MODULES")


def install_patches(tracker: Tracker) -> None:
    """Wraps the methods of the GAN and of the trainers that build the models, the optimizers and run the steps."""
    from gans import GAN, CausalGAN
    from training import CausalGANTrainer, GANTrainer

    def trainer_phase(trainer: GANTrainer) -> str:
        return "generator" if isinstance(trainer, CausalGANTrainer) else "controller"

    def gan_phase(gan: GAN) -> str:
        return "generator" if isinstance(gan, CausalGAN) else "controller"

    def patch_event(cls: type, method: str, name: str, phase_of: Callable[[Any], str]) -> None:
        original = getattr(cls, method)

        @functools.wraps(original)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if tracker.busy:  # e.g. CausalGANTrainer._init_optimizers calls GANTrainer._init_optimizers
                return original(self, *args, **kwargs)
            with tracker.event(phase_of(self), name):
                return original(self, *args, **kwargs)

        setattr(cls, method, wrapper)

    def patch_step(cls: type, phase: str) -> None:
        original = cls._training_step  # pyright: ignore[reportAttributeAccessIssue]

        @functools.wraps(original)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if tracker.current_step is not None:  # CausalGANTrainer._training_step calls GANTrainer._training_step
                return original(self, *args, **kwargs)
            return tracker.run_step(phase, self.step, lambda: original(self, *args, **kwargs))

        cls._training_step = wrapper  # pyright: ignore[reportAttributeAccessIssue]

    def patch_stage(cls: type, method: str, name: str) -> None:
        original = getattr(cls, method)

        @functools.wraps(original)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if tracker.current_step is None:
                return original(self, *args, **kwargs)
            with tracker.stage(name):
                return original(self, *args, **kwargs)

        setattr(cls, method, wrapper)

    gan_init = GAN.__init__

    @functools.wraps(gan_init)
    def gan_init_wrapper(self: GAN, *args: Any, **kwargs: Any) -> None:
        if type(self) is not GAN:  # the causal GAN calls GAN.__init__ from its own constructor
            return gan_init(self, *args, **kwargs)
        with tracker.event("controller", "model_init") as record:
            gan_init(self, *args, **kwargs)
            record["modules"] = module_sizes({"gen": self.gen, "crit": self.crit})

    causal_gan_init = CausalGAN.__init__

    @functools.wraps(causal_gan_init)
    def causal_gan_init_wrapper(self: CausalGAN, *args: Any, **kwargs: Any) -> None:
        with tracker.event("generator", "model_init") as record:
            causal_gan_init(self, *args, **kwargs)
            # gen contains the frozen causal controller, which is also listed on its own
            record["modules"] = module_sizes({
                "gen": self.gen,
                "crit": self.crit,
                "labeler": self.labeler,
                "antilabeler": self.antilabeler,
                "causal_controller": self.causal_controller,
            })

    GAN.__init__ = gan_init_wrapper  # pyright: ignore[reportAttributeAccessIssue]
    CausalGAN.__init__ = causal_gan_init_wrapper  # pyright: ignore[reportAttributeAccessIssue]

    patch_event(GANTrainer, "_init_optimizers", "optimizer_init", trainer_phase)
    patch_event(CausalGANTrainer, "_init_optimizers", "optimizer_init", trainer_phase)
    patch_event(GAN, "log_tensorboard_graph", "tensorboard_graph", gan_phase)  # only runs for un-compiled models

    patch_step(GANTrainer, "controller")
    patch_step(CausalGANTrainer, "generator")

    patch_stage(GANTrainer, "_train_critic", "critic")
    patch_stage(GANTrainer, "_train_generator", "generator")
    patch_stage(CausalGANTrainer, "_train_generator", "generator")
    patch_stage(CausalGANTrainer, "_train_labelers", "labelers")


def prepare_environment() -> None:
    missing = [name for name in REQUIRED_ENVIRONMENT if name not in os.environ]
    if missing:
        raise click.UsageError(f"Missing environment variables: {', '.join(missing)}. Run through run.sh.")
    for name, value in HYPERPARAMETER_DEFAULTS.items():
        os.environ.setdefault(name, value)
    if "NUM_GENES" not in os.environ:
        import anndata

        train = Path(os.environ["DATA_DIR"]) / "train.h5ad"
        os.environ["NUM_GENES"] = str(anndata.read_h5ad(train, backed="r").n_vars)


def read_hyperparameters(config: Path) -> dict[str, Any]:
    """B, d, w and n are configured; F, G and k are the properties of the causal graph."""
    cfg = get_configparser()
    cfg.read(config)
    graph_path = cfg.getpath("Data", "causal graph")
    with graph_path.open("rb") as handle:
        graph: dict[int, Any] = pickle.load(handle)
    regulators = [len(tfs) for tfs in graph.values()]
    return {
        "hyperparameters": {
            "B": cfg.getint("Training", "batch size"),
            "F": len(set().union(*graph.values())),
            "G": len(graph),
            "k": max(regulators),
            "d": cfg.getint("Model", "depth per gene"),
            "w": cfg.getint("Model", "width per gene"),
            "n": cfg.getint("Model", "noise per gene"),
        },
        "graph": {
            "path": str(graph_path),
            "total_genes": cfg.getint("Data", "number of genes"),
            "k_min": min(regulators),
            "k_mean": round(statistics.fmean(regulators), 4),
            "k_max": max(regulators),
        },
        "max_steps": {
            "controller": cfg.getint("CC Training", "maximum steps"),
            "generator": cfg.getint("Training", "maximum steps"),
        },
        "compiled": cfg.getboolean("EXPERIMENT", "compile modules"),
        "output_dir": cfg.getpath("EXPERIMENT", "output directory"),
    }


@click.command(context_settings={"show_default": True, "help_option_names": ["-h", "--help"]})
@click.option("--config", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--experiment", default="", help="Name of the experiment, stored in the summary.")
@click.option("--run", "run_name", required=True, help="Name of the run within the experiment, stored in the summary.")
@click.option(
    "--warmup-steps",
    type=click.IntRange(min=0),
    default=5,
    help="Steps before this one are excluded from the average step time and from the steady-state peaks.",
)
@click.option(
    "--smi-interval", type=click.FloatRange(min=0.05), default=0.2, help="Seconds between nvidia-smi samples."
)
def cli(config: Path, experiment: str, run_name: str, warmup_steps: int, smi_interval: float) -> None:
    """Profiles one causal GAN run, configured by CONFIG and the environment (see common.sh)."""
    if not torch.cuda.is_available():
        raise click.ClickException("CUDA is required to measure GPU memory")
    prepare_environment()
    setup = read_hyperparameters(config)
    output_dir: Path = setup["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    sampler = SmiSampler(smi_interval)
    tracker = Tracker(sampler)
    install_patches(tracker)
    sampler.start()

    error: BaseException | None = None
    try:
        main(config, train=True)
    except BaseException as exc:  # noqa: BLE001 - the summary is written for any failure, then the error is re-raised
        error = exc
    finally:
        tracker.finish()
        sampler.stop()

    status = write_results(
        tracker,
        sampler,
        error,
        experiment=experiment,
        run_name=run_name,
        mode=os.environ.get("PROFILE_MODE") or ("compiled" if setup["compiled"] else "eager"),
        setup=setup,
        warmup_steps=warmup_steps,
        smi_interval=smi_interval,
        output_dir=output_dir,
        git=git_state(ROOT),
    )

    if error is None:
        return
    if status == "ok":
        logger.warning("Training completed, but an error occurred afterwards: %r", error)
    elif status == "oom":
        logger.error("Ran out of GPU memory, the partial measurements are recorded: %r", error)
    else:
        raise error


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cli()
