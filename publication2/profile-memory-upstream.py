#!/usr/bin/env python3
"""Train the original implementation (upstream/) through its main.py and record the step time and GPU memory of the
causal controller and of the causal generator, in the format of profile-memory.py.

The original implementation has its own environment (Python 3.9, torch 1.13, the image
docker://yazdanz/groundgan:GRouNdGAN-toolkit), its own config format and no training step method: its trainers are the
`train` methods of the GAN classes, which run the loop inline. The measurement therefore wraps the training stages
(`_train_critic`, `_train_generator`, `_train_labelers`) and defines a step as everything from the first stage of a loop
iteration to the end of its last one, which is the same work as a step of the new implementation: the critic iterations,
the generator update and the labeler updates. The evaluation, logging and saving of the loop are not part of it.

main.py is run as `main.py --config <config> --train`. The config is the template with its placeholders filled in from
the environment (OUT_DIR, DATA_DIR, CAUSAL_GRAPH and the optional NUM_GENES, BATCH_SIZE, DEPTH, WIDTH, NOISE), and the
outputs are those of profile-memory.py: summary.json, trace.json and nvidia-smi.csv in OUT_DIR. The causal controller is
saved by main.py to OUT_DIR + "_CC".

This file has to run in Python 3.9.
"""

from __future__ import annotations

import argparse
import functools
import logging
import os
import pickle
import runpy
import statistics
import sys
from collections.abc import Callable
from pathlib import Path
from string import Template
from typing import Any

import torch
from memory_tools import SmiSampler, Tracker, git_state, module_sizes, write_results

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE.parent / "upstream"
sys.path.insert(0, str(UPSTREAM / "src"))

logger = logging.getLogger("profile-memory")

HYPERPARAMETER_DEFAULTS = {"BATCH_SIZE": "1024", "DEPTH": "2", "WIDTH": "1", "NOISE": "1"}
REQUIRED_ENVIRONMENT = ("OUT_DIR", "DATA_DIR", "CAUSAL_GRAPH")


def render_config(template: Path, output_dir: Path) -> Path:
    """Fills in the placeholders of the template. NUM_GENES defaults to the genes of the causal graph: TFs + targets."""
    missing = [name for name in REQUIRED_ENVIRONMENT if name not in os.environ]
    if missing:
        raise SystemExit("Missing environment variables: " + ", ".join(missing) + ". Run through run.sh.")
    for name, value in HYPERPARAMETER_DEFAULTS.items():
        os.environ.setdefault(name, value)
    if "NUM_GENES" not in os.environ:
        with open(os.environ["CAUSAL_GRAPH"], "rb") as handle:
            graph = pickle.load(handle)
        os.environ["NUM_GENES"] = str(len(graph) + len(set().union(*graph.values())))
    config = output_dir / "causal_gan.cfg"
    config.write_text(Template(template.read_text()).substitute(os.environ))
    return config


def read_setup(config: Path) -> dict[str, Any]:
    """B, d, w and n are configured; F, G and k are the properties of the causal graph."""
    from custom_parser import get_configparser

    cfg = get_configparser()
    cfg.read(config)
    with open(cfg.get("Data", "causal graph"), "rb") as handle:
        graph = pickle.load(handle)
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
            "path": cfg.get("Data", "causal graph"),
            "total_genes": cfg.getint("Data", "number of genes"),
            "k_min": min(regulators),
            "k_mean": round(statistics.fmean(regulators), 4),
            "k_max": max(regulators),
        },
        "max_steps": {
            "controller": cfg.getint("CC Training", "maximum steps"),
            "generator": cfg.getint("Training", "maximum steps"),
        },
        "labeler_interval": cfg.getfloat("Training", "labeler and antilabeler training intervals"),
    }


def install_patches(tracker: Tracker, labeler_interval: float) -> None:
    """Wraps the model construction, the optimizer construction and the training stages of the original GAN classes."""
    from gans.causal_gan import CausalGAN
    from gans.gan import GAN

    def phase_of(gan: GAN) -> str:
        return "generator" if isinstance(gan, CausalGAN) else "controller"

    def labelers_trained(gan: GAN) -> bool:
        # CausalGAN.train calls _train_labelers after the generator update when should_run(interval)
        return gan.step > 0 and gan.step % labeler_interval == 0

    def patch_stage(cls: type, method: str, stage: str, ends_step: Callable[[Any], bool]) -> None:
        original = getattr(cls, method)

        @functools.wraps(original)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if tracker.current_step is None:  # the first stage of a loop iteration opens the step
                tracker.begin_step(phase_of(self), self.step)
            failed = True
            try:
                with tracker.stage(stage):
                    result = original(self, *args, **kwargs)
                failed = False
                return result
            finally:
                if tracker.current_step is not None and (failed or ends_step(self)):
                    tracker.end_step(failed)

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
            record["modules"] = module_sizes({
                "gen": self.gen,  # contains the frozen causal controller, which is also listed on its own
                "crit": self.crit,
                "labeler": self.labeler,
                "antilabeler": self.antilabeler,
                "causal_controller": self.causal_controller,
            })

    adamw_init = torch.optim.AdamW.__init__

    @functools.wraps(adamw_init)
    def adamw_init_wrapper(self: Any, *args: Any, **kwargs: Any) -> None:
        phase = tracker.active_phase
        if phase is None or tracker.busy or tracker.current_step is not None:
            return adamw_init(self, *args, **kwargs)
        with tracker.event(phase, "optimizer_init"):  # recorded once per optimizer
            return adamw_init(self, *args, **kwargs)

    GAN.__init__ = gan_init_wrapper  # type: ignore[method-assign]
    CausalGAN.__init__ = causal_gan_init_wrapper  # type: ignore[method-assign]
    torch.optim.AdamW.__init__ = adamw_init_wrapper  # type: ignore[method-assign]

    patch_stage(GAN, "_train_critic", "critic", lambda gan: False)
    patch_stage(GAN, "_train_generator", "generator", lambda gan: True)  # the last stage of a controller step
    patch_stage(CausalGAN, "_train_generator", "generator", lambda gan: not labelers_trained(gan))
    patch_stage(CausalGAN, "_train_labelers", "labelers", lambda gan: True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--template", type=Path, required=True, help="config template with placeholders")
    parser.add_argument("--experiment", default="", help="name of the experiment, stored in the summary")
    parser.add_argument("--run", dest="run_name", required=True, help="name of the run, stored in the summary")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="steps before this one are excluded from the average step time and from the steady-state peaks",
    )
    parser.add_argument("--smi-interval", type=float, default=0.2, help="seconds between nvidia-smi samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to measure GPU memory")
    output_dir = Path(os.environ.get("OUT_DIR", ""))
    if not os.environ.get("OUT_DIR"):
        raise SystemExit("Missing environment variable OUT_DIR. Run through run.sh.")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = render_config(args.template, output_dir)
    setup = read_setup(config)

    sampler = SmiSampler(args.smi_interval)
    tracker = Tracker(sampler)
    install_patches(tracker, setup["labeler_interval"])
    sampler.start()

    error = None
    sys.argv = ["main.py", "--config", str(config), "--train"]
    try:
        runpy.run_path(str(UPSTREAM / "src" / "main.py"), run_name="__main__")
    except BaseException as exc:  # noqa: BLE001 - the summary is written for any failure, then the error is re-raised
        error = exc
    finally:
        tracker.finish()
        sampler.stop()

    status = write_results(
        tracker,
        sampler,
        error,
        experiment=args.experiment,
        run_name=args.run_name,
        mode="upstream",
        setup=setup,
        warmup_steps=args.warmup_steps,
        smi_interval=args.smi_interval,
        output_dir=output_dir,
        git=git_state(UPSTREAM),
        extra_settings={"implementation": "upstream", "python": sys.version.split()[0]},
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
    main()
