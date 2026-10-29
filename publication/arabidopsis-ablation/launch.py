#!/usr/bin/env python3
"""Runs ../profile-memory.py for one variant of arabidopsis-ablation that runs the current code, in the same process and
with the same command line arguments, after selecting the generator layers and the float32 matmul precision.

Environment, set by run.sh (COMPILE_MODULES and PROFILE_MODE are read by causal_gan.cfg and profile-memory.py):
  GENERATOR_LAYERS   sparse (SparseLinear, the current code) or dense (MaskedLinear of GRouNdGAN, see dense_generator.py)
  TF32               1: TF32 matmuls, as the current code sets with torch.set_float32_matmul_precision("high") at the
                     start of training; 0: full float32 matmuls ("highest"), as in GRouNdGAN, by turning that call into
                     a call with "highest"

The summary records the generator layers and the matmul precision that is in effect when the run ends.

    python launch.py --config ... --experiment ... --run ...
"""

from __future__ import annotations

import logging
import os
import runpy
import sys
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # publication/, for memory_tools

import memory_tools

logger = logging.getLogger("launch")

GENERATOR_LAYERS = os.environ["GENERATOR_LAYERS"]
TF32 = os.environ["TF32"]
if GENERATOR_LAYERS not in {"sparse", "dense"} or TF32 not in {"0", "1"}:
    raise SystemExit(f"Unknown GENERATOR_LAYERS={GENERATOR_LAYERS!r} or TF32={TF32!r}")

if GENERATOR_LAYERS == "dense":
    import dense_generator  # noqa: F401 - importing it applies the patch

if TF32 == "0":
    _set_float32_matmul_precision = torch.set_float32_matmul_precision

    def _full_float32_matmuls(precision: str) -> None:
        logger.info("Matmul precision %r requested, keeping full float32 matmuls ('highest')", precision)
        _set_float32_matmul_precision("highest")

    torch.set_float32_matmul_precision = _full_float32_matmuls  # type: ignore[assignment]
    _set_float32_matmul_precision("highest")

_write_results = memory_tools.write_results


def _write_results_with_variant(*args: Any, extra_settings: dict[str, Any] | None = None, **kwargs: Any) -> str:
    extra_settings = {
        **(extra_settings or {}),
        "generator_layers": GENERATOR_LAYERS,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "python": sys.version.split()[0],
    }
    return _write_results(*args, extra_settings=extra_settings, **kwargs)


memory_tools.write_results = _write_results_with_variant  # profile-memory.py imports it from the module when it runs


if __name__ == "__main__":
    sys.argv[0] = str(HERE.parent / "profile-memory.py")
    runpy.run_path(sys.argv[0], run_name="__main__")
