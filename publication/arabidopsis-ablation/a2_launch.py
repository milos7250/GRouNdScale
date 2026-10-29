#!/usr/bin/env python3
"""Runs ../profile-memory-upstream.py for the variant a2-upstream-newenv of arabidopsis-ablation (the original code in
the groundscale image), in the same process and with the same command line arguments.

The only addition is an early `import torch._dynamo`, which loads Triton. The original code imports tensorflow at
start-up (a2-pydeps), and every torch.optim optimizer of torch 2.x imports torch._dynamo, hence Triton, when it is
created; loading Triton's native library after tensorflow's crashes with a segmentation fault, loading it first does
not. The original code and the profiler are unchanged.

    python a2_launch.py --template ... --experiment ... --run ...
"""

import runpy
import sys
from pathlib import Path

import torch._dynamo  # noqa: F401 - must be loaded before tensorflow, see above

PROFILER = Path(__file__).resolve().parent.parent / "profile-memory-upstream.py"

if __name__ == "__main__":
    sys.argv[0] = str(PROFILER)
    sys.path.insert(0, str(PROFILER.parent))  # memory_tools, as when the profiler runs as a script
    runpy.run_path(sys.argv[0], run_name="__main__")
