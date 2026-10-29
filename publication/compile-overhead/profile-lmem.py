#!/usr/bin/env python3
"""profile-memory.py with the CUDA local-memory (stack) limit of the context recorded at every memory snapshot.

The driver grows the per-thread stack, and with it the local-memory pool of the context (stack size x the maximum
number of resident threads), whenever a kernel needs more local memory, and it never shrinks it on its own. That pool is
GPU memory that nvidia-smi sees and the caching allocator does not. With TRIM_STACK=1 the stack limit is set back to its
value before training after each of the steps 0..TRIM_UNTIL_STEP (default 4) of each phase, which releases the pool
grown by the kernels that autotuning benchmarked. Kernels that run afterwards grow it again as far as they need.

Arguments are those of profile-memory.py.
"""

import ctypes
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import memory_tools

CU_LIMIT_STACK_SIZE = 0
libcuda = ctypes.CDLL("libcuda.so.1")


def stack_size() -> int | None:
    value = ctypes.c_size_t()
    if libcuda.cuCtxGetLimit(ctypes.byref(value), CU_LIMIT_STACK_SIZE) != 0:  # no current context yet
        return None
    return value.value


def set_stack_size(value: int) -> int:
    return libcuda.cuCtxSetLimit(CU_LIMIT_STACK_SIZE, ctypes.c_size_t(value))


def install() -> None:
    tracker_class = memory_tools.Tracker
    trim = os.environ.get("TRIM_STACK", "0") == "1"
    trim_until = int(os.environ.get("TRIM_UNTIL_STEP", "4"))
    initial: dict[str, int] = {}

    snapshot = tracker_class.snapshot

    def snapshot_with_stack(self):
        record = snapshot(self)
        record["stack_size_bytes"] = stack_size()
        if "value" not in initial and record["stack_size_bytes"] is not None:
            initial["value"] = record["stack_size_bytes"]
        return record

    end_step = tracker_class.end_step

    def end_step_with_trim(self, failed):
        record, phase = self.current_step, self._step_phase
        end_step(self, failed)
        if trim and not failed and record["step"] <= trim_until and "value" in initial:
            memory_tools.torch.cuda.synchronize()
            before = self.snapshot()
            result = set_stack_size(initial["value"])
            record["stack_trim"] = {"result": result, "before": before, "after": self.snapshot()}
            memory_tools.logger.info(
                "%s: stack limit %s -> %s bytes (cuCtxSetLimit %d), nvidia-smi %d -> %d MiB",
                phase,
                before["stack_size_bytes"],
                record["stack_trim"]["after"]["stack_size_bytes"],
                result,
                before["device_used_mib"],
                record["stack_trim"]["after"]["device_used_mib"],
            )

    tracker_class.snapshot = snapshot_with_stack
    tracker_class.end_step = end_step_with_trim


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    install()
    spec = importlib.util.spec_from_file_location("profile_memory", HERE.parent / "profile-memory.py")
    assert spec is not None and spec.loader is not None
    profile_memory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile_memory)
    profile_memory.cli()
