#!/usr/bin/env python3
"""Run causal GAN training while recording a PyTorch CUDA memory snapshot."""

import logging
import os
import socket
import sys
from datetime import datetime
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from main import main  # noqa: E402

logger = logging.getLogger(__name__)

if len(sys.argv) > 1:
    FILE_PREFIX = f"{sys.argv[1]}-genes"
else:
    hostname = socket.gethostname()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    FILE_PREFIX = f"{hostname}_{timestamp}.pickle"


def start_record_memory_history() -> None:
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not recording memory history")
        return

    logger.info("Starting CUDA memory history recording")
    torch.cuda.memory._record_memory_history(max_entries=500_000)


def export_memory_snapshot() -> None:
    if not torch.cuda.is_available():
        return
    
    snapshot_path = HERE / f"{FILE_PREFIX}.pickle"
    try:
        logger.info("Saving CUDA memory snapshot to %s", snapshot_path)
        torch.cuda.memory._dump_snapshot(str(snapshot_path), True)
    except Exception:
        logger.exception("Failed to capture CUDA memory snapshot")


def stop_record_memory_history() -> None:
    if torch.cuda.is_available():
        logger.info("Stopping CUDA memory history recording")
        torch.cuda.memory._record_memory_history(enabled=None)


TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"


def trace_handler(prof: torch.profiler.profile):
    # Prefix for file names.
    
    # Construct the trace file.
    prof.export_chrome_trace(str(HERE / f"{FILE_PREFIX}.json.gz"))

    # Construct the memory timeline file.
    prof.export_memory_timeline(str(HERE / f"{FILE_PREFIX}.html"), device="cuda:0")
    

def patch_gan_trainer():
    """Patch the CausalGANTrainer to record memory history during generator training."""
    from training.causal_gan import CausalGANTrainer

    _train_generator = CausalGANTrainer._train_generator

    def _train_generator_with_memory_tracking(self: CausalGANTrainer):
        """Wrap the generator forward pass to track memory usage."""
        if self.step > 2:
            start_record_memory_history()
        return _train_generator(self)

    CausalGANTrainer._train_generator = _train_generator_with_memory_tracking


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    os.chdir(HERE)  # Config data/checkpoint paths are relative to mem-profile2/.

    if "compile" in FILE_PREFIX:
        patch_gan_trainer()
    else:
        start_record_memory_history()
    try:
        # with torch.profiler.profile(
        #     activities=[
        #         torch.profiler.ProfilerActivity.CPU,
        #         torch.profiler.ProfilerActivity.CUDA,
        #     ],
        #     schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        #     record_shapes=True,
        #     profile_memory=True,
        #     with_stack=True,
        #     on_trace_ready=trace_handler,
        # ) as prof:
        main(HERE / "causal_gan.cfg", train=True)
            # prof.step()
    finally:
        export_memory_snapshot()
        stop_record_memory_history()
