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
sys.path.insert(0, str(HERE.parent / "upstream/src"))

logger = logging.getLogger(__name__)


def start_record_memory_history() -> None:
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not recording memory history")
        return

    logger.info("Starting CUDA memory history recording")
    torch.cuda.memory._record_memory_history(max_entries=100_000)


def export_memory_snapshot() -> None:
    if not torch.cuda.is_available():
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = HERE / f"{socket.gethostname()}_{timestamp}.pickle"
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
    host_name = socket.gethostname()
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = f"{host_name}_{timestamp}"

    # Construct the trace file.
    prof.export_chrome_trace(f"{file_prefix}.json.gz")

    # Construct the memory timeline file.
    prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")


def main(config, train):
    from custom_parser import get_configparser
    from factory import get_factory
    cfg_parser = get_configparser()
    cfg_parser.read(config)
    fac = get_factory(cfg_parser)
    fac.get_trainer()()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    os.chdir(HERE)  # Config data/checkpoint paths are relative to mem-profile2/.
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
        main(HERE / "causal_gan_upstream.cfg", train=True)
            # prof.step()
    finally:
        export_memory_snapshot()
        stop_record_memory_history()
