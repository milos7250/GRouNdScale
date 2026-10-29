#!/usr/bin/env python3
"""Occupies GPU memory so that only --free-gib GiB stay free, then holds it until killed.

This simulates a smaller GPU on a large one: every process started afterwards (the training, its compile workers and
their CUDA contexts) has to fit into what is left. The free memory is taken from cudaMemGetInfo after the ballast's own
CUDA context exists, so it is what other processes can still get. When the memory is held, a JSON file with the free
and total memory and the UUID of the GPU is written to --ready-file.
"""

import argparse
import json
import signal
from pathlib import Path

import torch

MIB = 2**20
CHUNK = 2**30


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--free-gib", type=float, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    args = parser.parse_args()

    torch.cuda.init()
    free, total = torch.cuda.mem_get_info()
    target = int(args.free_gib * 2**30)
    excess = (free - target) // (2 * MIB) * (2 * MIB)  # the caching allocator rounds large blocks to 2 MiB
    if excess < 0:
        raise SystemExit(f"Only {free / MIB:.0f} MiB are free, less than the requested {target / MIB:.0f} MiB")

    # Chunks of 1 GiB and the remainder merged into the last chunk: blocks below 10 MiB would get a 20 MiB segment
    sizes = [CHUNK] * (excess // CHUNK)
    remainder = excess - sum(sizes)
    if sizes:
        sizes[-1] += remainder
    elif remainder:
        sizes = [remainder]
    ballast = [torch.empty(size, dtype=torch.uint8, device="cuda") for size in sizes]
    torch.cuda.synchronize()

    free_after, _ = torch.cuda.mem_get_info()
    uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
    report = {
        "requested_free_mib": target / MIB,
        "free_mib": free_after / MIB,
        "total_mib": total / MIB,
        "ballast_mib": sum(sizes) / MIB,
        "uuid": uuid if uuid.startswith("GPU-") else f"GPU-{uuid}",
    }
    args.ready_file.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)

    signal.signal(signal.SIGTERM, lambda *_: raise_exit())
    while ballast:
        signal.pause()


def raise_exit() -> None:
    raise SystemExit(0)


if __name__ == "__main__":
    main()
