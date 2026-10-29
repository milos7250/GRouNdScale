#!/usr/bin/env python3
"""Where the GPU memory of one GAN step goes, for comparing variants of arabidopsis-ablation (see README.md, Results).

Runs an entry point of the experiment (../launch.py or ../a2_launch.py) in the same process with its arguments, with the
CUDA caching allocator recording every allocation and its Python stack from the start. Both entry points measure steps
through memory_tools.Tracker, whose begin_step and end_step are wrapped here: at step DIAG_STEP (default 6) of the
generator phase it writes to DIAG_OUT and exits as soon as that step has ended.

- census.json      the live CUDA tensors before the step, by role: parameters with or without an optimizer, gradients,
                   optimizer states, other tensors reachable from Python (largest listed by shape and referrer), and
                   allocated memory not reachable from Python (e.g. saved tensors of autograd graphs)
- before.json      the blocks allocated before the step, by the code that allocated them
- peak.json        the blocks live at the peak of the step, by the code that allocated them (replay of the trace)
- snapshot.pickle  the allocator snapshot at the end of the step, for https://pytorch.org/memory_viz

    python diagnose_memory.py ../launch.py --config ... --experiment ... --run ...
"""

import gc
import json
import os
import pickle
import runpy
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))  # publication/, for memory_tools

import memory_tools

STEP = int(os.environ.get("DIAG_STEP", "6"))
OUT = Path(os.environ["DIAG_OUT"])
MIB = 2**20
REPOSITORY_CODE = ("/upstream/src/", "/GRouNdScale/src/", "/GRouNdGAN/src/")


def site(frames: list[dict[str, Any]]) -> str:
    """The innermost frame in repository code, and the innermost frame overall if that is somewhere else."""
    inner = frames[0] if frames else None
    for frame in frames:
        if any(part in frame["filename"] for part in REPOSITORY_CODE):
            where = f"{frame['filename'].split('/src/', 1)[-1]}:{frame['line']} {frame['name']}"
            if inner is not None and inner is not frame:
                where += f"  <- {Path(inner['filename']).name}:{inner['line']} {inner['name']}"
            return where
    return f"{Path(inner['filename']).name}:{inner['line']} {inner['name']}" if inner else "?"


def by_site(blocks: list[tuple[int, list]]) -> list[dict[str, Any]]:
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for size, frames in blocks:
        totals[site(frames)][0] += size
        totals[site(frames)][1] += 1
    return [
        {"site": key, "mib": round(size / MIB, 1), "blocks": count}
        for key, (size, count) in sorted(totals.items(), key=lambda item: -item[1][0])
        if size >= MIB
    ]


def is_a(obj: Any, cls: type) -> bool:
    try:
        return isinstance(obj, cls)
    except ReferenceError:  # a weakref proxy whose object is gone
        return False


def census() -> dict[str, Any]:
    labels: dict[int, str] = {}  # storage data_ptr -> role

    def label(tensor: Any, role: str) -> None:
        if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
            labels.setdefault(tensor.untyped_storage().data_ptr(), role)

    objects = gc.get_objects()
    for obj in objects:
        if is_a(obj, torch.optim.Optimizer):
            for group in obj.param_groups:
                for param in group["params"]:
                    label(param, "parameter (optimised)")
                    label(param.grad, "gradient")
                    for value in obj.state.get(param, {}).values():
                        label(value, "optimizer state")
    for obj in objects:
        if is_a(obj, torch.nn.Module):
            for param in obj.parameters(recurse=False):
                label(param, f"parameter (not optimised, {'trainable' if param.requires_grad else 'frozen'})")
                label(param.grad, "gradient (parameter not optimised)")
            for buffer in obj.buffers(recurse=False):
                label(buffer, "buffer")

    roles: dict[str, int] = defaultdict(int)
    others: dict[int, tuple[int, str]] = {}
    seen: set[int] = set()
    for obj in objects:
        if not is_a(obj, torch.Tensor) or not obj.is_cuda:
            continue
        storage = obj.untyped_storage()
        ptr, size = storage.data_ptr(), storage.nbytes()
        if ptr in seen:
            continue
        seen.add(ptr)
        role = labels.get(ptr)
        if role is None:
            referrers = sorted({type(ref).__name__ for ref in gc.get_referrers(obj) if ref is not objects})
            others[ptr] = (size, f"{tuple(obj.shape)} {obj.dtype} referred by {', '.join(referrers)}")
            role = "other tensor reachable from Python"
        roles[role] += size
    allocated = torch.cuda.memory_allocated()
    roles["allocated, not reachable from Python"] = allocated - sum(roles.values())
    return {
        "allocated_mib": round(allocated / MIB, 1),
        "roles_mib": {role: round(size / MIB, 1) for role, size in sorted(roles.items(), key=lambda item: -item[1])},
        "largest_other_tensors": [
            {"mib": round(size / MIB, 1), "tensor": text}
            for size, text in sorted(others.values(), key=lambda item: -item[0])[:25]
        ],
    }


def live_blocks(snapshot: dict[str, Any]) -> dict[int, tuple[int, list]]:
    blocks = {}
    for segment in snapshot["segments"]:
        address = segment["address"]
        for block in segment["blocks"]:
            if block["state"] == "active_allocated":
                blocks[address] = (block["requested_size"], block.get("frames", []))
            address += block["size"]
    return blocks


def write(name: str, data: Any) -> None:
    (OUT / name).write_text(json.dumps(data, indent=1))


_begin_step = memory_tools.Tracker.begin_step
_end_step = memory_tools.Tracker.end_step
_state: dict[str, Any] = {}


def begin_step(self: memory_tools.Tracker, phase: str, step: int) -> None:
    if phase == "generator" and step == STEP:
        torch.cuda.synchronize()
        OUT.mkdir(parents=True, exist_ok=True)
        write("census.json", census())
        before = torch.cuda.memory._snapshot()
        _state["live"] = live_blocks(before)
        _state["start_us"] = max((e["time_us"] for t in before["device_traces"] for e in t), default=0)
        write("before.json", by_site(list(_state["live"].values())))
    _begin_step(self, phase, step)


def end_step(self: memory_tools.Tracker, failed: bool) -> None:
    record = self.current_step
    _end_step(self, failed)
    if record is None or record["step"] != STEP or "live" not in _state:
        return
    snapshot = torch.cuda.memory._snapshot()
    with (OUT / "snapshot.pickle").open("wb") as handle:
        pickle.dump(snapshot, handle)
    # replay the allocations and frees of the step from the blocks that were live before it
    live = dict(_state["live"])
    total = peak = sum(size for size, _ in live.values())
    at_peak = dict(live)
    events = sorted(
        (e for trace in snapshot["device_traces"] for e in trace if e["time_us"] > _state["start_us"]),
        key=lambda e: e["time_us"],
    )
    for event in events:
        if event["action"] == "alloc":
            live[event["addr"]] = (event["size"], event.get("frames", []))
            total += event["size"]
        elif event["action"] == "free_requested" and event["addr"] in live:
            total -= live.pop(event["addr"])[0]
        if total > peak:
            peak, at_peak = total, dict(live)
    before_sites = {item["site"]: item["mib"] for item in by_site(list(_state["live"].values()))}
    write(
        "peak.json",
        {
            "events_replayed": len(events),
            "peak_mib": round(peak / MIB, 1),
            "recorded_peak_allocated_mib": round(record["peak_allocated_bytes"] / MIB, 1),
            "sites": [
                item | {"live_before_step_mib": before_sites.get(item["site"], 0.0)}
                for item in by_site(list(at_peak.values()))
            ],
        },
    )
    print(f"diagnosis written to {OUT}", flush=True)
    os._exit(0)


memory_tools.Tracker.begin_step = begin_step  # type: ignore[method-assign]
memory_tools.Tracker.end_step = end_step  # type: ignore[method-assign]

if __name__ == "__main__":
    torch.cuda.memory._record_memory_history(enabled="all", context="all", stacks="python", max_entries=2_000_000)
    entry = Path(sys.argv[1]).resolve()
    sys.argv = sys.argv[1:]
    sys.path.insert(0, str(entry.parent))
    runpy.run_path(str(entry), run_name="__main__")
