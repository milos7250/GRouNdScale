"""Measurement tools shared by the publication2 profiling scripts (profile-memory.py, profile-memory-upstream.py).

Torch memory is read from the CUDA caching allocator. Peaks are tracked in nested windows, so a phase-wide peak and the
peaks of every event, step and step stage can be taken from one allocator without resetting each other. nvidia-smi
memory is the memory.used of the GPU, sampled by a background thread and at every event boundary. It includes the CUDA
context and anything else on the GPU. Nothing here imports the code under test, so it can be used with either
implementation.

A run has two phases, "controller" and "generator", each measured from the start of the model initialization to the end
of its last training step.

The module has to run in Python 3.9 with torch 1.13, the environment of the original implementation.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import socket
import statistics
import subprocess
import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger("profile-memory")

MIB = 2**20
PHASES = ("controller", "generator")


def to_mib(value: float | None) -> float | None:
    return None if value is None else round(value / MIB, 3)


class SmiSampler(threading.Thread):
    """Samples the memory used on the GPU as reported by nvidia-smi every `interval` seconds."""

    def __init__(self, interval: float) -> None:
        super().__init__(daemon=True, name="nvidia-smi-sampler")
        self.interval = interval
        self.samples: list[tuple[float, int]] = []
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._command = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        try:
            uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
            self._command.append(f"--id={uuid if uuid.startswith('GPU-') else f'GPU-{uuid}'}")
        except Exception:
            logger.warning("Could not resolve the GPU UUID, nvidia-smi will report the first listed GPU", exc_info=True)

    def sample(self) -> tuple[float, int]:
        """Reads nvidia-smi once and returns the time of the reading and the used memory in MiB."""
        before = time.time()
        output = subprocess.run(self._command, capture_output=True, text=True, check=True).stdout
        moment = (before + time.time()) / 2
        used = int(output.strip().splitlines()[0])
        with self._lock:
            self.samples.append((moment, used))
        return moment, used

    def run(self) -> None:
        while not self._stop_event.wait(self.interval):
            try:
                self.sample()
            except Exception:
                logger.exception("nvidia-smi sampling failed")

    def stop(self) -> None:
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=10)

    def peak_between(self, start: float, end: float) -> int | None:
        with self._lock:
            values = [used for moment, used in self.samples if start <= moment <= end]
        return max(values) if values else None

    def write_csv(self, path: Path) -> None:
        with self._lock, path.open("w") as handle:
            handle.write("timestamp,memory_used_mib\n")
            handle.writelines(f"{moment:.3f},{used}\n" for moment, used in self.samples)


@dataclasses.dataclass(eq=False)
class Window:
    """Peak allocated and reserved bytes since the window was opened."""

    allocated: int
    reserved: int


class Tracker:
    def __init__(self, sampler: SmiSampler) -> None:
        self.sampler = sampler
        self.phases: dict[str, dict[str, Any]] = {}
        self.current_step: dict[str, Any] | None = None  # the step being measured, None outside of training steps
        self.busy = False  # True inside an event, so that nested calls of a measured method are not measured twice
        self._open: list[Window] = []
        self._phase_window: dict[str, Window] = {}
        self._active_phase: str | None = None
        self._step_phase = ""
        self._step_window: Window | None = None
        self._step_start = 0.0

    @property
    def active_phase(self) -> str | None:
        """The phase that is currently measured, None before the first model is built."""
        return self._active_phase

    def _fold(self) -> None:
        """Merges the allocator's peak into every open window."""
        allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        for window in self._open:
            window.allocated = max(window.allocated, allocated)
            window.reserved = max(window.reserved, reserved)

    def _open_window(self) -> Window:
        self._fold()
        torch.cuda.reset_peak_memory_stats()
        window = Window(torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
        self._open.append(window)
        return window

    def _close_window(self, window: Window) -> None:
        self._fold()
        self._open.remove(window)

    def snapshot(self) -> dict[str, float | int]:
        allocated, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
        moment, used = self.sampler.sample()
        return {"t": moment, "allocated_bytes": allocated, "reserved_bytes": reserved, "device_used_mib": used}

    def _record_end(self, phase: str, moment: float) -> None:
        window = self._phase_window[phase]
        self.phases[phase]["end"] = {
            "t": moment,
            "peak_allocated_bytes": window.allocated,
            "peak_reserved_bytes": window.reserved,
        }

    def begin_phase(self, phase: str) -> None:
        """Opens the window of a phase, which stays open until the next phase starts or the run ends."""
        if phase in self.phases:
            return
        if self._active_phase is not None:
            self._close_window(self._phase_window[self._active_phase])
        start = self.snapshot()
        self._phase_window[phase] = self._open_window()
        self.phases[phase] = {"start": start, "end": None, "events": [], "steps": []}
        self._active_phase = phase

    @contextmanager
    def event(self, phase: str, name: str) -> Generator[dict[str, Any]]:
        """Measures the memory before, after and at the peak of a block such as the model initialization."""
        self.begin_phase(phase)
        record: dict[str, Any] = {"event": name}
        before = self.snapshot()
        window = self._open_window()
        self.busy = True
        start = time.perf_counter()
        try:
            yield record
        finally:
            self.busy = False
            with suppress(Exception):
                torch.cuda.synchronize()
            record["duration_s"] = round(time.perf_counter() - start, 4)
            self._close_window(window)
            record.update(
                before=before,
                after=self.snapshot(),
                peak_allocated_bytes=window.allocated,
                peak_reserved_bytes=window.reserved,
            )
            self.phases[phase]["events"].append(record)
            if not self.phases[phase]["steps"]:
                self._record_end(phase, record["after"]["t"])

    @contextmanager
    def stage(self, name: str) -> Generator[None]:
        """Measures the peak of a part of the current training step, e.g. the generator update."""
        step = self.current_step
        assert step is not None
        baseline = torch.cuda.memory_allocated()
        window = self._open_window()
        try:
            yield
        finally:
            self._close_window(window)
            stage = step["stages"].setdefault(
                name,
                {"calls": 0, "baseline_allocated_bytes": baseline, "peak_allocated_bytes": 0, "peak_reserved_bytes": 0},
            )
            stage["calls"] += 1
            stage["peak_allocated_bytes"] = max(stage["peak_allocated_bytes"], window.allocated)
            stage["peak_reserved_bytes"] = max(stage["peak_reserved_bytes"], window.reserved)

    def begin_step(self, phase: str, step: int) -> None:
        """Starts measuring a training step. Its stages are measured until end_step."""
        self.begin_phase(phase)
        record: dict[str, Any] = {"step": step, "stages": {}}
        record["before"] = self.snapshot()
        torch.cuda.synchronize()
        self._step_phase, self._step_window = phase, self._open_window()
        self.current_step = record
        self._step_start = time.perf_counter()

    def end_step(self, failed: bool) -> None:
        """Ends the step started by begin_step and records its time and memory."""
        record, phase, window = self.current_step, self._step_phase, self._step_window
        assert record is not None
        if not failed:
            torch.cuda.synchronize()
        record["time_ms"] = round((time.perf_counter() - self._step_start) * 1000, 3)
        self.current_step = None
        self._close_window(window)
        after = self.snapshot()
        record.update(
            after=after,
            peak_allocated_bytes=window.allocated,
            peak_reserved_bytes=window.reserved,
            peak_nvidia_smi_mib=self.sampler.peak_between(record["before"]["t"], after["t"]),
            failed=failed,
        )
        self.phases[phase]["steps"].append(record)
        self._record_end(phase, after["t"])
        logger.info(
            "%s step %d%s: %.0f ms, torch peak allocated %.0f MiB, reserved %.0f MiB, nvidia-smi %d MiB",
            phase,
            record["step"],
            " FAILED" if failed else "",
            record["time_ms"],
            window.allocated / MIB,
            window.reserved / MIB,
            after["device_used_mib"],
        )

    def run_step(self, phase: str, step: int, run: Callable[[], Any]) -> Any:
        """Runs one training step and records its time and memory."""
        self.begin_step(phase, step)
        failed = True
        try:
            result = run()
            failed = False
            return result
        finally:
            self.end_step(failed)

    def finish(self) -> None:
        for phase in self.phases:
            end = self.phases[phase]["end"]
            if end is None:  # the phase started but nothing finished
                self._record_end(phase, self.snapshot()["t"])
        self._fold()
        for window in list(self._open):
            self._open.remove(window)


def module_sizes(modules: dict[str, torch.nn.Module]) -> dict[str, dict[str, int]]:
    return {
        name: {
            "parameters": sum(p.numel() for p in module.parameters()),
            "trainable_parameters": sum(p.numel() for p in module.parameters() if p.requires_grad),
            "parameter_bytes": sum(p.numel() * p.element_size() for p in module.parameters()),
            "buffer_bytes": sum(b.numel() * b.element_size() for b in module.buffers()),
        }
        for name, module in modules.items()
    }


def summarize_phase(tracker: Tracker, phase: str, max_steps: int, warmup_steps: int) -> dict[str, Any]:
    record = tracker.phases.get(phase)
    if record is None:
        return {"complete": False, "steps_recorded": 0}
    steps = record["steps"]
    steady = [s for s in steps if s["step"] >= warmup_steps and not s["failed"]]
    complete = bool(steps) and not steps[-1]["failed"] and steps[-1]["step"] == max_steps
    end = record["end"]
    stages = {
        name: to_mib(max(s["stages"][name]["peak_allocated_bytes"] for s in steady if name in s["stages"]))
        for name in sorted({name for s in steady for name in s["stages"]})
    }
    return {
        "complete": complete,
        "steps_recorded": len(steps),
        "average_step_time_ms": round(statistics.fmean(s["time_ms"] for s in steady), 3) if steady else None,
        "peak_torch_allocated_mib": to_mib(end["peak_allocated_bytes"]),
        "peak_torch_reserved_mib": to_mib(end["peak_reserved_bytes"]),
        "peak_nvidia_smi_mib": tracker.sampler.peak_between(record["start"]["t"], end["t"]),
        "steady_state": {
            "first_step": warmup_steps,
            "steps": len(steady),
            "peak_torch_allocated_mib": to_mib(max((s["peak_allocated_bytes"] for s in steady), default=None)),
            "peak_torch_reserved_mib": to_mib(max((s["peak_reserved_bytes"] for s in steady), default=None)),
            "stage_peak_torch_allocated_mib": stages,
        },
    }


def is_out_of_memory(error: BaseException | None) -> bool:
    while error is not None:
        if isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower():
            return True
        error = error.__cause__ or error.__context__
    return False


def git_state(repo: Path, paths: tuple[str, ...] = ("src",)) -> dict[str, Any]:
    """The commit of a repository and whether the given paths (by default src/) have uncommitted changes."""
    try:
        git = ["git", "-C", str(repo)]
        commit = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(
            [*git, "status", "--porcelain", "--", *paths], capture_output=True, text=True, check=True
        )
        return {"commit": commit, "src_modified": bool(dirty.stdout.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "src_modified": None}


def write_results(
    tracker: Tracker,
    sampler: SmiSampler,
    error: BaseException | None,
    *,
    experiment: str,
    run_name: str,
    mode: str,
    setup: dict[str, Any],
    warmup_steps: int,
    smi_interval: float,
    output_dir: Path,
    git: dict[str, Any],
    extra_settings: dict[str, Any] | None = None,
) -> str:
    """Summarizes the run, writes summary.json, trace.json and nvidia-smi.csv and returns the status of the run."""
    phases = {phase: summarize_phase(tracker, phase, setup["max_steps"][phase], warmup_steps) for phase in PHASES}
    if all(phases[phase]["complete"] for phase in PHASES):
        status = "ok"
    elif is_out_of_memory(error):
        status = "oom"
    else:
        status = "error"
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    summary = {
        "experiment": experiment,
        "run": run_name,
        "mode": mode,
        "status": status,
        "error": None if error is None else f"{type(error).__name__}: {str(error)[:500]}",
        "hyperparameters": setup["hyperparameters"],
        "graph": setup["graph"],
        **phases,
        "settings": {
            "units": "mebibytes (2**20 bytes), milliseconds",
            "warmup_steps": warmup_steps,
            "smi_interval_s": smi_interval,
            "max_steps": setup["max_steps"],
            "gpu": properties.name,
            "gpu_total_mib": round(properties.total_memory / MIB),
            "torch": torch.__version__,
            "host": socket.gethostname(),
            "git": git,
            **(extra_settings or {}),
        },
    }
    trace = {"phases": {phase: tracker.phases[phase] for phase in tracker.phases if phase in PHASES}}

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    sampler.write_csv(output_dir / "nvidia-smi.csv")
    logger.info("Run %s/%s (%s) finished with status %s, results in %s", experiment, run_name, mode, status, output_dir)
    return status
