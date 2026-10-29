#!/usr/bin/env python3
"""
Summarise causal controller and causal GAN training time, step time, steps and peak GPU memory of a run.

Time spans run from the ``Starting training loop`` line to the ``Step <MAX>: Computed Random Forest AUROC`` line of
each phase. Average step time is that span divided by the configured maximum steps. Peak memory is the maximum
``memory_used`` in ``gpu-usage.csv`` between the same two timestamps, reported in GiB.

Usage:
    python scripts/extract-run-stats.py <RUN_DIR>
"""

import csv
import json
import logging
import re
from datetime import datetime
from pathlib import Path

try:
    import rich_click as click
except ImportError:
    import click

logger = logging.getLogger(__name__)

TIMESTAMP_RE = re.compile(r"\[(\d{2}:\d{2}:\d{2})\]")
SECTION_RE = re.compile(r"^\s*\[(.+?)\]\s*$")
MAX_STEPS_RE = re.compile(r"^\s*maximum steps\s*=\s*([\d_]+)")
LOOP_RE = re.compile(r"Starting training loop")
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DAY = 86_400
MIB_PER_GIB = 1024


def fail(message: str) -> None:
    logger.error(message)
    raise SystemExit(1)


def time_of_day(stamp: str) -> int:
    hours, minutes, seconds = (int(part) for part in stamp.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def find_single(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        fail(f"Expected exactly one '{pattern}' in {directory}, found {len(matches)}")
    return matches[0]


def maximum_steps(cfg: Path) -> dict[str, int]:
    """Return the configured maximum steps of the ``[CC Training]`` and ``[Training]`` sections."""
    section = None
    steps: dict[str, int] = {}
    for line in cfg.read_text(errors="replace").splitlines():
        header = SECTION_RE.match(line)
        if header:
            section = header.group(1)
            continue
        found = MAX_STEPS_RE.match(line)
        if found and section in ("CC Training", "Training"):
            steps[section] = int(found.group(1).replace("_", ""))
    missing = {"CC Training", "Training"} - steps.keys()
    if missing:
        fail(f"Missing 'maximum steps' in {cfg} for sections: {sorted(missing)}")
    return steps


def parse_log(log: Path, cc_steps: int, gan_steps: int) -> dict:
    """Return the phase boundaries in the log's seconds-since-first-midnight frame, which handles midnight crossings."""
    cc_end_re = re.compile(rf"Step {cc_steps}: Computed Random Forest AUROC")
    gan_end_re = re.compile(rf"Step {gan_steps}: Computed Random Forest AUROC")
    loop_starts: list[int] = []
    cc_end: int | None = None
    gan_end: int | None = None
    first: int | None = None
    previous: int | None = None
    rollover = 0
    current: int | None = None

    for line in log.read_text(errors="replace").splitlines():
        found = TIMESTAMP_RE.search(line)
        if found:
            value = time_of_day(found.group(1))
            if first is None:
                first = value
            if previous is not None and previous - value > DAY / 2:
                rollover += DAY
            previous = value
            current = value + rollover
        if current is None:
            continue
        if LOOP_RE.search(line):
            loop_starts.append(current)
        elif len(loop_starts) == 1 and cc_end is None and cc_end_re.search(line):
            cc_end = current
        elif len(loop_starts) == 2 and gan_end is None and gan_end_re.search(line):
            gan_end = current

    if len(loop_starts) < 2 or cc_end is None or gan_end is None:
        fail(
            f"{log} does not contain a completed controller and generator phase "
            f"(training loops found: {len(loop_starts)}, controller end: {cc_end is not None}, "
            f"generator end: {gan_end is not None})"
        )
    return {
        "first": first,
        "cc_start": loop_starts[0],
        "cc_end": cc_end,
        "gan_start": loop_starts[1],
        "gan_end": gan_end,
    }


def peak_memory_gib(csv_path: Path, start: float, end: float) -> float:
    low, high = start, end
    peak: int | None = None
    with csv_path.open() as handle:
        for row in csv.DictReader(handle):
            moment = datetime.strptime(row["timestamp"], TIME_FORMAT).timestamp()
            if low <= moment <= high:
                used = int(row["memory_used"])
                peak = used if peak is None else max(peak, used)
    if peak is None:
        fail(f"No GPU samples in {csv_path} between the phase timestamps")
    return peak / MIB_PER_GIB


def summarise(run_dir: Path) -> dict:
    cfg = find_single(run_dir, "*.cfg")
    logs = sorted((run_dir / "logs/causalgan-train").rglob("*.out"), key=lambda path: path.stat().st_mtime)
    if not logs:
        fail(f"No .out files under {run_dir / 'logs'}")
    log = logs[-1]

    memory_csv = run_dir / "gpu-usage.csv"
    if not memory_csv.exists():
        memory_csv = run_dir / "results" / "gpu-usage.csv"
    if not memory_csv.exists():
        fail(f"No gpu-usage.csv in {run_dir} or {run_dir / 'results'}")

    logger.info(f"Config: {cfg}")
    logger.info(f"Log: {log}")
    logger.info(f"GPU memory: {memory_csv}")

    steps = maximum_steps(cfg)
    cc_steps = steps["CC Training"]
    gan_steps = steps["Training"]
    phases = parse_log(log, cc_steps, gan_steps)

    with memory_csv.open() as handle:
        first_row = next(csv.DictReader(handle))
    csv_start = datetime.strptime(first_row["timestamp"], TIME_FORMAT).timestamp()
    midnight = datetime.fromtimestamp(csv_start).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    if midnight + phases["first"] - csv_start > DAY / 2:
        midnight -= DAY
    elif midnight + phases["first"] - csv_start < -DAY / 2:
        midnight += DAY

    cc_seconds = phases["cc_end"] - phases["cc_start"]
    gan_seconds = phases["gan_end"] - phases["gan_start"]
    logger.info(f"Controller: {cc_seconds:.0f} s over {cc_steps} steps")
    logger.info(f"Generator: {gan_seconds:.0f} s over {gan_steps} steps")

    return {
        "cc_time": round(cc_seconds / 3600, 4),
        "cc_time_per_step": round(cc_seconds / cc_steps * 1000, 3),
        "cc_steps": cc_steps,
        "cc_gpu_memory": round(
            peak_memory_gib(memory_csv, midnight + phases["cc_start"], midnight + phases["cc_end"]), 3
        ),
        "gan_time": round(gan_seconds / 3600, 4),
        "gan_time_per_step": round(gan_seconds / gan_steps * 1000, 3),
        "gan_steps": gan_steps,
        "gan_gpu_memory": round(
            peak_memory_gib(memory_csv, midnight + phases["gan_start"], midnight + phases["gan_end"]), 3
        ),
    }


@click.command()
@click.argument(
    "run_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--output",
    default=None,
    help="Output JSON file path. Defaults to <RUN_DIR>/run-stats.json",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
)
def cli(run_dir: Path, output: Path | None):
    """
    Summarise training time, step time, steps and peak GPU memory of a run.

    RUN_DIR must contain the .cfg file, logs/ and gpu-usage.csv (or results/gpu-usage.csv).
    """
    output = output or run_dir / "run-stats.json"
    summary = summarise(run_dir)
    output.write_text(json.dumps(summary, indent=4) + "\n")
    logger.info(f"Saved summary to '{output}'")


if __name__ == "__main__":
    try:
        from rich.console import Console
        from rich.logging import RichHandler

        FORMAT = "%(message)s"
        logging.basicConfig(
            level="INFO",
            format=FORMAT,
            handlers=[RichHandler(console=Console(stderr=True), rich_tracebacks=True, tracebacks_show_locals=True)],
        )
    except ImportError:
        FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt="%H:%M:%S")

    cli()
