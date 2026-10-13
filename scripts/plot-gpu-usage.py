#!/usr/bin/env python3
"""
Live console plot of the GPU memory usage recorded by ``scripts/measure-gpu-usage.py``.

The CSV is tailed while the measurement script keeps appending to it, so the chart is
redrawn in place with ``plotext`` and can be left running next to a training job.  Only
complete lines are consumed, a restarting (truncated) file is detected, and several
files can be overlaid in a single chart.

Usage:
    python plot-gpu-usage.py publication/size-compare2/results2/1000/gpu-usage.csv
    python plot-gpu-usage.py -f --height 30 --window 600 results/seed_*/gpu-usage.csv
    python plot-gpu-usage.py --no-follow --unit percent gpu-usage.csv

Press q or Ctrl-C to stop.
"""

from __future__ import annotations

import csv
import logging
import math
import os
import sys
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import plotext as plt

try:
    import rich_click as click
except ImportError:
    import click

logger = logging.getLogger(__name__)

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# Header aliases, so a re-ordered or nvidia-smi flavoured header still resolves.
COLUMN_ALIASES = {
    "timestamp": ("timestamp", "time", "datetime"),
    "gpu_id": ("gpu_id", "gpuid", "gpu", "device", "id"),
    "memory_used": ("memory_used", "memory.used", "used"),
    "memory_total": ("memory_total", "memory.total", "total"),
}


@dataclass(frozen=True)
class Sample:
    """A single ``measure-gpu-usage.py`` record."""

    timestamp: datetime
    gpu_id: str
    memory_used: float
    memory_total: float


@dataclass
class GpuSeries:
    """Rolling window of the memory used by one GPU."""

    gpu_id: str
    times: deque[float] = field(default_factory=deque)  # elapsed seconds
    used: deque[float] = field(default_factory=deque)  # MiB
    total: deque[float] = field(default_factory=deque)  # MiB

    def current(self) -> float:
        return self.used[-1] if self.used else 0.0

    def peak(self) -> float:
        return max(self.used) if self.used else 0.0

    def mean(self) -> float:
        return sum(self.used) / len(self.used) if self.used else 0.0


class CsvTail:
    """Incremental CSV reader that only ever yields complete rows.

    The measurement script writes and flushes one row per sample, so a reader can
    legitimately catch a half-written line; the remainder is buffered until its
    terminating newline shows up.  Truncation is treated as "the run restarted".
    """

    def __init__(self, path: Path, max_samples: int):
        self.path = path
        self.max_samples = max_samples
        self.offset = 0
        self.pending = ""
        self.header: list[str] | None = None
        self.columns: dict[str, int] = {}
        self.warned = False

    def resolve_columns(self, header: Sequence[str], adopt: bool = True) -> None:
        """Map the required logical columns onto positions in ``header``."""
        lowered = {name.strip().lstrip("﻿").lower(): index for index, name in enumerate(header)}
        resolved: dict[str, int] = {}
        for key, aliases in COLUMN_ALIASES.items():
            resolved[key] = next((lowered[alias] for alias in aliases if alias in lowered), -1)
        if min(resolved.values()) < 0:
            missing = sorted(key for key, position in resolved.items() if position < 0)
            raise ValueError(f"Missing column(s) {', '.join(missing)}; found {', '.join(header)}")
        self.columns = resolved
        if adopt:
            self.header = list(header)

    def validate(self) -> None:
        """Parse the header eagerly so a malformed file fails fast with a clear message."""
        with self.path.open("r", newline="") as handle:
            first = handle.readline()
        row = next(csv.reader([first]), [])
        if not row:
            raise ValueError("the file is empty")
        # Check the columns but leave self.header unset, so that read_new() still
        # consumes the header line itself instead of parsing it as a data row.
        self.resolve_columns(row, adopt=False)

    def _warn_once(self, message: str, *args: object) -> None:
        if not self.warned:
            logger.warning(message, *args)
            self.warned = True

    def read_new(self) -> list[Sample]:
        """Return every row appended since the previous call."""
        try:
            size = self.path.stat().st_size
        except OSError:
            return []
        if size < self.offset:
            logger.info("'%s' was truncated, restarting from the beginning", self.path)
            self.offset = 0
            self.pending = ""
        if size == self.offset:
            return []

        with self.path.open("r", newline="") as handle:
            handle.seek(self.offset)
            chunk = handle.read()
            self.offset = handle.tell()

        lines = (self.pending + chunk).split("\n")
        self.pending = lines.pop()  # Keep a possibly incomplete trailing row.
        samples: list[Sample] = []
        for line in lines:
            row = next(csv.reader([line]), [])
            if not row or (len(row) == 1 and not row[0].strip()):
                continue
            if self.header is None:
                self.resolve_columns(row)
                continue
            sample = self._to_sample(row)
            if sample is not None:
                samples.append(sample)
        return samples

    def _to_sample(self, row: Sequence[str]) -> Sample | None:
        if len(row) <= max(self.columns.values()):
            self._warn_once("Skipping short row in '%s': %s", self.path, ",".join(row))
            return None
        columns = self.columns

        def value(key: str) -> str:
            return row[columns[key]].strip()

        try:
            timestamp = datetime.strptime(value("timestamp"), TIMESTAMP_FORMAT)
        except ValueError:
            self._warn_once("Skipping row with unparsable timestamp in '%s': %s", self.path, value("timestamp"))
            return None
        try:
            memory_used = float(value("memory_used"))
            memory_total = float(value("memory_total"))
        except ValueError:
            self._warn_once("Skipping row with non-numeric memory in '%s': %s", self.path, ",".join(row))
            return None
        return Sample(timestamp, value("gpu_id") or "?", memory_used, memory_total)


class UsageMonitor:
    """Accumulates tailed samples into per-GPU series on a shared time origin."""

    def __init__(self, tails: Sequence[CsvTail]):
        self.tails = list(tails)
        self.series: dict[str, GpuSeries] = {}
        self.origin: datetime | None = None
        self.total_samples = 0

    def poll(self) -> None:
        """Read everything appended since the previous call."""
        for tail in self.tails:
            for sample in tail.read_new():
                if self.origin is None:
                    self.origin = sample.timestamp
                entry = self.series.get(sample.gpu_id)
                if entry is None:
                    entry = self.series[sample.gpu_id] = GpuSeries(sample.gpu_id)
                elapsed = (sample.timestamp - self.origin).total_seconds()
                # Guard against a clock that steps backwards between rows.
                entry.times.append(max(elapsed, entry.times[-1] if entry.times else 0.0))
                entry.used.append(sample.memory_used)
                entry.total.append(sample.memory_total)
                while len(entry.used) > tail.max_samples:
                    entry.times.popleft()
                    entry.used.popleft()
                    entry.total.popleft()
                self.total_samples += 1


def nice_ceiling(value: float) -> float:
    """Round ``value`` up to a visually tidy axis maximum."""
    if value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    scaled = value / 10**exponent
    for step in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0):
        if scaled <= step:
            return step * 10**exponent
    return 10.0 ** (exponent + 1)


def format_value(value: float, unit: str) -> str:
    """Format an axis or readout value in the requested unit."""
    if unit == "percent":
        return f"{value:.0f}%"
    if unit == "gib":
        return f"{value / 1024.0:,.1f}"
    return f"{value:,.0f}"


def to_display(entry: GpuSeries, unit: str) -> tuple[list[float], list[float]]:
    """Convert a series' raw MiB samples into the unit used for plotting."""
    if unit == "percent":
        return list(entry.times), [
            100.0 * used / total if total else 0.0 for used, total in zip(entry.used, entry.total)
        ]
    return list(entry.times), list(entry.used)


def visible_samples(entry: GpuSeries, unit: str, window: float | None) -> tuple[list[float], list[float]]:
    """Return a series' times and values, restricted to the trailing ``window``."""
    times, values = to_display(entry, unit)
    if window is not None and times:
        cutoff = times[-1] - window
        start = next((index for index, stamp in enumerate(times) if stamp >= cutoff), 0)
        times, values = times[start:], values[start:]
    return times, values


def series_label(entry: GpuSeries, unit: str) -> str:
    """Build the legend entry summarising one GPU's current usage."""
    total = entry.total[-1] if entry.total else 0.0
    now = format_value(entry.current(), unit)
    # In percent mode the value is already a share, so the ratio would be redundant.
    share = "" if unit == "percent" else f" ({100.0 * entry.current() / total:,.1f}%)" if total else ""
    return f"GPU {entry.gpu_id} now {now}{share}"


def draw_frame(
    figure,
    series: dict[str, GpuSeries],
    unit: str,
    window: float | None,
    title: str,
    plot_size: tuple[int, int],
    colour_enabled: bool,
) -> None:
    """Render one chart with plotext, clearing only the data of the previous frame.

    Limits, labels and size survive across frames via ``clear.data()``, so the plot is
    rebuilt from scratch each time without wiping the terminal.
    """
    figure.clear.data()
    figure.plot_size(*plot_size)
    figure.label("memory used" if unit != "percent" else "memory used (%)", axis=1)
    figure.label("elapsed (s)", axis=0)
    figure.legend(True)

    visible: list[tuple[GpuSeries, list[float], list[float]]] = []
    for entry in (series[key] for key in sorted(series)):
        times, values = visible_samples(entry, unit, window)
        if values:
            visible.append((entry, times, values))

    if not visible:
        figure.title(f"{title} - waiting for samples ...")
        figure.ruler("x").lim(0, 1)
        figure.ruler("y").lim(0, 1)
        figure.show(colorless=not colour_enabled, flush=True)
        return

    span = max((times[-1] for _, times, _ in visible if times), default=0.0)
    # With --window the oldest visible sample is no longer the origin, so the axis
    # has to start there rather than at zero.
    origin = min((times[0] for _, times, _ in visible if times), default=0.0)
    if span <= origin:
        origin, span = 0.0, 1.0
    peak = max(max(values) for _, _, values in visible)
    maximum = 100.0 if unit == "percent" else nice_ceiling(peak)

    figure.ruler("x").lim(origin, span)
    figure.ruler("y").lim(0, maximum)
    figure.title(title)
    for entry, times, values in visible:
        signal = figure.signal(times, values).lines().label(series_label(entry, unit))
        figure.draw(signal)
    figure.show(colorless=not colour_enabled, flush=True)


@click.command()
@click.argument("csv_files", nargs=-1, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--follow/--no-follow",
    "-f/-F",
    default=True,
    help="Keep tailing the files and redraw; disable to render a single frame and exit.",
)
@click.option("--interval", default=0.5, show_default=True, type=float, help="Seconds between redraws.")
@click.option(
    "--height", "rows", default=None, type=int, help="Height of the plot in rows (defaults to the terminal height)."
)
@click.option(
    "--width",
    default=None,
    type=int,
    help="Plot width in columns (defaults to the terminal width).",
)
@click.option(
    "--window",
    default=None,
    type=float,
    help="Only show the last N seconds of samples; by default the full history is shown.",
)
@click.option(
    "--max-samples",
    default=100_000,
    show_default=True,
    type=int,
    help="Samples retained per GPU, bounding memory for long runs.",
)
@click.option(
    "--unit",
    default="mib",
    show_default=True,
    type=click.Choice(["mib", "gib", "percent"]),
    help="Unit for the memory axis.",
)
@click.option("--theme", default="default", show_default=True, help="plotext colour theme, e.g. 'clear' or 'pro'.")
@click.option("--no-color", is_flag=True, help="Disable ANSI colours.")
def cli(
    csv_files: tuple[Path, ...],
    follow: bool,
    interval: float,
    rows: int | None,
    width: int | None,
    window: float | None,
    max_samples: int,
    unit: str,
    theme: str,
    no_color: bool,
):
    """Live-plot GPU memory usage from CSV files written by measure-gpu-usage.py."""
    if not csv_files:
        raise click.ClickException("Provide at least one CSV file, e.g. 'python plot-gpu-usage.py gpu-usage.csv'.")
    if rows is not None and rows < 3:
        raise click.ClickException("--height must be at least 3.")
    if interval <= 0:
        raise click.ClickException("--interval must be positive.")
    if window is not None and window <= 0:
        raise click.ClickException("--window must be positive.")

    colour_enabled = not no_color and sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    tails = [CsvTail(path, max_samples) for path in csv_files]
    for tail in tails:
        try:
            tail.validate()
        except (OSError, ValueError) as error:
            raise click.ClickException(f"Cannot read '{tail.path}': {error}") from error
    monitor = UsageMonitor(tails)
    title = f"GPU memory usage ({', '.join(path.name for path in csv_files)})"
    figure = plt.figure
    figure.clear()
    figure.theme(theme)
    # Rows occupied by the frame on screen, so the next pass can erase exactly them.
    drawn_rows = 0
    stopped = False

    try:
        while True:
            monitor.poll()
            terminal_width, terminal_height = plt.terminal.size(update=True)
            # Keep two rows spare: writing a newline on the very last row scrolls the
            # screen, which would break the in-place redraw for good.
            plot_height = min(rows or terminal_height - 2, max(terminal_height - 2, 3))
            if drawn_rows:
                # Erase the previous frame so this one takes its place, no scrolling.
                plt.terminal.clean(drawn_rows)
            draw_frame(
                figure=figure,
                series=monitor.series,
                unit=unit,
                window=window,
                title=title,
                plot_size=(width or terminal_width, plot_height),
                colour_enabled=colour_enabled,
            )
            # A frame occupies exactly its plot height, which is what clean() needs.
            drawn_rows = plot_height
            if not follow:
                break
            plt.sleep(interval)
            if sys.stdin.isatty() and plt.terminal.is_pressed("q"):
                break
    except KeyboardInterrupt:
        stopped = True
    except BrokenPipeError:  # e.g. piped into `head`
        pass
    finally:
        # Erase the plot before anything else is written, so that the shell prompt
        # and any message land on the line the plot was started from.
        if follow and drawn_rows and sys.stdout.isatty():
            plt.terminal.clean(drawn_rows)
    if stopped:
        logger.info("Stopped by user.")


if __name__ == "__main__":
    try:
        from rich.logging import RichHandler

        logging.basicConfig(
            level="INFO",
            format="%(message)s",
            handlers=[RichHandler(rich_tracebacks=True, tracebacks_show_locals=True)],
        )
    except ImportError:
        logging.basicConfig(
            level="INFO", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
        )

    cli()
