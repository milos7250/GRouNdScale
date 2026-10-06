#!/usr/bin/env python3
"""Extract PyTorch CUDA memory timelines from `_dump_snapshot` pickles.

A snapshot pickle has three top-level keys that matter here:

* `device_traces` -- the event timeline. This is the only part carrying a time
  dimension, and therefore the only valid source for a memory-over-time plot.
* `segments` -- an end-of-run snapshot of each CUDA segment (block pool). Every
  field there (`total_size`, `allocated_size`, `active_size`, `blocks`, ...)
  describes the final state only. Plotting `segments[i]["total_size"]` against
  `i` therefore yields per-segment capacity indexed by segment, which is not a
  memory timeline and is not peak usage.
* `allocator_settings` -- notably `expandable_segments`, which changes which
  segment action names appear in the trace.

Two quantities are reconstructed by walking `device_traces`:

* allocated -- sum of live blocks. `+size` on `alloc`, `-size` on `free_requested`.
* reserved -- sum of live segments. `+size` on `segment_alloc` or `segment_map`,
  `-size` on `segment_free` or `segment_unmap`.

Both segment vocabularies occur in this repository: runs with
`expandable_segments=False` emit `segment_alloc`/`segment_free`, whereas
`expandable_segments=True` runs emit `segment_map`/`segment_unmap`. A reserved
walk that only knows `segment_alloc` reports zero for the latter.

Note that `_record_memory_history(max_entries=...)` writes into a fixed-size
ring buffer. A trace holding exactly `max_entries` events has wrapped and lost
its oldest entries, so the allocated series underflows below zero and the peak
is only a lower bound. Such snapshots are detected and reported as truncated.

Usage::

    python get-size-from-trace.py                        # every mem-profile2/*.pickle
    python get-size-from-trace.py mem-profile2/1000-genes.pickle
    python get-size-from-trace.py --out-dir mem-profile2 mem-profile2/*.pickle
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

import matplotlib

matplotlib.use("Agg")  # headless HPC nodes have no display

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

BYTES_PER_GIB = 1024**3
BYTES_PER_MIB = 1024**2

ALLOC_ACTION = "alloc"
FREE_ACTION = "free_requested"
SEGMENT_UP_ACTIONS = frozenset({"segment_alloc", "segment_map"})
SEGMENT_DOWN_ACTIONS = frozenset({"segment_free", "segment_unmap"})

# Matches `1000-genes.pickle`, `2000-genes-orig.pickle`, `3000-compile-genes.pickle`.
SNAPSHOT_NAME = re.compile(r"^(?P<genes>\d+)-(?P<kind>genes-orig|compile-genes|genes).*$")

VARIANT_NEW = "new"
VARIANT_ORIG = "orig"
VARIANT_COMPILE = "compile"
VARIANT_STYLE = {
    VARIANT_NEW: "tab:blue",
    VARIANT_ORIG: "tab:orange",
    VARIANT_COMPILE: "tab:green",
}

METRIC_TITLES = {"allocated": "Peak allocated", "reserved": "Peak reserved"}

# Where the per-run nvidia-smi samples live, relative to this script.
SMI_DIRNAME = "results-compile"
SMI_FILENAME = "gpu-usage.csv"


@dataclass(frozen=True)
class RunLabel:
    """Run identity inferred from a snapshot file name."""

    genes: int | None
    variant: str | None  # VARIANT_NEW, VARIANT_ORIG, VARIANT_COMPILE, or None


@dataclass
class Snapshot:
    """One reconstructed memory timeline."""

    path: Path
    label: RunLabel
    times_s: list[float]
    allocated: list[int]
    reserved: list[int]
    underflows: int
    expandable_segments: bool
    raw_events: int

    @property
    def truncated(self) -> bool:
        """True when the ring buffer wrapped, making the peak a lower bound.

        Detected from cumulative allocated going negative: a `free` was recorded
        whose matching `alloc` had already been evicted. Event count alone is not a
        usable signal, because `_record_memory_history(max_entries=N)` varies with
        the run -- the compile traces here hold >300k events with no underflow and
        are demonstrably complete, so comparing against a fixed cap would
        misreport them.
        """
        return self.underflows > 0

    @property
    def span_s(self) -> float:
        return self.times_s[-1] if self.times_s else 0.0

    def peak(self, metric: str) -> int:
        series = self.allocated if metric == "allocated" else self.reserved
        return max(series, default=0)


VARIANT_BY_KIND = {
    "genes": VARIANT_NEW,
    "genes-orig": VARIANT_ORIG,
    "compile-genes": VARIANT_COMPILE,
}


def parse_label(path: Path) -> RunLabel:
    """Infer gene count and implementation variant from a snapshot file name."""
    match = SNAPSHOT_NAME.match(path.stem)
    if match is None:
        return RunLabel(genes=None, variant=None)
    return RunLabel(genes=int(match["genes"]), variant=VARIANT_BY_KIND[match["kind"]])


_T = TypeVar("_T")


def cached(func: Callable[[Path], _T]) -> Callable[[Path], _T]:
    """
    Caching decorator to save function return values to disk and avoid reprocessing.
    Only works for functions taking a single Path argument that points to a file (not directories).

    The cache is stored in a `.cache` subdirectory of the input file's parent directory, with the cache file name
    derived from the input file's name and its MD5 hash. If the cache file exists, it is loaded and returned instead of
    calling the function. If the cache file does not exist or is invalid, the function is called, and its return value
    is saved to the cache file for future use.
    """

    @wraps(func)
    def wrapper(path: Path) -> _T:
        if not path.is_file():
            return func(path)

        with path.open("rb") as f:
            md5sum = hashlib.file_digest(f, "md5").hexdigest()
        cache_dir = path.parent / ".cache"

        cached_path = cache_dir / f"{path.stem}-{md5sum}.pickle"
        if cached_path.exists():
            try:
                with cached_path.open("rb") as f:
                    return pickle.load(f)
            except (OSError, pickle.UnpicklingError, EOFError):
                pass

        snapshot = func(path)
        cache_dir.mkdir(exist_ok=True)

        tmp_path = cached_path.with_name(cached_path.name + ".tmp")

        with tmp_path.open("wb") as f:
            pickle.dump(snapshot, f)
            f.flush()
            os.fsync(f.fileno())

        tmp_path.replace(cached_path)
        return snapshot

    return wrapper


@cached
def load_snapshot(path: Path) -> Snapshot:
    """Reconstruct the allocated and reserved timelines from one snapshot pickle."""
    with path.open("rb") as handle:
        raw: dict[str, Any] = pickle.load(handle)
        
    if isinstance(raw, Snapshot):
        return raw  # already processed and cached

    traces = raw.get("device_traces") or []
    if not traces:
        raise ValueError(f"{path}: snapshot carries no device_traces, so no timeline can be reconstructed")
    trace = traces[0]

    times_s: list[float] = []
    allocated: list[int] = []
    reserved: list[int] = []
    allocated_total = 0
    reserved_total = 0
    underflows = 0
    start_us = trace[0]["time_us"]

    for event in trace:
        action = event["action"]
        size = event["size"]
        if action == ALLOC_ACTION:
            allocated_total += size
        elif action == FREE_ACTION:
            allocated_total -= size
            if allocated_total < 0:
                underflows += 1
        elif action in SEGMENT_UP_ACTIONS:
            reserved_total += size
        elif action in SEGMENT_DOWN_ACTIONS:
            reserved_total -= size
        else:
            # `free_completed` marks a block returning to its segment's free list; it
            # changes neither the live byte totals nor any segment's capacity.
            continue
        times_s.append((event["time_us"] - start_us) / 1e6)
        allocated.append(allocated_total)
        reserved.append(reserved_total)

    settings = raw.get("allocator_settings") or {}
    return Snapshot(
        path=path,
        label=parse_label(path),
        times_s=times_s,
        allocated=allocated,
        reserved=reserved,
        underflows=underflows,
        expandable_segments=bool(settings.get("expandable_segments", False)),
        raw_events=len(trace),
    )


def _mark_peak(axes: Axes, snap: Snapshot, metric: str, colour: str) -> None:
    """Annotate the largest value of one series on a timeline plot."""
    values = [size / BYTES_PER_GIB for size in (snap.allocated if metric == "allocated" else snap.reserved)]
    if not values:
        return
    peak = max(values)
    index = values.index(peak)
    x, y = snap.times_s[index], peak
    axes.plot(x, y, "o", color=colour, markersize=5, zorder=5)
    # Anchor the label to the left of the marker for peaks near the right edge, so it
    # cannot run off the axes or slide under the legend.
    near_right_edge = index > 0.7 * len(snap.times_s)
    axes.annotate(
        f"{metric} peak {peak:.2f} GiB" + (" (lower bound)" if snap.truncated else ""),
        xy=(x, y),
        xytext=(-8, 8) if near_right_edge else (8, 8),
        textcoords="offset points",
        ha="right" if near_right_edge else "left",
        fontsize=8,
        color=colour,
    )


def plot_timeline(snapshots: list[Snapshot], out_path: Path, log_y: bool) -> None:
    """Plot allocated and reserved bytes against wall-clock time."""
    figure, axes = plt.subplots(figsize=(11, 5))
    for index, snap in enumerate(snapshots):
        shade = f"C{index}"
        axes.plot(snap.times_s, [size / BYTES_PER_GIB for size in snap.allocated], linewidth=1.1, color=shade)
        axes.plot(
            snap.times_s,
            [size / BYTES_PER_GIB for size in snap.reserved],
            linewidth=1.0,
            linestyle="--",
            color=shade,
        )
        _mark_peak(axes, snap, "allocated", shade)
        _mark_peak(axes, snap, "reserved", shade)

    axes.set_xlabel("Time since first traced event (s)")
    axes.set_ylabel("CUDA memory (GiB)")
    axes.set_title(_timeline_title(snapshots))
    axes.grid(alpha=0.3)
    axes.margins(y=0.15)
    if log_y:
        axes.set_yscale("log")

    handles = [
        plt.Line2D([], [], color="black", linewidth=1.1, label="allocated"),
        plt.Line2D([], [], color="black", linewidth=1.0, linestyle="--", label="reserved"),
    ]
    if len(snapshots) > 1:
        handles += [
            plt.Line2D([], [], color=f"C{index}", linewidth=1.1, label=snap.path.stem)
            for index, snap in enumerate(snapshots)
        ]
    # Kept outside the axes so peak annotations can never collide with it.
    axes.legend(handles=handles, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    for index, snap in enumerate(snapshots):
        if snap.truncated:
            axes.annotate(
                f"{snap.path.stem}: trace truncated ({snap.raw_events} events, "
                f"{snap.underflows} underflows) - peak is a lower bound",
                xy=(0.01, -0.22 - 0.06 * index),
                xycoords="axes fraction",
                fontsize=8,
                color="tab:red",
            )
    figure.tight_layout()
    figure.savefig(out_path, dpi=600)
    figure.savefig(out_path.with_suffix(".pdf"), dpi=600)
    plt.close(figure)


def _timeline_title(snapshots: list[Snapshot]) -> str:
    if len(snapshots) == 1:
        snap = snapshots[0]
        suffix = " [TRUNCATED]" if snap.truncated else ""
        return f"{snap.path.stem}: CUDA memory timeline{suffix}"
    return "CUDA memory timelines"


def group_by_variant(snapshots: list[Snapshot]) -> dict[str, list[Snapshot]]:
    """Bucket snapshots by variant, dropping unrecognised file names."""
    grouped: dict[str, list[Snapshot]] = {variant: [] for variant in VARIANT_STYLE}
    for snap in snapshots:
        if snap.label.variant is None or snap.label.genes is None:
            continue
        grouped[snap.label.variant].append(snap)
    for runs in grouped.values():
        runs.sort(key=lambda item: item.label.genes or 0)
    return grouped


def read_smi_peak(result_dir: Path) -> tuple[int, int] | None:
    """Return `(max_used_MiB, total_MiB)` from a run's nvidia-smi samples.

    nvidia-smi reports whole-device usage, so these numbers sit above the caching
    allocator's reserved bytes by a fixed context plus inductor-arena cost.
    """
    csv_path = result_dir / SMI_FILENAME
    if not csv_path.exists():
        return None
    max_used = 0
    total = 0
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            max_used = max(max_used, int(row["memory_used"]))
            total = max(total, int(row["memory_total"]))
    return (max_used, total) if max_used else None


def collect_smi_peaks(root: Path) -> dict[int, int]:
    """Peak nvidia-smi usage in bytes, keyed by gene count."""
    peaks: dict[int, int] = {}
    base = root / SMI_DIRNAME
    if not base.is_dir():
        return peaks
    for result_dir in sorted(base.iterdir()):
        if not result_dir.is_dir() or not result_dir.name.isdigit():
            continue
        found = read_smi_peak(result_dir)
        if found is not None:
            peaks[int(result_dir.name)] = found[0] * BYTES_PER_MIB
    return peaks


def plot_peaks_vs_genes(snapshots: list[Snapshot], out_path: Path, log_y: bool) -> None:
    """Plot peak memory against gene count, one curve per implementation variant."""
    grouped = group_by_variant(snapshots)

    figure, axes_grid = plt.subplots(1, len(METRIC_TITLES), figsize=(6 * len(METRIC_TITLES), 5), sharex=True)
    axes_list = list(axes_grid) if len(METRIC_TITLES) > 1 else [axes_grid]  # type: ignore[list-item]

    for axes, metric in zip(axes_list, METRIC_TITLES, strict=True):
        for variant, colour in VARIANT_STYLE.items():
            runs = grouped[variant]
            if not runs:
                continue
            genes = [run.label.genes for run in runs]
            peaks = [run.peak(metric) / BYTES_PER_GIB for run in runs]
            axes.plot(genes, peaks, "-o", color=colour, label=variant)
            for x, y in zip(genes, peaks, strict=True):
                axes.annotate(
                    f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color=colour
                )
        axes.set_title(METRIC_TITLES[metric])
        axes.set_xlabel("Number of genes")
        axes.set_ylabel("Peak memory (GiB)")
        axes.grid(alpha=0.3)
        axes.margins(y=0.15)
        axes.set_xticks(
            sorted({run.label.genes for runs in grouped.values() for run in runs if run.label.genes is not None})
        )
        if log_y:
            axes.set_yscale("log")

    axes_list[0].legend(fontsize=9, loc="upper left")
    figure.suptitle("Peak CUDA memory vs number of genes (from memory-history snapshots)")
    figure.tight_layout()
    figure.savefig(out_path, dpi=600)
    figure.savefig(out_path.with_suffix(".pdf"), dpi=600)
    plt.close(figure)


def load_theory_model() -> tuple[list[int], Any, Any, Any, float]:
    """Import the analytic model from the repository root.

    Returns `(gene counts, theoretical_gib, theory_curves, compute_smi, smi overhead)`.
    Imported lazily so the timeline and peaks-only paths still work if
    `gpu_memory_model.py` is absent.
    """
    root = Path(__file__).resolve().parent.parent / "theoretical-size"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from gpu_memory_model import SIZES, compute_smi, theoretical_gib, theory_curves
    except ImportError as error:  # pragma: no cover - depends on checkout layout
        raise SystemExit(f"cannot import gpu_memory_model from {root}: it holds the analytic memory model") from error
    try:
        from gpu_memory_model import SMI_OVERHEAD_BYTES
    except ImportError:  # pragma: no cover - older model without the constant
        SMI_OVERHEAD_BYTES = 0
    return sorted(SIZES), theoretical_gib, theory_curves, compute_smi, SMI_OVERHEAD_BYTES


def plot_measured_vs_theory(
    snapshots: list[Snapshot],
    out_path: Path,
    gpu_limit_gib: float | None,
    smi_peaks: dict[int, int],
) -> None:
    """Per variant: measured allocated/reserved peaks plus the analytic estimate.

    Colour encodes the variant and line style encodes the metric: solid line =
    analytic estimate, solid markers = measured allocated peak, dashed markers =
    measured reserved peak. Markers appear only where a snapshot exists, so the
    dense series stops at 2000 genes (OOM) while its model curve keeps climbing.

    Two further curves describe the compiled runs: whole-device peak from
    `nvidia-smi`, and that peak less the fixed context/arena overhead, which is the
    quantity the allocator figures should be compared against.
    """
    theory_sizes, _, theory_curves, compute_smi, smi_overhead = load_theory_model()
    grouped = group_by_variant(snapshots)
    figure, axes = plt.subplots(figsize=(11.5, 6.5))

    for variant, colour in VARIANT_STYLE.items():
        genes, curve = theory_curves(variant)
        axes.plot(genes, curve, "-", color=colour, linewidth=1.6, alpha=0.85, label=f"{variant} theoretical")

        runs = grouped[variant]
        if not runs:
            continue
        x = [run.label.genes for run in runs]
        for metric, style, suffix in (("allocated", "-o", "allocated"), ("reserved", "--s", "reserved")):
            values = [run.peak(metric) / BYTES_PER_GIB for run in runs]
            # No per-point labels here: with ~30 points they overlap into mush.
            # Exact values are in the stdout table and maxima-corrected.json.
            axes.plot(x, values, style, color=colour, markersize=6, linewidth=1.4, label=f"{variant} measured {suffix}")

    if smi_peaks:
        smi_genes = sorted(smi_peaks)
        smi_gib = [smi_peaks[g] / BYTES_PER_GIB for g in smi_genes]
        axes.plot(
            smi_genes,
            smi_gib,
            ":D",
            color="tab:red",
            markersize=4,
            linewidth=1.3,
            label="compile nvidia-smi peak",
        )
        axes.plot(
            smi_genes,
            [value - smi_overhead / BYTES_PER_GIB for value in smi_gib],
            ":D",
            color="tab:red",
            markersize=4,
            linewidth=1.3,
            alpha=0.45,
            label="  less fixed overhead",
        )
        model_smi = [(g, compute_smi(g) / BYTES_PER_GIB) for g in smi_genes if g in set(theory_sizes)]
        if model_smi:
            axes.plot(
                [g for g, _ in model_smi],
                [v for _, v in model_smi],
                "-",
                color="tab:red",
                linewidth=1.0,
                alpha=0.5,
                label="  model + overhead",
            )

    if gpu_limit_gib:
        axes.axhline(
            gpu_limit_gib, color="0.35", linestyle=":", linewidth=1.4, label=f"{gpu_limit_gib:.0f} GiB GPU limit"
        )

    axes.set_xlabel("Number of genes")
    axes.set_ylabel("Peak CUDA memory (GiB, log scale)")
    axes.set_title(
        "Measured peaks vs analytic estimate\nmarkers = measured from memory-history snapshots, lines = model"
    )
    axes.grid(alpha=0.3, which="both")
    axes.legend(fontsize=8, loc="upper left")

    axes.set_yscale("log")
    figure.tight_layout()
    figure.savefig(out_path.with_stem(f"{out_path.stem}-log"), dpi=600)
    figure.savefig(out_path.with_stem(f"{out_path.stem}-log").with_suffix(".pdf"), dpi=600)

    axes.set_yscale("linear")
    if gpu_limit_gib:
        axes.set_ylim(top=gpu_limit_gib * 1.05, bottom=0)
    figure.tight_layout()
    figure.savefig(out_path, dpi=600)
    figure.savefig(out_path.with_suffix(".pdf"), dpi=600)

    plt.close(figure)


def summarise(snapshots: list[Snapshot], smi_peaks: dict[int, int]) -> dict[str, Any]:
    """Build the JSON payload, mirroring maxima.json's `B`/`MiB` keys for easy diffing."""
    details = {
        snap.path.stem: {
            "source": snap.path.name,
            "genes": snap.label.genes,
            "variant": snap.label.variant,
            "expandable_segments": snap.expandable_segments,
            "truncated": snap.truncated,
            "peak_is_lower_bound": snap.truncated,
            "underflow_events": snap.underflows,
            "trace_events": snap.raw_events,
            "span_s": round(snap.span_s, 3),
            "peak_allocated_B": snap.peak("allocated"),
            "peak_reserved_B": snap.peak("reserved"),
            "final_allocated_B": snap.allocated[-1] if snap.allocated else 0,
        }
        for snap in snapshots
    }
    payload: dict[str, Any] = {
        "_note": (
            "Peaks are exact maxima over every traced event, not maxima over a "
            "downsampled series. Entries with peak_is_lower_bound=true come from a "
            "trace that reached max_entries and wrapped; their peaks are underestimates."
        ),
        "B": {name: info["peak_allocated_B"] for name, info in details.items()},
        "MiB": {name: info["peak_allocated_B"] / BYTES_PER_MIB for name, info in details.items()},
        "GiB": {name: info["peak_allocated_B"] / BYTES_PER_GIB for name, info in details.items()},
        "peak_reserved_B": {name: info["peak_reserved_B"] for name, info in details.items()},
        "peak_reserved_GiB": {name: info["peak_reserved_B"] / BYTES_PER_GIB for name, info in details.items()},
        "details": details,
    }
    if smi_peaks:
        payload["nvidia_smi_note"] = (
            "Whole-device peak per compile run, from gpu-usage.csv (MiB, 1-second samples). "
            "Exceeds the caching allocator's reserved bytes by a fixed context/arena cost."
        )
        payload["nvidia_smi_B"] = smi_peaks
        payload["nvidia_smi_GiB"] = {genes: value / BYTES_PER_GIB for genes, value in smi_peaks.items()}
    return payload


def print_table(snapshots: list[Snapshot], smi_peaks: dict[int, int] | None = None) -> None:
    smi_peaks = smi_peaks or {}
    header = (
        f"{'snapshot':22s} {'genes':>6s} {'variant':>8s} {'peak alloc':>12s} "
        f"{'peak resv':>12s} {'nvidia-smi':>12s} {'span':>8s}  flags"
    )
    print(header)
    print("-" * len(header))
    for snap in snapshots:
        flags = []
        if snap.truncated:
            flags.append(f"TRUNCATED ({snap.underflows} underflows)")
        if snap.expandable_segments:
            flags.append("expandable_segments")
        smi = smi_peaks.get(snap.label.genes or -1)
        print(
            f"{snap.path.stem:22s} "
            f"{snap.label.genes or '-':>6} "
            f"{snap.label.variant or '-':>8} "
            f"{snap.peak('allocated') / BYTES_PER_GIB:>9.3f} GiB "
            f"{snap.peak('reserved') / BYTES_PER_GIB:>9.3f} GiB "
            f"{(f'{smi / BYTES_PER_GIB:8.3f} GiB' if smi is not None else '          -')} "
            f"{snap.span_s:>7.2f}s  {' '.join(flags)}"
        )


def report_theory_agreement(snapshots: list[Snapshot], smi_peaks: dict[int, int]) -> None:
    """Print measured/theory ratios and flag runs that disagree with the model.

    A run whose peak sits far from the cohort's median ratio is usually not a
    measurement of the same configuration -- typically it terminated early, so its
    snapshot was dumped before the true peak. Those should not share a curve with
    the well-behaved runs.
    """
    try:
        _, theoretical_gib, _, _, smi_overhead = load_theory_model()
    except SystemExit as error:  # pragma: no cover - depends on checkout layout
        print(f"skipping model agreement check: {error}", file=sys.stderr)
        return

    for variant in VARIANT_STYLE:
        ratios: list[tuple[int, float]] = []
        for snap in group_by_variant(snapshots)[variant]:
            try:
                theory = theoretical_gib(snap.label.genes, variant)
            except KeyError:
                print(
                    f"  {snap.path.stem}: no SIZES entry for {snap.label.genes} genes, so the model cannot predict it"
                )
                continue
            if theory > 0:
                ratios.append((snap.label.genes, snap.peak("allocated") / BYTES_PER_GIB / theory))
        if len(ratios) < 3:
            continue
        ordered = sorted(value for _, value in ratios)
        median = ordered[len(ordered) // 2]
        print(f"\n{variant}: measured/theory ratio, median {median:.3f} over {len(ratios)} runs")
        outliers = [(genes, value) for genes, value in ratios if abs(value - median) > 0.05]
        for genes, value in sorted(outliers):
            print(f"  OUTLIER {genes:>6d} genes: ratio {value:.3f} (median {median:.3f}, delta {value - median:+.3f})")
        if outliers:
            print(
                f"  {len(outliers)} of {len(ratios)} runs disagree with the model; check they ran the same config to completion"
            )

    if not smi_peaks:
        return
    # Only compare against compile traces: the nvidia-smi samples come from the
    # compile runs, so pairing them with eager `new`/`orig` reserved would be
    # meaningless. (That mismatch is what made an earlier version report a large
    # negative "overhead".)
    compile_reserved = {
        snap.label.genes: snap.peak("reserved")
        for snap in group_by_variant(snapshots)[VARIANT_COMPILE]
        if snap.label.genes is not None
    }
    gaps = [
        (smi_peaks[genes] - compile_reserved[genes]) / BYTES_PER_GIB
        for genes in sorted(smi_peaks)
        if genes in compile_reserved
    ]
    if not gaps:
        print("\nnvidia-smi samples found, but no matching compile traces to compare against")
        return
    mean = sum(gaps) / len(gaps)
    spread = max(abs(value - mean) for value in gaps)
    print(
        f"\nnvidia-smi minus allocator reserved, over {len(gaps)} compile runs: "
        f"fixed cost {mean:.3f} +/- {spread:.3f} GiB "
        f"[model constant {smi_overhead / BYTES_PER_GIB:.3f} GiB]"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "snapshots",
        nargs="*",
        type=Path,
        help="snapshot pickles to read (default: every traces/*.pickle)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="where to write the figure and JSON (default: the traces directory's parent)",
    )
    parser.add_argument("--json-name", default="maxima-corrected.json", help="summary JSON file name")
    parser.add_argument("--log-y", action="store_true", help="use a logarithmic y-axis")
    parser.add_argument(
        "--with-theory",
        action="store_true",
        help="overlay the analytic estimate from gpu_memory_model.py: measured allocated, "
        "measured reserved and theoretical for each variant, plus the nvidia-smi peak "
        "for the compiled runs",
    )
    parser.add_argument(
        "--gpu-limit-gib",
        type=float,
        default=80.0,
        help="horizontal reference line in GiB for --with-theory (0 disables it)",
    )
    parser.add_argument(
        "--no-smi",
        action="store_true",
        help="ignore results-compile/*/gpu-usage.csv and omit the nvidia-smi curves",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    here = Path(__file__).resolve().parent
    paths = args.snapshots or sorted((here / "traces").glob("*.pickle"))
    if not paths:
        print("no snapshot pickles found", file=sys.stderr)
        return 1

    snapshots: list[Snapshot] = []
    for path in paths:
        try:
            snapshots.append(load_snapshot(path))
        except (OSError, ValueError, pickle.UnpicklingError) as error:
            print(f"skipping {path}: {error}", file=sys.stderr)
    if not snapshots:
        return 1

    smi_peaks = {} if args.no_smi else collect_smi_peaks(here)
    print_table(snapshots, smi_peaks)

    out_dir = args.out_dir or snapshots[0].path.parent.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    comparable = [snap for snap in snapshots if snap.label.variant is not None]
    limit = args.gpu_limit_gib or None
    if args.with_theory:
        figure_path = out_dir / "memory-vs-genes.png"
        plot_measured_vs_theory(snapshots, figure_path, limit, smi_peaks)
        report_theory_agreement(snapshots, smi_peaks)
        excluded = [snap.path.stem for snap in snapshots if snap not in comparable]
        if excluded:
            print(f"memory-vs-genes excludes {', '.join(excluded)} (unrecognised names)")
    elif len(comparable) >= 2:
        figure_path = out_dir / "peaks-vs-genes.png"
        plot_peaks_vs_genes(snapshots, figure_path, args.log_y)
        skipped = [snap.path.stem for snap in snapshots if snap not in comparable]
        if skipped:
            print(f"peaks-vs-genes excludes {', '.join(skipped)} (unrecognised names)")
    else:
        stem = snapshots[0].path.stem if len(snapshots) == 1 else "memory-timelines"
        figure_path = out_dir / f"{stem}-timeline.png"
        plot_timeline(snapshots, figure_path, args.log_y)

    json_path = out_dir / args.json_name
    json_path.write_text(json.dumps(summarise(snapshots, smi_peaks), indent=2) + "\n")
    print(f"\nwrote {figure_path}\nwrote {json_path}")
    return 0


# Wrap all locally defined functions with announce() to print a message before calling them
# def announce(func: Callable[..., Any]) -> Callable[..., Any]:
#     """Decorator to print a message before calling a function."""

#     @wraps(func)
#     def wrapper(*args: Any, **kwargs: Any):
#         print(f"=== {func.__name__} ===")
#         return func(*args, **kwargs)

#     return wrapper

# for name, obj in list(globals().items()):
#     if name in [
#         "main",
#         "load_snapshot",
#         "plot_timeline",
#         "plot_peaks_vs_genes",
#         "plot_measured_vs_theory",
#         "summarise",
#         "print_table",
#         "report_theory_agreement",
#         "collect_smi_peaks",
#         "read_smi_peak",
#         "group_by_variant",
#         "load_theory_model",
#         "parse_label",
#         "parse_args",
#     ]:
#         globals()[name] = announce(obj)

if __name__ == "__main__":
    raise SystemExit(main())
