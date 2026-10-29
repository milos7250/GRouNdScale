#!/usr/bin/env python3
"""Plot the results of the publication experiments for the causal generator, with the analytical memory estimate.

Reads every `<experiment>/results/<run>/<mode>/summary.json` written by profile-memory.py. The hyperparameter that
differs between the runs of an experiment (k in tf-k, G in target-gene-num) is the x axis. For each of them this writes,
as PNG and PDF, two figures with every curve on one axis:

- `memory-vs-<x>`: peak memory of the compiled and of the un-compiled sparse model as torch allocated, torch reserved
  and nvidia-smi memory, and the analytical estimate of the peak (`compute_compile` and the sparse side of
  `compute_all` of theory/gpu_memory_model.py), which is the quantity to compare with torch
  allocated.
- `time-vs-<x>`: average step time of the compiled and of the un-compiled model. There is no analytical estimate of time.

The runs of `<experiment>-upstream` (the original, dense implementation) are drawn in the figures of `<experiment>` as a
third series, with the analytical estimate of the dense model (the dense side of `compute_all`). Every estimate is
evaluated at the hyperparameters of the runs.

A run that ran out of memory has no measurement of its own: its partial values are not plotted, and a cross at the top
marks the run. The torch peaks are those of the whole phase, from the model initialization to the last step; `--steady`
plots the peaks of the steps after the warm-up instead.

Usage::

    python plot-results.py                      # figures in publication/plots
    python plot-results.py --out-dir . --log-y
"""

import argparse
import importlib.util
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless HPC nodes have no display

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
THEORY_DIR = HERE / "theory"
THEORY_SOURCE = THEORY_DIR / "gpu_memory_model.py"  # the analytical model: compiled and eager sparse, and dense
MIB_PER_GIB = 1024

HYPERPARAMETERS = ("B", "F", "G", "k", "d", "w", "n")
X_LABELS = {
    "B": "Batch size (B)",
    "F": "Number of TFs (F)",
    "G": "Number of target genes (G)",
    "k": "TFs per target gene (k)",
    "d": "Depth of the target generators (d)",
    "w": "Width scale of the target generators (w)",
    "n": "Noise inputs per target gene (n)",
}

# Categorical slots 1 and 2 of the reference palette, validated with validate_palette.js (light mode)
# Slot 3 is below 3:1 contrast on the surface: the series are also labelled at their ends and the table is printed.
MODES = {
    "compiled": ("compiled", "#2a78d6"),
    "eager": ("un-compiled", "#eb6834"),
    "upstream": ("original (dense)", "#1baf7a"),
}
OOM_COLOUR = "#d03b3b"  # the status colour "critical"
INK, INK_SECONDARY, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

# Colour encodes the mode, line style and marker encode the metric
MEMORY_METRICS = (
    ("torch allocated", "peak_torch_allocated_mib", "-", "o"),
    ("torch reserved", "peak_torch_reserved_mib", "--", "s"),
    ("nvidia-smi", "peak_nvidia_smi_mib", ":", "D"),
)

plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": INK_MUTED,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "text.color": INK_SECONDARY,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

Run = dict[str, Any]
Estimate = Callable[[dict[str, int]], float]  # hyperparameters of a run -> peak in MiB


def load_summaries(root: Path) -> dict[str, list[Run]]:
    experiments: dict[str, list[Run]] = {}
    for path in sorted(root.glob("*/results/*/*/summary.json")):
        summary = json.loads(path.read_text())
        experiments.setdefault(summary["experiment"] or path.parts[-5], []).append(summary)
    return experiments


def load_theory() -> dict[str, Estimate]:
    """The analytical peak per mode, taken from theory/gpu_memory_model.py to keep one copy of it."""
    if not THEORY_SOURCE.exists():
        print(f"{THEORY_SOURCE} not found, no analytical estimate is drawn", file=sys.stderr)
        return {}
    spec = importlib.util.spec_from_file_location("gpu_memory_model", THEORY_SOURCE)
    assert spec is not None and spec.loader is not None
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)

    def arguments(h: dict[str, int]) -> tuple[int, ...]:
        return h["B"], h["F"], h["G"], h["k"], h["w"], h["d"], h["n"]

    return {
        "compiled": lambda h: model.compute_compile(*arguments(h)) / 2**20,
        "eager": lambda h: model.compute_all(*arguments(h))[1] / 2**20,
        "upstream": lambda h: model.compute_all(*arguments(h))[0] / 2**20,  # the dense original implementation
    }


def varying_hyperparameter(name: str, runs: list[Run]) -> str | None:
    varying = [h for h in HYPERPARAMETERS if len({run["hyperparameters"][h] for run in runs}) > 1]
    if len(varying) != 1:
        print(f"{name}: expected one varying hyperparameter, found {varying or 'none'}, skipped", file=sys.stderr)
        return None
    return varying[0]


def peak_value(phase: dict[str, Any], key: str, steady: bool) -> float | None:
    if not phase["complete"]:
        return None
    if steady and key != "peak_nvidia_smi_mib":  # nvidia-smi cannot be cut at a step, the allocator keeps its memory
        return phase["steady_state"][key]
    return phase[key]


def series(runs: list[Run], x: str, mode: str, value: Callable[[Run], float | None]):
    """Returns the points (x, y) of one mode with a measurement, and the x of the runs that have none."""
    chosen = sorted((run for run in runs if run["mode"] == mode), key=lambda run: run["hyperparameters"][x])
    points = [(run["hyperparameters"][x], value(run)) for run in chosen]
    return [(a, b) for a, b in points if b is not None], [a for a, b in points if b is None]


def style_axes(axes: Axes, xticks: list[int], log_y: bool, y_max: float) -> None:
    axes.grid(axis="y", color=GRID, linewidth=0.7)
    axes.set_axisbelow(True)
    axes.set_xticks(xticks)
    axes.tick_params(axis="x", labelsize=7)
    if max(len(str(tick)) for tick in xticks) > 2:
        plt.setp(axes.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    axes.margins(x=0.03)
    if log_y:
        axes.set_yscale("log")
        axes.set_ylim(top=y_max * 1.6)
    else:
        axes.set_ylim(0, y_max * 1.12)


def mark_out_of_memory(axes: Axes, xs: list[int]) -> None:
    """A cross on the top edge of the axes, whatever its scale, so it cannot be read as a measurement."""
    if xs:
        axes.plot(
            xs,
            [0.97] * len(xs),
            "x",
            color=OOM_COLOUR,
            markersize=6,
            markeredgewidth=1.6,
            transform=axes.get_xaxis_transform(),
            clip_on=False,
            zorder=5,
        )


def present_modes(runs: list[Run]) -> list[str]:
    return [mode for mode in MODES if any(run["mode"] == mode for run in runs)]


def label_end(axes: Axes, points: list[tuple[int, float]], mode: str) -> None:
    """Names the series at its last point, as a text in ink, so that it can be told apart without its colour."""
    axes.annotate(
        MODES[mode][0],
        points[-1],
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=7,
        color=INK_SECONDARY,
        annotation_clip=False,
    )


def pad_right(axes: Axes) -> None:
    low, high = axes.get_xlim()
    axes.set_xlim(low, high + 0.1 * (high - low))


def mode_handles(modes: list[str], any_oom: bool) -> list[Line2D]:
    handles = [Line2D([], [], color=MODES[mode][1], linewidth=2, label=MODES[mode][0]) for mode in modes]
    if any_oom:
        handles.append(
            Line2D(
                [],
                [],
                color=OOM_COLOUR,
                linestyle="",
                marker="x",
                markersize=6,
                markeredgewidth=1.6,
                label="ran out of memory",
            )
        )
    return handles


def title_for(what: str, runs: list[Run], x: str) -> tuple[str, str]:
    fixed = ", ".join(f"{h}={runs[0]['hyperparameters'][h]}" for h in HYPERPARAMETERS if h != x)
    return f"Causal generator {what} vs {X_LABELS[x]}", f"{runs[0]['experiment']}, fixed: {fixed}"


def save(figure: plt.Figure, path: Path) -> None:
    # Crop to the drawn content (titles, axes, legends, notes) so that LaTeX centres the graph, not empty margins
    figure.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    print(f"wrote {path.with_suffix('.png')}")


def estimate_curve(runs: list[Run], x: str, estimate: Estimate) -> list[tuple[int, float]]:
    """The analytical peak in GiB at every x of the experiment, whether the runs there ran out of memory or not."""
    first_at = {run["hyperparameters"][x]: run for run in runs}
    chosen = [first_at[value] for value in sorted(first_at)]
    return [(run["hyperparameters"][x], estimate(run["hyperparameters"]) / MIB_PER_GIB) for run in chosen]


def plot_memory(
    runs: list[Run],
    x: str,
    out_dir: Path,
    log_y: bool,
    steady: bool,
    limit_gib: float | None,
    theory: dict[str, Estimate],
) -> None:
    xticks = sorted({run["hyperparameters"][x] for run in runs})
    figure, axes = plt.subplots(figsize=(11, 5.8))
    y_max = limit_gib or 0.0
    oom: list[int] = []

    modes = present_modes(runs)
    for mode in modes:
        colour = MODES[mode][1]
        for _, key, style, marker in MEMORY_METRICS:

            def value(run: Run, key: str = key) -> float | None:
                peak = peak_value(run["generator"], key, steady)
                return None if peak is None else peak / MIB_PER_GIB

            points, failed = series(runs, x, mode, value)
            if points:
                axes.plot(*zip(*points, strict=True), style, marker=marker, color=colour, linewidth=1.6, markersize=4.5)
                y_max = max(y_max, max(y for _, y in points))
            if key == MEMORY_METRICS[0][1]:
                oom += failed
                mark_out_of_memory(axes, failed)
                if points:
                    label_end(axes, points, mode)
        if mode in theory:
            curve = estimate_curve(runs, x, theory[mode])
            axes.plot(*zip(*curve, strict=True), "-", color=colour, linewidth=3.2, alpha=0.35, zorder=1)

    style_axes(axes, xticks, log_y, y_max)
    pad_right(axes)
    if limit_gib:
        axes.axhline(limit_gib, color=INK_MUTED, linewidth=1.0)
        axes.annotate(
            f"GPU memory {limit_gib:g} GiB",
            (xticks[0], limit_gib),
            xytext=(0, 3),
            textcoords="offset points",
            fontsize=7,
            color=INK_SECONDARY,
        )
    axes.set_xlabel(X_LABELS[x])
    axes.set_ylabel("Peak memory (GiB)")

    metric_handles = [
        Line2D([], [], color=INK_SECONDARY, linestyle=style, marker=marker, markersize=4.5, linewidth=1.6, label=name)
        for name, _, style, marker in MEMORY_METRICS
    ]
    if theory:
        metric_handles.append(
            Line2D([], [], color=INK_SECONDARY, linewidth=3.2, alpha=0.35, label="analytical estimate")
        )
    first = axes.legend(
        handles=mode_handles(modes, bool(oom)), loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False
    )
    axes.add_artist(first)
    axes.legend(handles=metric_handles, loc="upper left", bbox_to_anchor=(1.01, 0.72), frameon=False)

    title, subtitle = title_for("peak memory", runs, x)
    figure.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    axes.set_title(subtitle, loc="left", fontsize=8, color=INK_SECONDARY)
    note = (
        "Torch peaks over the whole phase, including the warm-up." if not steady else "Torch peaks after the warm-up."
    )
    note += " The analytical estimate is for the generator step and is drawn beyond the GPU memory, where runs failed."
    if any(run["mode"] == "compiled" for run in runs):
        note += "\nCompiled nvidia-smi values include the local memory of the CUDA context grown by max-autotune (see compile-overhead/)."
    figure.text(0.01, 0.005, note, fontsize=7, color=INK_SECONDARY, va="bottom")
    figure.tight_layout(rect=(0, 0.06, 0.85, 0.95))
    save(figure, out_dir / f"memory-vs-{x}")


def plot_time(runs: list[Run], x: str, out_dir: Path, log_y: bool) -> None:
    xticks = sorted({run["hyperparameters"][x] for run in runs})
    figure, axes = plt.subplots(figsize=(11, 4.6))
    y_max = 0.0
    oom: list[int] = []
    modes = present_modes(runs)
    for mode in modes:
        colour = MODES[mode][1]
        points, failed = series(
            runs,
            x,
            mode,
            lambda run: run["generator"]["average_step_time_ms"] if run["generator"]["complete"] else None,
        )
        if points:
            axes.plot(*zip(*points, strict=True), "-o", color=colour, linewidth=1.8, markersize=4.5)
            y_max = max(y_max, max(y for _, y in points))
            label_end(axes, points, mode)
        oom += failed
        mark_out_of_memory(axes, failed)
    style_axes(axes, xticks, log_y, y_max)
    pad_right(axes)
    axes.set_xlabel(X_LABELS[x])
    axes.set_ylabel("Average step time (ms)")
    axes.legend(handles=mode_handles(modes, bool(oom)), loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)

    title, subtitle = title_for("average step time", runs, x)
    figure.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    axes.set_title(subtitle, loc="left", fontsize=8, color=INK_SECONDARY)
    figure.tight_layout(rect=(0, 0, 0.85, 0.95))
    save(figure, out_dir / f"time-vs-{x}")


def print_table(runs: list[Run], x: str, theory: dict[str, Estimate]) -> None:
    print(f"\n== {runs[0]['experiment']}, causal generator ({X_LABELS[x]}); memory in GiB, step time in ms")
    print(f"{x:>6} {'mode':>12} | {'step':>7} {'alloc':>7} {'resvd':>7} {'smi':>7} | {'estimate':>8}")
    for run in sorted(runs, key=lambda run: (run["hyperparameters"][x], run["mode"])):
        hyper, data = run["hyperparameters"], run["generator"]
        estimate = theory.get(run["mode"])
        theory_cell = f"{estimate(hyper) / MIB_PER_GIB:>8.2f}" if estimate else f"{'-':>8}"
        if data["complete"]:
            cells = (
                f"{data['average_step_time_ms']:>7.0f} {data['peak_torch_allocated_mib'] / MIB_PER_GIB:>7.2f} "
                f"{data['peak_torch_reserved_mib'] / MIB_PER_GIB:>7.2f} {data['peak_nvidia_smi_mib'] / MIB_PER_GIB:>7.2f}"
            )
        else:
            cells = f"{run['status'].upper():>7}{'':>24}"
        print(f"{hyper[x]:>6} {MODES[run['mode']][0]:>12} | {cells} | {theory_cell}")


def gpu_memory_gib(settings: dict[str, Any]) -> int:
    """The memory of the GPU as nvidia-smi reports it, e.g. 80 GiB (81920 MiB) for an "A100-SXM4-80GB".

    The total that torch reports (81176 MiB) leaves out what the CUDA context takes, so the capacity is taken from the
    device name and only falls back to the torch total when the name has none.
    """
    named = re.search(r"(\d+)GB", settings["gpu"])
    return int(named.group(1)) if named else round(settings["gpu_total_mib"] / MIB_PER_GIB)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=HERE / "plots", help="where to write the figures")
    parser.add_argument("--log-y", action="store_true", help="use a logarithmic y axis")
    parser.add_argument(
        "--steady", action="store_true", help="plot the torch peaks of the steps after the warm-up, not of the phase"
    )
    parser.add_argument(
        "--gpu-limit-gib",
        type=float,
        default=None,
        help="reference line in GiB in the memory plots (default: the GPU memory in the summaries, 0 disables it)",
    )
    parser.add_argument("--no-theory", action="store_true", help="do not draw the analytical estimate")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    experiments = load_summaries(HERE)
    if not experiments:
        print(f"no summary.json found below {HERE}", file=sys.stderr)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_theory = {} if args.no_theory else load_theory()

    for name, runs in experiments.items():
        if name.endswith("-upstream") and name.removesuffix("-upstream") in experiments:
            continue  # drawn in the figures of the experiment it repeats
        runs = runs + experiments.get(f"{name}-upstream", [])
        unknown = sorted({run["mode"] for run in runs} - set(MODES))
        if unknown:
            print(f"{name}: modes {unknown} are not plotted, skipped", file=sys.stderr)
            continue
        x = varying_hyperparameter(name, runs)
        if x is None:
            continue
        theory = all_theory
        limit = args.gpu_limit_gib
        if limit is None:
            limit = gpu_memory_gib(runs[0]["settings"])
        print_table(runs, x, theory)
        plot_memory(runs, x, args.out_dir, args.log_y, args.steady, limit or None, theory)
        plot_time(runs, x, args.out_dir, args.log_y)
    return 0


if __name__ == "__main__":
    sys.exit(main())
