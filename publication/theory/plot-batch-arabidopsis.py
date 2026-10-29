#!/usr/bin/env python3
"""Analytical peak memory of the dense, sparse and compiled sparse generator for the Arabidopsis configuration vs batch size.

F = 422 TFs and G = 578 target genes, k = 15, d = 3, w = 2, n = 1 and the critic, labeler and causal controller of
publication/arabidopsis-ablation are fixed; the batch size B of the causal GAN varies (2350 in the configuration).
The measured torch-allocated peaks of the generator phase of the ablation (B = 2350) are drawn as filled markers.
Configuration, colours and style are those of plot-scaled-arabidopsis.py.

Usage::

    python plot-batch-arabidopsis.py
    python plot-batch-arabidopsis.py --b-min 16 --b-max 131072   # beyond the practical range of B (default 64..8192)
"""

import argparse
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("plot_scaled_arabidopsis", HERE / "plot-scaled-arabidopsis.py")
assert spec is not None and spec.loader is not None
scaled = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scaled)  # also sets the rcParams of the style

model = scaled.model
F, G, B0, K, W, D, N = scaled.F0, scaled.G0, scaled.B, scaled.K, scaled.W, scaled.D, scaled.N
A = scaled.ARCHITECTURE
GPU_GIB, GIB = scaled.GPU_GIB, scaled.GIB
INK, INK_SECONDARY, INK_MUTED, GRID, SURFACE = (
    scaled.INK, scaled.INK_SECONDARY, scaled.INK_MUTED, scaled.GRID, scaled.SURFACE
)

COLOURS = {name: colour for name, colour, _, _ in scaled.SERIES}
VARIANTS = {name: variant for name, _, _, variant in scaled.SERIES}
PEAKS = {
    "compiled": lambda b: model.compute_compile(b, F, G, K, W, D, N, A),
    "un-compiled": lambda b: model.compute_all(b, F, G, K, W, D, N, A)[1],
    "original (dense)": lambda b: model.compute_all(b, F, G, K, W, D, N, A)[0],
}
# Where the GPU-memory crossing of each series is labelled, in points from it, and the horizontal alignment. The
# un-compiled curve crosses steeply and is below the line to its left, so its label goes there, above the line. The
# dense and compiled crossings are close together at the right, where both curves come from below: their labels are
# stacked above the line, ending at the crossing.
CROSSING_LABELS = {
    "un-compiled": ((-6, 4), "right"),
    "original (dense)": ((-6, 4), "right"),
    "compiled": ((6, 14), "right"),
}

# Vertical offsets in points of the labels at the ends of the curves: the dense and compiled curves end close together
END_LABEL_OFFSETS = {"un-compiled": 0, "original (dense)": 5, "compiled": -5}


def gpu_limit(peak, high: float = 1e8) -> float:
    """The batch size at which the peak reaches the GPU memory, by bisection."""
    low = 0.0
    for _ in range(100):
        middle = (low + high) / 2
        low, high = (middle, high) if peak(middle) / GIB < GPU_GIB else (low, middle)
    return (low + high) / 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=HERE / "theory-batch-arabidopsis.pdf")
    parser.add_argument("--b-min", type=float, default=64)
    parser.add_argument("--b-max", type=float, default=8192)  # larger batches are not used in practice
    args = parser.parse_args()

    bs = np.geomspace(args.b_min, args.b_max, 400)
    figure, axes = plt.subplots(figsize=(11, 5.8))
    print(f"{'series':>17} | {'estimate B=2350 (GiB)':>21} {'measured B=2350 (GiB)':>21} | {'B at 80 GiB':>11}")
    for name, peak in PEAKS.items():
        colour = COLOURS[name]
        ys = np.array([peak(b) / GIB for b in bs])
        axes.plot(bs, ys, "-", color=colour, linewidth=1.8, label=name)
        axes.annotate(name, (bs[-1], ys[-1]), xytext=(6, END_LABEL_OFFSETS[name]), textcoords="offset points",
                      va="center", fontsize=7,
                      color=INK_SECONDARY, annotation_clip=False)
        b_limit = gpu_limit(peak)
        if args.b_min <= b_limit <= args.b_max:
            offset, alignment = CROSSING_LABELS[name]
            axes.plot([b_limit], [GPU_GIB], "o", color=colour, markersize=5, zorder=5)
            axes.annotate(f"{name}: B = {b_limit:,.0f}", (b_limit, GPU_GIB), xytext=offset, ha=alignment,
                          textcoords="offset points", fontsize=7, color=INK_SECONDARY)
        # Arabidopsis configuration, B = 2350: estimate (open) and measurement (filled)
        estimate = peak(B0) / GIB
        axes.plot([B0], [estimate], "o", color=colour, markersize=4.5, markerfacecolor=SURFACE, zorder=5)
        measured = scaled.measured_gib(VARIANTS[name])
        if measured is not None:
            axes.plot([B0], [measured], "D", color=colour, markersize=4.5, zorder=6)
        measured_cell = f"{measured:>21.2f}" if measured is not None else f"{'-':>21}"
        print(f"{name:>17} | {estimate:>21.2f} {measured_cell} | {b_limit:>11,.0f}")

    axes.axhline(GPU_GIB, color=INK_MUTED, linewidth=1.0)
    # Below the line at the left end, where every curve is far below it
    axes.annotate(f"GPU memory {GPU_GIB} GiB", (bs[0], GPU_GIB), xytext=(0, -10), textcoords="offset points",
                  fontsize=7, color=INK_SECONDARY)
    axes.axvline(B0, color=GRID, linewidth=1.0, zorder=0)
    axes.annotate(f"Arabidopsis configuration (B = {B0})", (B0, 0.02), xytext=(4, 0), textcoords="offset points",
                  xycoords=("data", "axes fraction"), fontsize=7, color=INK_SECONDARY)
    axes.set_xscale("log", base=2)
    axes.xaxis.set_major_formatter(FuncFormatter(lambda b, _: f"{b:,.0f}"))
    axes.xaxis.set_minor_formatter(NullFormatter())
    axes.set_yscale("log")
    axes.grid(axis="y", color=GRID, linewidth=0.7, which="major")
    axes.set_axisbelow(True)
    axes.margins(x=0.01)
    low, high = axes.get_xlim()
    axes.set_xlim(low, high * (high / low) ** 0.14)  # room for the end labels on the log axis
    axes.set_xlabel("Batch size of the causal GAN (B)")
    axes.set_ylabel("Analytical peak memory (GiB)")
    handles, _ = axes.get_legend_handles_labels()
    handles += [
        plt.Line2D([], [], color=INK_SECONDARY, linestyle="", marker="o", markersize=4.5, markerfacecolor=SURFACE,
                   label=f"estimate, B = {B0}"),
        plt.Line2D([], [], color=INK_SECONDARY, linestyle="", marker="D", markersize=4.5,
                   label=f"measured, B = {B0} (torch allocated)"),
    ]
    axes.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)

    figure.suptitle("Analytical peak memory of the causal generator, Arabidopsis configuration vs batch size", x=0.01,
                    ha="left", fontsize=11, color=INK)
    figure.text(0.01, 0.915, f"fixed: F={F}, G={G}, k={K}, d={D}, w={W}, n={N}; measured: "
                "publication/arabidopsis-ablation (GRouNdGAN, sparse eager and sparse compiled with TF32)",
                fontsize=8, color=INK_SECONDARY)
    figure.tight_layout(rect=(0, 0, 0.85, 0.9))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Crop to the drawn content (titles, axes, legend) so that LaTeX centres the graph, not empty margins
    figure.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(args.out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
