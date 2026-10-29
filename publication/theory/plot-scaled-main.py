#!/usr/bin/env python3
"""Analytical peak memory of the dense, sparse and compiled sparse generator for the main configuration scaled by x.

G = 937x target genes and F = 63x TFs (the PBMC-All dataset of the main experiments), with w = 1, d = 2, n = 1,
B = 1024 and k = 15. The peaks come from gpu_memory_model.py; the style follows publication/plot-results.py.

Usage::

    python plot-scaled-main.py --out ../../../GRouNdGAN-Paper/figures/scaling/theory-scaled-main.pdf
"""

import argparse
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gpu_memory_model", HERE / "gpu_memory_model.py")
assert spec is not None and spec.loader is not None
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)

G0, F0, B, K, W, D, N = 937, 63, 1024, 15, 1, 2, 1
GPU_GIB = 80
GIB = 2**30

# Colours of publication/plot-results.py, by implementation
SERIES = (
    ("compiled", "#2a78d6", lambda F, G: model.compute_compile(B, F, G, K, W, D, N)),
    ("un-compiled", "#eb6834", lambda F, G: model.compute_all(B, F, G, K, W, D, N)[1]),
    ("original (dense)", "#1baf7a", lambda F, G: model.compute_all(B, F, G, K, W, D, N)[0]),
)
INK, INK_SECONDARY, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=HERE / "theory-scaled-main.pdf")
    parser.add_argument("--x-max", type=float, default=20)
    args = parser.parse_args()

    xs = np.linspace(0.25, args.x_max, 400)
    figure, axes = plt.subplots(figsize=(11, 5.8))
    for name, colour, peak in SERIES:
        ys = np.array([peak(F0 * x, G0 * x) / GIB for x in xs])
        axes.plot(xs, ys, "-", color=colour, linewidth=1.8, label=name)
        axes.annotate(name, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=7,
                      color=INK_SECONDARY, annotation_clip=False)
        if ys[-1] > GPU_GIB:  # where the peak reaches the GPU memory, by bisection
            low, high = xs[0], xs[-1]
            for _ in range(60):
                middle = (low + high) / 2
                low, high = (middle, high) if peak(F0 * middle, G0 * middle) / GIB < GPU_GIB else (low, middle)
            x_limit = (low + high) / 2
            axes.plot([x_limit], [GPU_GIB], "o", color=colour, markersize=5, zorder=5)
            axes.annotate(f"x = {x_limit:.1f}\n(G = {G0 * x_limit:,.0f})", (x_limit, GPU_GIB), xytext=(4, -22),
                          textcoords="offset points", fontsize=7, color=INK_SECONDARY)
        # main configuration, x = 1
        axes.plot([1], [peak(F0, G0) / GIB], "o", color=colour, markersize=4.5, markerfacecolor=SURFACE, zorder=5)

    axes.axhline(GPU_GIB, color=INK_MUTED, linewidth=1.0)
    axes.annotate(f"GPU memory {GPU_GIB} GiB", (xs[0], GPU_GIB), xytext=(0, 3), textcoords="offset points",
                  fontsize=7, color=INK_SECONDARY)
    axes.axvline(1, color=GRID, linewidth=1.0, zorder=0)
    axes.set_yscale("log")
    axes.annotate("main configuration (x = 1)", (1, 0.02), xytext=(4, 0), textcoords="offset points",
                  xycoords=("data", "axes fraction"), fontsize=7, color=INK_SECONDARY)
    axes.grid(axis="y", color=GRID, linewidth=0.7, which="major")
    axes.set_axisbelow(True)
    axes.margins(x=0.01)
    low, high = axes.get_xlim()
    axes.set_xlim(low, high + 0.14 * (high - low))
    axes.set_xlabel(f"Scale factor x of the main configuration (G = {G0}x target genes, F = {F0}x TFs)")
    axes.set_ylabel("Analytical peak memory (GiB)")
    secondary = axes.secondary_xaxis("top", functions=(lambda x: G0 * x, lambda g: g / G0))
    secondary.set_xlabel("Number of target genes (G)")
    secondary.tick_params(colors=INK_MUTED)
    axes.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)

    figure.suptitle("Analytical peak memory of the causal generator, main configuration scaled by x", x=0.01,
                    ha="left", fontsize=11, color=INK)
    figure.text(0.01, 0.915, f"fixed: B={B}, k={K}, d={D}, w={W}, n={N}; open circles: main configuration",
                fontsize=8, color=INK_SECONDARY)
    figure.tight_layout(rect=(0, 0, 0.85, 0.9))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Crop to the drawn content (titles, axes, legend) so that LaTeX centres the graph, not empty margins
    figure.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(args.out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
