#!/usr/bin/env python3
"""Analytical peak memory of the dense, sparse and compiled sparse generator for the Arabidopsis configuration scaled by x.

The configuration of publication/arabidopsis-ablation (the paper's Arabidopsis model): G = 578x target genes and
F = 422x TFs, with k = 15, B = 2350, d = 3, w = 2, n = 1, and the critic, labeler and causal controller of
paper-arabidopsis (passed to gpu_memory_model.py as `architecture`). The measured torch-allocated peaks of the
generator phase of the ablation (x = 1) are drawn as filled markers. The style follows plot-scaled-main.py.

Usage::

    python plot-scaled-arabidopsis.py              # x up to 10
    python plot-scaled-arabidopsis.py --x-max 0    # x up to where the compiled model reaches the GPU memory
    python plot-scaled-arabidopsis.py --out ../../../GRouNdGAN-Paper/figures/scaling/theory-scaled-arabidopsis.pdf
"""

import argparse
import importlib.util
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ABLATION = HERE.parent / "arabidopsis-ablation" / "results"
spec = importlib.util.spec_from_file_location("gpu_memory_model", HERE / "gpu_memory_model.py")
assert spec is not None and spec.loader is not None
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)

# publication/arabidopsis-ablation/causal_gan.cfg ([Model], [Training] and [CC Model])
G0, F0, B, K, W, D, N = 578, 422, 2350, 15, 2, 3, 1
ARCHITECTURE = {
    "latent_dim": 64,
    "cc_layers": (224, 752, 912),
    "crit_layers": (752, 368, 176),
    "labeler_layers": (992, 496, 256),
}
GPU_GIB = 80
GIB, MIB_PER_GIB = 2**30, 1024

# Colours of publication/plot-results.py, by implementation; the last item is the ablation variant measured at x = 1
SERIES = (
    ("compiled", "#2a78d6", lambda F, G: model.compute_compile(B, F, G, K, W, D, N, ARCHITECTURE),
     "sparse-compiled-tf32"),
    ("un-compiled", "#eb6834", lambda F, G: model.compute_all(B, F, G, K, W, D, N, ARCHITECTURE)[1],
     "sparse-eager-tf32"),
    ("original (dense)", "#1baf7a", lambda F, G: model.compute_all(B, F, G, K, W, D, N, ARCHITECTURE)[0],
     "a-upstream"),
)
# Where the GPU-memory crossing of each series is labelled: (x of the anchor on the GPU-memory line, None for the
# crossing itself; offset in points). The dense and un-compiled crossings are close together near x = 2, so their
# labels, with the series name, are stacked above the line at x = 5, below the steep rise of both curves.
CROSSING_LABELS = {
    "compiled": (None, (4, -22), "x = {x:.1f}\n(G = {G:,.0f})"),
    "un-compiled": (5, (0, 13), "un-compiled: x = {x:.1f} (G = {G:,.0f})"),
    "original (dense)": (5, (0, 3), "original (dense): x = {x:.1f} (G = {G:,.0f})"),
}
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


def gpu_limit(peak, high: float = 1000.0) -> float:
    """The x at which the peak reaches the GPU memory, by bisection."""
    low = 0.0
    for _ in range(60):
        middle = (low + high) / 2
        low, high = (middle, high) if peak(F0 * middle, G0 * middle) / GIB < GPU_GIB else (low, middle)
    return (low + high) / 2


def measured_gib(variant: str) -> float | None:
    """The torch-allocated peak of the generator phase of an ablation run, if it ran and completed."""
    path = ABLATION / variant / "summary.json"
    if not path.exists():
        return None
    generator = json.loads(path.read_text())["generator"]
    return generator["peak_torch_allocated_mib"] / MIB_PER_GIB if generator["complete"] else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=HERE / "theory-scaled-arabidopsis.pdf")
    parser.add_argument("--x-max", type=float, default=10,
                        help="largest x (default 10; 0: just beyond where the compiled model reaches the GPU memory)")
    args = parser.parse_args()

    limits = {name: gpu_limit(peak) for name, _, peak, _ in SERIES}
    x_max = args.x_max or math.ceil(1.15 * max(limits.values()))
    xs = np.linspace(0.25, x_max, 400)
    figure, axes = plt.subplots(figsize=(11, 5.8))
    print(f"{'series':>17} | {'estimate x=1 (GiB)':>18} {'measured x=1 (GiB)':>18} | {'x at 80 GiB':>11} {'G':>7} {'F':>7}")
    for name, colour, peak, variant in SERIES:
        ys = np.array([peak(F0 * x, G0 * x) / GIB for x in xs])
        axes.plot(xs, ys, "-", color=colour, linewidth=1.8, label=name)
        axes.annotate(name, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=7,
                      color=INK_SECONDARY, annotation_clip=False)
        x_limit = limits[name]
        if x_limit <= x_max:
            axes.plot([x_limit], [GPU_GIB], "o", color=colour, markersize=5, zorder=5)
            anchor, offset, text = CROSSING_LABELS[name]
            axes.annotate(text.format(x=x_limit, G=G0 * x_limit), (anchor or x_limit, GPU_GIB), xytext=offset,
                          textcoords="offset points", fontsize=7, color=INK_SECONDARY)
        # Arabidopsis configuration, x = 1: estimate (open) and measurement (filled)
        estimate = peak(F0, G0) / GIB
        axes.plot([1], [estimate], "o", color=colour, markersize=4.5, markerfacecolor=SURFACE, zorder=5)
        measured = measured_gib(variant)
        if measured is not None:
            axes.plot([1], [measured], "D", color=colour, markersize=4.5, zorder=6)
        measured_cell = f"{measured:>18.2f}" if measured is not None else f"{'-':>18}"
        print(f"{name:>17} | {estimate:>18.2f} {measured_cell} | {x_limit:>11.2f} {G0 * x_limit:>7.0f} {F0 * x_limit:>7.0f}")

    axes.axhline(GPU_GIB, color=INK_MUTED, linewidth=1.0)
    # Labelled away from the crossings of the dense and un-compiled series and their labels: above the line in the
    # middle when the compiled crossing (labelled below the line) is drawn, else below the line at the right end
    if limits["compiled"] <= x_max:
        gpu_label = {"xy": (0.55 * x_max, GPU_GIB), "xytext": (0, 3)}
    else:
        gpu_label = {"xy": (x_max, GPU_GIB), "xytext": (0, -10), "ha": "right"}
    axes.annotate(f"GPU memory {GPU_GIB} GiB", **gpu_label, textcoords="offset points", fontsize=7,
                  color=INK_SECONDARY)
    axes.axvline(1, color=GRID, linewidth=1.0, zorder=0)
    axes.set_yscale("log")
    axes.annotate("Arabidopsis configuration (x = 1)", (1, 0.02), xytext=(4, 0), textcoords="offset points",
                  xycoords=("data", "axes fraction"), fontsize=7, color=INK_SECONDARY)
    axes.grid(axis="y", color=GRID, linewidth=0.7, which="major")
    axes.set_axisbelow(True)
    axes.margins(x=0.01)
    low, high = axes.get_xlim()
    axes.set_xlim(low, high + 0.14 * (high - low))
    axes.set_xlabel(f"Scale factor x of the Arabidopsis configuration (G = {G0}x target genes, F = {F0}x TFs)")
    axes.set_ylabel("Analytical peak memory (GiB)")
    secondary = axes.secondary_xaxis("top", functions=(lambda x: G0 * x, lambda g: g / G0))
    secondary.set_xlabel("Number of target genes (G)")
    secondary.tick_params(colors=INK_MUTED)
    handles, _ = axes.get_legend_handles_labels()
    handles += [
        plt.Line2D([], [], color=INK_SECONDARY, linestyle="", marker="o", markersize=4.5, markerfacecolor=SURFACE,
                   label="estimate, x = 1"),
        plt.Line2D([], [], color=INK_SECONDARY, linestyle="", marker="D", markersize=4.5,
                   label="measured, x = 1 (torch allocated)"),
    ]
    axes.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)

    figure.suptitle("Analytical peak memory of the causal generator, Arabidopsis configuration scaled by x", x=0.01,
                    ha="left", fontsize=11, color=INK)
    figure.text(0.01, 0.915, f"fixed: B={B}, k={K}, d={D}, w={W}, n={N}; measured: publication/arabidopsis-ablation "
                "(GRouNdGAN, sparse eager and sparse compiled with TF32)", fontsize=8, color=INK_SECONDARY)
    figure.tight_layout(rect=(0, 0, 0.85, 0.9))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Crop to the drawn content (titles, axes, legend) so that LaTeX centres the graph, not empty margins
    figure.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(args.out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
