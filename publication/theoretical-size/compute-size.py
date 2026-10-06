#!/usr/bin/env python3
"""Report the analytic GPU memory estimate for the causal GAN, and plot it.

The formulas live in `gpu_memory_model.py`, shared with
`../memory-profiling/get-size-from-trace.py`, so these theoretical curves cannot
drift from the ones overlaid on the measured peaks in `memory-vs-genes.png`.
Curve colours, the log/linear y-axis pair, and the PDF output all match that
script, so the two figures read as a set.

Three configurations are plotted, all from `theory_curves`:

* `orig` -- the dense masked-linear implementation.
* `new` -- the sparse implementation.
* `compile` -- the sparse implementation under torch.compile, whose generator
  temporaries are largely fused away (`TEMP_FACTOR_COMPILE`).

`nvidia-smi` is drawn for the compiled runs as well, since it reports whole-device
usage rather than allocator bytes and so sits well above every allocator curve.

Two things the original script handled in its caller, and this one gets from the
model itself:

* there are two discriminators, a labeler and an antilabeler. They are
  architecturally identical and both are resident, so both are counted. See
  `compute_antilabeler`.
* sparse gradient coordinates and BatchNorm `num_batches_tracked` are int64, not
  FP32. The model returns bytes throughout, so they are already weighted.
"""

import matplotlib.pyplot as plt
from gpu_memory_model import (
    BATCH_SIZE,
    SIZES,
    D,
    K,
    N,
    W,
    compute_antilabeler,
    compute_cc,
    compute_compile,
    compute_crit,
    compute_gen,
    compute_labeler,
    compute_smi,
    compute_sparse_gen,
    theory_curves,
    to_gib,
    to_mib,
)

# Same palette as get-size-from-trace.py, so variant identity carries across figures.
VARIANT_COLOURS = {"orig": "tab:orange", "new": "tab:blue", "compile": "tab:green"}
VARIANTS = tuple(VARIANT_COLOURS)
GPU_LIMIT_GIB = 80.0

B = BATCH_SIZE
REPORT_GENES = 7000  # gene count the per-part table below is reported for
F, G = SIZES[REPORT_GENES]
k, w, d, n = K, W, D, N


def report(name: str, parts: dict[str, int]) -> dict[str, float]:
    """Print one component's figures in both bytes-as-given and MiB."""
    memory = {key: to_mib(value) for key, value in parts.items()}
    print(f"{name} bytes: {parts}")
    print(f"{name} memory: {memory}")
    print(f"{name} total params bytes: {parts['model']}")
    print(f"{name} memory: {memory['model']:.3f} MiB")
    return memory


gen_memory = report("Gen", compute_gen(B, F, G, k, w, d, n))
gen_sparse_memory = report("Gen sparse", compute_sparse_gen(B, F, G, k, w, d, n))
cc_memory = report("CC", compute_cc(F, G, B))
crit_memory = report("Critic", compute_crit(F, G, B))
labeler_memory = report("Labeler", compute_labeler(F, G, B))
antilabeler_memory = report("Antilabeler", compute_antilabeler(F, G, B))

# The generator, causal controller, critic and both discriminators are all resident.
SHARED = [cc_memory, crit_memory, labeler_memory, antilabeler_memory]
PARTS = [gen_memory, *SHARED]
SPARSE_PARTS = [gen_sparse_memory, *SHARED]


def total(parts: list[dict[str, float]]) -> float:
    """Model + gradients + optimizer states + activations + temporaries."""
    return sum(part["model"] + part["grad"] + part["opt"] + part["act"] + part["temp"] for part in parts)


print(
    f"Total model memory (including gradients, optimizer states, activations and temporary tensors): {total(PARTS):.3f} MiB"
)
print(
    f"Total sparse model memory (including gradients, optimizer states, activations and temporary tensors): {total(SPARSE_PARTS):.3f} MiB"
)
# These two use the same narrower term set as `compute_all`: parameters and optimizer
# state for every part, plus the generator's temporaries, activations and gradients.
print(
    f"Estimated maximum dense GPU usage: {sum(p['model'] + p['opt'] for p in PARTS) + gen_memory['temp'] + gen_memory['act'] + gen_memory['grad']:.3f} MiB"
)
print(
    f"Estimated maximum sparse GPU usage: {sum(p['model'] + p['opt'] for p in SPARSE_PARTS) + gen_sparse_memory['temp'] + gen_sparse_memory['act'] + gen_sparse_memory['grad']:.3f} MiB"
)

# The compiled peak is the sparse peak with the generator's temporaries scaled by
# TEMP_FACTOR_COMPILE; nvidia-smi adds a fixed context/arena cost on top. Reported
# so the per-part table above can be reconciled with the curves below.
print(f"Estimated maximum compiled GPU usage: {to_mib(compute_compile(B, F, G, k, w, d, n)):.3f} MiB")
print(f"Predicted compile nvidia-smi peak: {to_mib(compute_smi(REPORT_GENES)):.3f} MiB")

genes = sorted(SIZES)
figure, axes = plt.subplots(figsize=(11.5, 6.5))

for variant in VARIANTS:
    colour = VARIANT_COLOURS[variant]
    _, curve = theory_curves(variant, B)
    axes.plot(genes, curve, "-", color=colour, linewidth=1.6, alpha=0.85, label=f"{variant} theoretical")

# nvidia-smi reports whole-device usage, so the allocator curves should not be read
# against it directly. `compile_smi` is that peak; it is the compile curve plus a
# fixed context/arena cost, which is why the overhead-corrected form would coincide
# with the green curve exactly. Only the peak is drawn, to avoid duplicating it.
smi_genes = genes
axes.plot(
    smi_genes,
    [to_gib(compute_smi(num_genes, B)) for num_genes in smi_genes],
    ":D",
    color="tab:red",
    markersize=4,
    linewidth=1.3,
    label="compile nvidia-smi peak (model)",
)

axes.axhline(GPU_LIMIT_GIB, color="0.35", linestyle=":", linewidth=1.4, label=f"{GPU_LIMIT_GIB:.0f} GiB GPU limit")

axes.set_xlabel("Number of genes")
axes.set_ylabel("Peak CUDA memory (GiB)")
axes.set_title("Analytic memory estimate\nlines = gpu_memory_model.py, no measurements")
axes.grid(alpha=0.3, which="both")
axes.legend(fontsize=8, loc="upper left")

# Both scales, as in memory-vs-genes.png: the log view shows the three orders of
# magnitude between variants, the linear view shows where the 80 GiB limit bites.
axes.set_yscale("log")
figure.tight_layout()
figure.savefig("gpu_memory_estimate-log.png", dpi=600)
figure.savefig("gpu_memory_estimate-log.pdf", dpi=600)

axes.set_yscale("linear")
axes.set_ylim(top=GPU_LIMIT_GIB * 1.05, bottom=0)
figure.tight_layout()
figure.savefig("gpu_memory_estimate.png", dpi=600)
figure.savefig("gpu_memory_estimate.pdf", dpi=600)

plt.close(figure)
