"""Analytic GPU-memory model for the causal GAN.

Shared by `compute-size.py` (which reports per-component memory and plots the
estimate on its own) and `get-size-from-trace.py` (which overlays the estimate
on peaks measured from PyTorch memory-history snapshots). Keeping the formulas
in one place matters because these estimates are the paper's claim: two copies
would inevitably drift.

UNITS
-----
Every function returns **bytes**, not element counts. That is deliberate. GPU
memory here is not homogeneous: parameters, gradients, activations and Adam
state are FP32 (4 bytes/element), but two things are int64 (8 bytes/element):

* the sparse generator's gradient coordinates -- an (index, value) pair per
  nonzero weight, so 16 bytes against the 4 of the dense FP32 tensor it replaces
  -- and
* each BatchNorm layer's `num_batches_tracked` counter.

Accounting in element counts forced callers to remember to reweight those,
which is exactly how `coords` came to be counted twice in the original script.
Weighting them here makes each case explicit at the point it is defined.

TWO DISCRIMINATORS
------------------
The model trains a *labeler* and an *antilabeler*. They are architecturally
identical -- the same 2000-2000-2000 network over the same inputs -- and both
are resident at the peak, so both are counted. `compute_all` includes each once;
`compute-size.py` prints them separately.

`compute_all` returns `(dense, sparse)` in that order -- dense first, matching
`compute_gen`, which models `upstream/src/layers/masked_linear.py`. Dense mode
forms `weight * mask` as a full dense FP32 copy on every forward and again in
backward, so the masked entries cost exactly as much as the live ones. Sparse
mode keeps only the nonzero coordinates.
"""

from __future__ import annotations

FLOAT_BYTES = 4
INT64_BYTES = 8

# Defaults used by `compute-size.py`. `BATCH_SIZE` must match the `batch size`
# setting in the profiled config (1024 in mem-profile2/causal_gan.cfg).
BATCH_SIZE = 1024
K = 15  # top TFs per gene; the dense profile in causal_gan.cfg uses 14
W = 1
D = 2
N = 1

# Fraction of the generator's `temp` requirement that survives torch.compile.
# Inductor fuses the elementwise work into the spmm/scatter kernels, so the
# full-size temporaries the eager model predicts are largely not materialised.
# Least-squares fitted against the measured peaks in
# mem-profile2/traces/*-compile-genes.pickle over 1000..16500 genes; it
# reproduces them to within 8.2% at 1000 genes and under 2% above 4000.
TEMP_FACTOR_COMPILE = 0.02504

# nvidia-smi reports total device usage, so it exceeds the caching allocator's
# reserved bytes by a roughly constant CUDA-context plus inductor-arena cost.
# Measured as 6.195 +/- 0.001 GiB across all 33 compile runs.
SMI_OVERHEAD_BYTES = int(6.195 * 1024**3)

# Number of genes -> (F, G), the fixed and gene-varying input dimensions.
SIZES: dict[int, tuple[int, int]] = {
    1000: (63, 928),
    1500: (89, 1410),
    2000: (127, 1873),
    2500: (161, 2339),
    3000: (202, 2798),
    3500: (237, 3263),
    4000: (287, 3713),
    4500: (324, 4176),
    5000: (370, 4630),
    5500: (414, 5086),
    6000: (464, 5536),
    6500: (506, 5994),
    7000: (555, 6445),
    7500: (599, 6901),
    8000: (639, 7361),
    8500: (685, 7815),
    9000: (733, 8267),
    9500: (774, 8726),
    10000: (817, 9183),
    10500: (851, 9649),
    11000: (887, 10113),
    11500: (918, 10582),
    12000: (949, 11051),
    12500: (974, 11526),
    13000: (1002, 11998),
    13500: (1027, 12473),
    14000: (1045, 12955),
    14500: (1064, 13436),
    15000: (1098, 13902),
    15500: (1118, 14382),
    16000: (1131, 14869),
    16500: (1146, 15354),
}


def compute_gen(batch_size: int, F: int, G: int, k: int, w: int, d: int, n: int) -> dict[str, int]:
    """Generator memory, in bytes, for the dense (`orig`) masked-linear implementation."""
    I = F + G * n
    H = G * (k + n) * w
    O = G

    l_i = FLOAT_BYTES * I * H
    l_i_bias = FLOAT_BYTES * H
    l_h = FLOAT_BYTES * H**2
    l_h_bias = FLOAT_BYTES * H
    l_o = FLOAT_BYTES * H * O
    l_o_bias = FLOAT_BYTES * O

    bnorm_train = FLOAT_BYTES * 2 * (d + 1) * H  # bias and weight
    # Running mean and variance are FP32, but num_batches_tracked is one int64 per layer.
    bnorm_state = FLOAT_BYTES * 2 * (d + 1) * H + INT64_BYTES * (d + 1)

    T = l_i + d * l_h + l_o + l_i_bias + d * l_h_bias + l_o_bias + bnorm_train
    gen = T + (l_i + l_h + l_o) + bnorm_state  # Extra ls for masks

    grad = T
    opt = 3 * T  # two Adam moments plus the parameter itself, all FP32

    act = FLOAT_BYTES * batch_size * ((d + 1) * H + O)  # activations for linear layers
    act += FLOAT_BYTES * batch_size * ((d + 1) * H)  # linear-layer inputs, saved for backward
    act += FLOAT_BYTES * batch_size * ((d + 1) * H)  # activations for batchnorm layers

    # In `upstream/src/layers/masked_linear.py`, every forward computes `weight * mask` and saves that resulting dense tensor for backward.
    # The mask zeros most entries, but multiplication still creates a full-sized dense FP32 copy.
    temp = l_i + d * l_h + l_o

    # During backward, `MaskedLinearFunction.backward` first forms a dense gradient matrix with a matrix multiply, then multiplies it by the mask.
    # That creates another full-matrix temporary.
    temp += l_i + d * l_h + l_o - act

    return {
        "l_i": l_i,
        "l_i_bias": l_i_bias,
        "l_h": l_h,
        "l_h_bias": l_h_bias,
        "l_o": l_o,
        "l_o_bias": l_o_bias,
        "bnorm_train": bnorm_train,
        "bnorm_state": bnorm_state,
        "T": T,
        "model": gen,
        "grad": grad,
        "opt": opt,
        "act": act,
        "temp": temp,
    }


def compute_sparse_gen(batch_size: int, F: int, G: int, k: int, w: int, d: int, n: int) -> dict[str, int]:
    """Generator memory, in bytes, for the sparse (`new`) implementation."""
    H = G * (k + n) * w
    O = G

    l_i = FLOAT_BYTES * G * (k + n) * ((k + n) * w)
    l_i_bias = FLOAT_BYTES * H
    l_h = FLOAT_BYTES * G * ((k + n) * w) * ((k + n) * w)
    l_h_bias = FLOAT_BYTES * H
    l_o = FLOAT_BYTES * G * ((k + n) * w)
    l_o_bias = FLOAT_BYTES * O

    # One (index, value) pair per nonzero weight, both held as int64: 16 bytes per
    # weight against 4 for the FP32 tensor of the same shape. `l_i + d*l_h + l_o` is
    # already that FP32 byte count, hence the divison and multiplication by the int64 size.
    coords = 2 * (l_i + d * l_h + l_o) // FLOAT_BYTES * INT64_BYTES

    bnorm_train = FLOAT_BYTES * 2 * (d + 1) * H  # bias and weight
    bnorm_state = FLOAT_BYTES * 2 * (d + 1) * H + INT64_BYTES * (d + 1)

    T = l_i + d * l_h + l_o + l_i_bias + d * l_h_bias + l_o_bias + bnorm_train
    gen = T + bnorm_state + coords  # Extra ls for mask

    grad = T
    opt = 3 * T

    act = FLOAT_BYTES * batch_size * ((d + 1) * H + O)  #  Activations for linear layers
    act += FLOAT_BYTES * batch_size * ((d + 1) * H)  # activations for batchnorm layers

    # Each forward creates a spmm tensor B x non-zero-weights and a scatter tensor B x output dim
    temp = batch_size * (l_i + d * l_h + l_o) + batch_size * (l_i_bias + d * l_h_bias + l_o_bias)

    # Each backward creates a tensor B x non-zero-weights. However, the output layers and last hidden layers
    # temp scatter tensors are deallocated beforehand, as well as activations
    temp += batch_size * (l_i + d * l_h + l_o) - batch_size * (l_h_bias + l_o_bias) - FLOAT_BYTES * batch_size * H - act

    return {
        "l_i": l_i,
        "l_i_bias": l_i_bias,
        "l_h": l_h,
        "l_h_bias": l_h_bias,
        "l_o": l_o,
        "l_o_bias": l_o_bias,
        "coords": coords,
        "bnorm_train": bnorm_train,
        "bnorm_state": bnorm_state,
        "T": T,
        "model": gen,
        "grad": grad,
        "opt": opt,
        "act": act,
        "temp": temp,
    }


def compute_cc(F: int, G: int, batch_size: int) -> dict[str, int]:
    """Causal-controller memory, in bytes."""
    latent_dim = 128
    cc_layers = (256, 512, 1024)

    # fully connected layers with batch normalization and leaky relu
    l_0 = FLOAT_BYTES * latent_dim * cc_layers[0]
    l_0_bias = FLOAT_BYTES * cc_layers[0]
    l_1 = FLOAT_BYTES * cc_layers[0] * cc_layers[1]
    l_1_bias = FLOAT_BYTES * cc_layers[1]
    l_2 = FLOAT_BYTES * cc_layers[1] * cc_layers[2]
    l_2_bias = FLOAT_BYTES * cc_layers[2]
    l_3 = FLOAT_BYTES * cc_layers[2] * (F + G)
    l_3_bias = FLOAT_BYTES * (F + G)

    # The bias terms above are already in bytes, so no FLOAT_BYTES factor here.
    bnorm_train = 2 * (l_0_bias + l_1_bias + l_2_bias)  # bias and weight
    bnorm_state = bnorm_train + INT64_BYTES * len(cc_layers)  # num_batches_tracked is int64

    cc_bytes = l_0 + l_1 + l_2 + l_3 + l_0_bias + l_1_bias + l_2_bias + l_3_bias + bnorm_train + bnorm_state
    # only outputs need to stay in memory for the next step, not the hidden layers
    act = FLOAT_BYTES * batch_size * (F + G)

    return {
        "l_0": l_0,
        "l_0_bias": l_0_bias,
        "l_1": l_1,
        "l_1_bias": l_1_bias,
        "l_2": l_2,
        "l_2_bias": l_2_bias,
        "l_3": l_3,
        "l_3_bias": l_3_bias,
        "bnorm_train": bnorm_train,
        "bnorm_state": bnorm_state,
        "model": cc_bytes,
        "grad": 0,
        "opt": 0,
        "act": act,
        "temp": 0,
    }


def compute_crit(F: int, G: int, batch_size: int) -> dict[str, int]:
    """Critic memory, in bytes."""
    crit_layers = (1024, 512, 256)

    l_0 = FLOAT_BYTES * (F + G) * crit_layers[0]
    l_0_bias = FLOAT_BYTES * crit_layers[0]
    l_1 = FLOAT_BYTES * crit_layers[0] * crit_layers[1]
    l_1_bias = FLOAT_BYTES * crit_layers[1]
    l_2 = FLOAT_BYTES * crit_layers[1] * crit_layers[2]
    l_2_bias = FLOAT_BYTES * crit_layers[2]
    l_3 = FLOAT_BYTES * crit_layers[2]
    l_3_bias = FLOAT_BYTES

    crit = l_0 + l_1 + l_2 + l_3 + l_0_bias + l_1_bias + l_2_bias + l_3_bias

    T = crit
    grad = T
    opt = 3 * T
    act = FLOAT_BYTES * batch_size * (l_0_bias + l_1_bias + l_2_bias + l_3_bias + 1)

    return {
        "l_0": l_0,
        "l_0_bias": l_0_bias,
        "l_1": l_1,
        "l_1_bias": l_1_bias,
        "l_2": l_2,
        "l_2_bias": l_2_bias,
        "l_3": l_3,
        "l_3_bias": l_3_bias,
        "model": crit,
        "T": T,
        "grad": grad,
        "opt": opt,
        "act": act,
        "temp": 0,
    }


def compute_labeler(F: int, G: int, batch_size: int) -> dict[str, int]:
    """Discriminator memory, in bytes.

    The model has both a labeler and an antilabeler. They share this architecture
    exactly, so `compute_antilabeler` returns the same figures; they are counted
    separately only so the per-part report can name them.
    """
    labeler_layers = (2000, 2000, 2000)

    l_0 = FLOAT_BYTES * G * labeler_layers[0]
    l_0_bias = FLOAT_BYTES * labeler_layers[0]
    l_1 = FLOAT_BYTES * labeler_layers[0] * labeler_layers[1]
    l_1_bias = FLOAT_BYTES * labeler_layers[1]
    l_2 = FLOAT_BYTES * labeler_layers[1] * labeler_layers[2]
    l_2_bias = FLOAT_BYTES * labeler_layers[2]
    l_3 = FLOAT_BYTES * labeler_layers[2] * F
    l_3_bias = FLOAT_BYTES * F

    # The bias terms above are already in bytes, so no FLOAT_BYTES factor here.
    bnorm_train = 2 * (l_0_bias + l_1_bias + l_2_bias)  # bias and weight
    bnorm_state = bnorm_train + INT64_BYTES * len(labeler_layers)  # num_batches_tracked is int64

    T = l_0 + l_1 + l_2 + l_3 + l_0_bias + l_1_bias + l_2_bias + l_3_bias + bnorm_train
    labeler_bytes = T + bnorm_state

    grad = T
    opt = 3 * T

    act = FLOAT_BYTES * batch_size * (l_0_bias + l_1_bias + l_2_bias + l_3_bias + F)  # activations for linear layers
    act += FLOAT_BYTES * batch_size * (l_0_bias + l_1_bias + l_2_bias)  # activations for batchnorm layers

    return {
        "l_0": l_0,
        "l_0_bias": l_0_bias,
        "l_1": l_1,
        "l_1_bias": l_1_bias,
        "l_2": l_2,
        "l_2_bias": l_2_bias,
        "l_3": l_3,
        "l_3_bias": l_3_bias,
        "bnorm_train": bnorm_train,
        "bnorm_state": bnorm_state,
        "model": labeler_bytes,
        "T": T,
        "grad": grad,
        "opt": opt,
        "act": act,
        "temp": 0,
    }


def compute_antilabeler(F: int, G: int, batch_size: int) -> dict[str, int]:
    """Antilabeler memory, in bytes. Architecturally identical to the labeler."""
    return compute_labeler(F, G, batch_size)
    all()


def compute_all(batch_size: int, F: int, G: int, k: int = K, w: int = W, d: int = D, n: int = N) -> tuple[int, int]:
    """Peak memory in bytes. Returns `(dense, sparse)` in that order.

    Both labelers are included: they are separate resident networks.
    """
    gen = compute_gen(batch_size, F, G, k, w, d, n)
    gen_sparse = compute_sparse_gen(batch_size, F, G, k, w, d, n)
    cc = compute_cc(F, G, batch_size)
    crit = compute_crit(F, G, batch_size)
    labeler = compute_labeler(F, G, batch_size)
    antilabeler = compute_antilabeler(F, G, batch_size)

    max_estimate = (
        sum(p["model"] + p["opt"] for p in [gen, cc, crit, labeler, antilabeler])
        + gen["temp"]
        + gen["act"]
        + gen["grad"]
    )
    max_sparse_estimate = (
        sum(p["model"] + p["opt"] for p in [gen_sparse, cc, crit, labeler, antilabeler])
        + gen_sparse["temp"]
        + gen_sparse["act"]
        + gen_sparse["grad"]
    )

    return max_estimate, max_sparse_estimate


def compute_compile(batch_size: int, F: int, G: int, k: int = K, w: int = W, d: int = D, n: int = N) -> int:
    """Peak memory in bytes for the compiled (`compile`) sparse implementation.

    Identical to the eager sparse estimate apart from `TEMP_FACTOR_COMPILE` on the
    generator's temporaries: everything resident (parameters, gradients, optimizer
    state, sparse coordinates) is unaffected by fusion.
    """
    gen_sparse = compute_sparse_gen(batch_size, F, G, k, w, d, n)
    cc = compute_cc(F, G, batch_size)
    crit = compute_crit(F, G, batch_size)
    labeler = compute_labeler(F, G, batch_size)
    antilabeler = compute_antilabeler(F, G, batch_size)

    return (
        sum(p["model"] + p["opt"] for p in [gen_sparse, cc, crit, labeler, antilabeler])
        + TEMP_FACTOR_COMPILE * gen_sparse["temp"]
        + gen_sparse["act"]
        + gen_sparse["grad"]
    )


def compute_smi(num_genes: int, batch_size: int = BATCH_SIZE) -> float:
    """Predicted peak `nvidia-smi` usage in bytes for a compiled run.

    nvidia-smi counts the whole device, so this is the compiled peak plus a fixed
    context/arena overhead rather than an allocator figure.
    """
    return theoretical_gib_bytes(num_genes, "compile", batch_size) + SMI_OVERHEAD_BYTES


def to_gib(num_bytes: float) -> float:
    return num_bytes / 1024**3


def to_mib(num_bytes: float) -> float:
    return num_bytes / 1024**2


def theoretical_gib_bytes(num_genes: int, variant: str, batch_size: int = BATCH_SIZE) -> float:
    """Estimated peak memory in bytes for `num_genes` under `variant`.

    `variant` is `"orig"` for the dense masked-linear implementation, `"new"` for the
    sparse one, or `"compile"` for the sparse implementation under torch.compile.
    These match the snapshot file-name infixes.
    """
    if num_genes not in SIZES:
        raise KeyError(f"no (F, G) entry for {num_genes} genes; add one to SIZES")
    F, G = SIZES[num_genes]
    if variant == "orig":
        return compute_all(batch_size, F, G)[0]
    if variant == "new":
        return compute_all(batch_size, F, G)[1]
    if variant == "compile":
        return compute_compile(batch_size, F, G)
    raise ValueError(f"unknown variant {variant!r}; expected 'orig', 'new' or 'compile'")


def theoretical_gib(num_genes: int, variant: str, batch_size: int = BATCH_SIZE) -> float:
    """Estimated peak memory in GiB for `num_genes` under `variant`."""
    return to_gib(theoretical_gib_bytes(num_genes, variant, batch_size))


def theory_curves(variant: str, batch_size: int = BATCH_SIZE) -> tuple[list[int], list[float]]:
    """`(genes, GiB)` for every size in `SIZES`, for plotting a model curve."""
    genes = sorted(SIZES)
    return genes, [theoretical_gib(num_genes, variant, batch_size) for num_genes in genes]
