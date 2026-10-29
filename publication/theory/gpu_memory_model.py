"""Analytic GPU-memory model for the causal GAN.

Shared by the `plot-*.py` scripts next to it and `../plot-results.py` (which
overlays the estimate on the measured peaks of the experiments). Keeping the
formulas in one place matters because these estimates are the paper's claim:
two copies would inevitably drift.

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
identical -- the same network over the same inputs -- and both are resident at
the peak, so `compute_labeler` is counted twice.

`compute_all` returns `(dense, sparse)` in that order -- dense first, matching
`compute_gen`, which models `upstream/src/layers/masked_linear.py`. Dense mode
forms `weight * mask` as a full dense FP32 copy on every forward and again in
backward, so the masked entries cost exactly as much as the live ones. Sparse
mode keeps only the nonzero coordinates.

Every `compute_*` of a network returns the bytes of its parameters and buffers
(`model`), gradients (`grad`), optimizer state (`opt`), activations (`act`) and
temporary tensors (`temp`).
"""

from __future__ import annotations

FLOAT_BYTES = 4
INT64_BYTES = 8


# Batch-sized FP32 tensors alive at the compiled peak, counted in an allocator snapshot of the 10000-gene compiled run
# of the former memory-profiling experiment (G = 9183, F = 817, k = 15, w = 1, d = 2, n = 1).
# Inductor fuses the gather, multiply and scatter of each sparse layer, so no B x nnz product is stored. What remains
# at the peak (backward pass through the last hidden layer) is:
#
# * B x H: the d + 1 pre-batch-norm outputs saved by the forward, d + 1 zero-initialised gradient buffers for the
#   inputs of the output and hidden layers, and 2 buffers of the batch-norm/ReLU backward of the last hidden layer,
# * B x (F + G): the real batch, the generated cells before LSN, the LSN output and the critic's input gradient,
# * B x G: the gradient of the output layer.
#
# Only d = 2 was profiled; the dependence of the B x H count on d follows from the structure of the generator.
def compiled_tensor_counts(d: int) -> tuple[int, int, int]:
    """`(alpha, gamma, delta)`: numbers of B x H, B x (F+G) and B x G tensors at the compiled peak."""
    return 2 * d + 4, 4, 1


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
    opt = 3 * T  # two Adam moments plus AMSGrad, all FP32

    act = FLOAT_BYTES * batch_size * ((d + 1) * H + O)  # activations for linear layers
    act += FLOAT_BYTES * batch_size * ((d + 1) * H)  # activations for batchnorm layers

    # In `upstream/src/layers/masked_linear.py`, every forward computes `weight * mask` and saves that resulting dense tensor for backward.
    # The mask zeros most entries, but multiplication still creates a full-sized dense FP32 copy.
    temp = l_i + d * l_h + l_o

    # The temporary tensors from training the critic are still present when training the generator, so we need to count them twice.
    temp += l_i + d * l_h + l_o

    # During backward, `MaskedLinearFunction.backward` first forms a dense gradient matrix with a matrix multiply, then multiplies it by the mask.
    # That creates another full-matrix temporary.
    temp += l_i + d * l_h + l_o
    return {"model": gen, "grad": grad, "opt": opt, "act": act, "temp": temp}


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
    # already that FP32 byte count, hence the division and multiplication by the int64 size.
    coords = 2 * (l_i + d * l_h + l_o) // FLOAT_BYTES * INT64_BYTES

    bnorm_train = FLOAT_BYTES * 2 * (d + 1) * H  # bias and weight
    bnorm_state = FLOAT_BYTES * 2 * (d + 1) * H + INT64_BYTES * (d + 1)

    T = l_i + d * l_h + l_o + l_i_bias + d * l_h_bias + l_o_bias + bnorm_train
    gen = T + bnorm_state + coords  # Extra ls for mask

    grad = T
    opt = 3 * T

    act_linear = FLOAT_BYTES * batch_size * ((d + 1) * H + O)  #  Activations for linear layers
    act_bnorm = FLOAT_BYTES * batch_size * ((d + 1) * H)  # activations for batchnorm layers
    act = act_linear + act_bnorm

    # Each forward creates a spmm tensor B x non-zero-weights and a scatter tensor B x output dim
    temp = batch_size * (l_i + d * l_h + l_o) + batch_size * (l_i_bias + d * l_h_bias + l_o_bias)

    # Before reaching the peak of the network, the output layers temp tensors, the batchnorm activations and last
    # hidden layers temp scatter tensors are deallocated
    temp -= batch_size * (l_o + l_o_bias) + FLOAT_BYTES * batch_size * H + batch_size * l_h_bias

    # linear layer activations are not actually kept in memory for backpropagation
    temp -= act_linear

    # The peak occurs during backward pass through the last hidden layer. Here, additional temp spmm tensors are created in each sparse layer
    temp += batch_size * (l_i + d * l_h)

    return {"model": gen, "grad": grad, "opt": opt, "act": act, "temp": temp}


def compute_cc(
    F: int,
    G: int,
    batch_size: int,
    latent_dim: int = 128,
    cc_layers: tuple[int, int, int] = (256, 512, 1024),
) -> dict[str, int]:
    """Causal-controller memory, in bytes. The defaults are the architecture of the main configuration."""

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

    return {"model": cc_bytes, "grad": 0, "opt": 0, "act": act, "temp": 0}


def compute_crit(
    F: int, G: int, batch_size: int, crit_layers: tuple[int, int, int] = (1024, 512, 256)
) -> dict[str, int]:
    """Critic memory, in bytes. The default is the architecture of the main configuration."""

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

    return {"model": crit, "grad": grad, "opt": opt, "act": act, "temp": 0}


def compute_labeler(
    F: int, G: int, batch_size: int, labeler_layers: tuple[int, int, int] = (2000, 2000, 2000)
) -> dict[str, int]:
    """Labeler memory, in bytes, and also that of the identical antilabeler. The default is the architecture of the
    main configuration."""

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

    return {"model": labeler_bytes, "grad": grad, "opt": opt, "act": act, "temp": 0}


# Keyword arguments of compute_cc, compute_crit and compute_labeler that describe the networks other than the
# generator, e.g. {"latent_dim": 64, "cc_layers": (224, 752, 912), "crit_layers": (752, 368, 176),
# "labeler_layers": (992, 496, 256)} for the Arabidopsis configuration. Missing keys take the main configuration.
Architecture = dict[str, object]


def other_networks(F: int, G: int, batch_size: int, architecture: Architecture | None = None) -> list[dict[str, int]]:
    """`[cc, crit, labeler, antilabeler]` for the given architecture (the main configuration by default)."""
    a = architecture or {}
    cc_args = {key: a[key] for key in ("latent_dim", "cc_layers") if key in a}
    crit_args = {key: a[key] for key in ("crit_layers",) if key in a}
    labeler_args = {key: a[key] for key in ("labeler_layers",) if key in a}
    labeler = compute_labeler(F, G, batch_size, **labeler_args)
    return [compute_cc(F, G, batch_size, **cc_args), compute_crit(F, G, batch_size, **crit_args), labeler, labeler]


def resident_grads(crit: dict[str, int], labeler: dict[str, int], antilabeler: dict[str, int]) -> int:
    """Gradients of the critic, labeler and antilabeler, which are held during the generator step.

    Each of these networks frees its gradients only with `zero_grad(set_to_none=True)` at the start of its own next
    training step, so the gradients of its last update are still allocated when the generator is trained. The
    generator's backward pass also runs through these networks and accumulates into the same gradients. The causal
    controller is frozen and has no gradients.
    """
    return crit["grad"] + labeler["grad"] + antilabeler["grad"]


def compute_all(
    batch_size: int,
    F: int,
    G: int,
    k: int,
    w: int,
    d: int,
    n: int,
    architecture: Architecture | None = None,
) -> tuple[int, int]:
    """Peak memory in bytes. Returns `(dense, sparse)` in that order.

    Both labelers are included: they are separate resident networks. `architecture` describes the networks other
    than the generator (see `Architecture`), the main configuration by default.
    """
    gen = compute_gen(batch_size, F, G, k, w, d, n)
    gen_sparse = compute_sparse_gen(batch_size, F, G, k, w, d, n)
    cc, crit, labeler, antilabeler = other_networks(F, G, batch_size, architecture)

    max_estimate = (
        sum(p["model"] + p["opt"] for p in [gen, cc, crit, labeler, antilabeler])
        + resident_grads(crit, labeler, antilabeler)
        + gen["temp"]
        + gen["act"]
        + gen["grad"]
    )
    max_sparse_estimate = (
        sum(p["model"] + p["opt"] for p in [gen_sparse, cc, crit, labeler, antilabeler])
        + resident_grads(crit, labeler, antilabeler)
        + gen_sparse["temp"]
        + gen_sparse["act"]
        + gen_sparse["grad"]
    )

    return max_estimate, max_sparse_estimate


def compute_compile(
    batch_size: int,
    F: int,
    G: int,
    k: int,
    w: int,
    d: int,
    n: int,
    architecture: Architecture | None = None,
) -> int:
    """Peak memory in bytes for the compiled (`compile`) sparse implementation.

    The persistent terms are those of the sparse peak: parameters and optimizer state of every network, the
    generator's gradients and the gradients held by the critic and both labelers. The temporary tensors and
    activations are `alpha` FP32 tensors of size B x H, `gamma` of size B x (F + G) and `delta` of size B x G, counted
    in an allocator snapshot of a compiled run (see `compiled_tensor_counts`). Smaller tensors are omitted.
    `architecture` describes the networks other than the generator (see `Architecture`).
    """
    gen_sparse = compute_sparse_gen(batch_size, F, G, k, w, d, n)
    cc, crit, labeler, antilabeler = other_networks(F, G, batch_size, architecture)

    H = G * (k + n) * w
    alpha, gamma, delta = compiled_tensor_counts(d)
    temporaries = FLOAT_BYTES * batch_size * (alpha * H + gamma * (F + G) + delta * G)

    return (
        sum(p["model"] + p["opt"] for p in [gen_sparse, cc, crit, labeler, antilabeler])
        + resident_grads(crit, labeler, antilabeler)
        + gen_sparse["grad"]
        + temporaries
    )
