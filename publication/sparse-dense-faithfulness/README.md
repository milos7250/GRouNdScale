# Sparse vs dense masked layers: numerical faithfulness

Checks that GRouNdScale's sparse masked layers (`sparselinear.SparseLinear`) compute the same function as GRouNdGAN's
dense `MaskedLinear` (loaded unchanged from the pinned upstream checkout, `upstream/` at 2df087f), in FP32 and TF32.

For 32 log-spaced GRN sizes (G = 10 to 2500 targets, F = 63 candidate TFs, k = 15 regulators per target, depth 2,
width 1, batch 1024, as in the main experiment) and 4 random GRNs, weight initialisations and inputs per size, the
script builds GRouNdScale's `CausalGenerator` and a dense twin with identical weights, then compares:

- **layer by layer**: each masked layer receives the same input and upstream gradient; output, input gradient, weight
  gradient (at the masked-in positions) and bias gradient are compared;
- **end to end**: the whole masked network (masked layers, batch normalisation in training mode, ReLU) on the same TF
  expressions and noise, with a fixed random linear loss;
- both against an FP64 reference (sparse network in float64; the dense network in float64 is also checked against it
  for widths up to 8000).

Masked-out positions: the dense layer's full weight gradient must have no non-zeros outside the mask; the sparse
layer's parameters must cover exactly the masked-in positions implied by the GRN (counted independently).

End to end, a pre-activation within rounding error of zero can fall on different sides of a ReLU kink in the two
implementations. Its gradient then switches between its full value and zero, which dominates the maximum gradient
deviation without indicating a difference between the layers. These flips are counted (`relu_flips`), and the
end-to-end deviations are also summarised over the trials without a flip (`end_to_end_same_relu_pattern`).

Relative deviations (`max_rel`) are the maximum absolute deviation divided by the largest magnitude of the reference
tensor; `rel_l2` is the relative 2-norm of the difference.

## Run

```bash
cd publication/sparse-dense-faithfulness
sbatch run.sh                       # A100, apptainer image milos7250/groundscale:latest
python faithfulness.py summarize    # or inside the container; writes results/summary.json
```

A resubmission resumes `results/records.jsonl`. Sizes whose dense twin does not fit on the GPU are recorded as `oom`
and larger sizes are skipped (the largest size, G = 2500, peaks at 48.9 GiB).

## Results

Slurm job 39379564 (2026-11-10, about 3 minutes): NVIDIA A100-SXM4-80GB, torch 2.14.0+cu132, CUDA 13.2, torch_sparse
0.6.18, upstream GRouNdGAN 2df087f. Full numbers in `results/summary.json`.

All 128 trials (32 sizes x 4 seeds) ran in both precisions, covering G = 10 to 2500 targets, hidden widths 160 to
40,000 and mask densities 0.04 % to 24.6 %. The calibration matmul had a relative error of 5.1e-7 under the FP32
setting and 2.9e-4 under the TF32 setting, so TF32 was in effect.

### Layer by layer

Maximum relative deviation over 512 layer comparisons per precision (maximum absolute deviation in parentheses).

| Quantity | FP32 sparse vs dense | FP32 sparse vs FP64 | FP32 dense vs FP64 | TF32 sparse vs dense | TF32 sparse vs FP64 | TF32 dense vs FP64 |
|---|---|---|---|---|---|---|
| Output | 3.3e-7 (7.2e-7) | 2.5e-7 | 2.4e-7 | 6.9e-4 (1.1e-3) | 2.4e-7 | 6.9e-4 |
| Input gradient | 4.6e-6 (2.0e-5) | 4.3e-6 | 1.3e-6 | 7.4e-4 (1.7e-3) | 4.2e-6 | 7.4e-4 |
| Weight gradient | 1.8e-6 (2.5e-4) | 1.7e-7 | 1.8e-6 | 5.1e-4 (5.5e-2) | 1.7e-7 | 5.1e-4 |
| Bias gradient | 0 (bitwise identical) | 1.9e-7 | 1.9e-7 | 0 (bitwise identical) | 1.9e-7 | 1.9e-7 |

- FP32: the two implementations differ by rounding only, at the level of their own differences from FP64. The forward
  deviation does not grow with network size (2e-7 to 3e-7 at every G). The input gradient deviates most in the input
  layer, where each TF fans out to about 10^4 hidden units and the sparse scatter accumulation is less accurate than the
  dense GEMM; for the weight gradient the dense GEMM is the less accurate of the two.
- TF32: the whole deviation comes from the dense layer. `sparselinear` computes its products with `index_select` and
  `scatter_add` rather than a matrix multiplication, so TF32 does not affect it and it stays at FP32 accuracy.
- Absolute deviations scale with the magnitude of each quantity; weight gradients reach about 100.

### Masked-out positions

- The dense layers' full weight gradients had no non-zero entry outside the mask: 0 among 9.08e10 masked-out positions
  checked per precision (each position once in the layer-by-layer and once in the end-to-end backward).
- In all 128 trials the sparse layers' parameters covered exactly the connections implied by the GRN (counted
  independently), so no masked-out position has a parameter that could receive a gradient.
- The dense weights were exactly zero outside the mask in all trials.

### FP64

In the 88 trials with hidden width up to 8000 (G up to 421), the dense and sparse networks in float64 agreed to a
relative deviation of 9.9e-16 in the output and 3.4e-15 in the input gradient, and to 1.7e-13 absolute in the weight
gradients.

### End to end

| Quantity | FP32 sparse vs dense | TF32 sparse vs dense |
|---|---|---|
| Output | 6.1e-7 (8.3e-7) | 1.3e-3 (1.9e-3) |
| Input gradient, all trials | 1.2e-1 | 5.8e-1 |
| Weight gradient, all trials | 4.5e-2 | 3.4e-1 |
| Input gradient, trials without a ReLU flip | 2.1e-6 | none without a flip |
| Weight gradient, trials without a ReLU flip | 1.1e-6 | none without a flip |
| Batch normalisation gradient, trials without a ReLU flip | 7.9e-7 | none without a flip |

- The end-to-end outputs agree as closely as single layers. The gradient maxima are set by ReLU flips: units whose
  pre-activation is within rounding of zero and lands on different sides of the kink in the two implementations.
- FP32: 236 flips between sparse and dense among 3.07e9 unit evaluations, against 197 between sparse and FP64 and 165
  between dense and FP64, so flips are a property of FP32 evaluation rather than of the sparse layers. 57 of the 128
  trials had none, and in those the gradients agree to about 1e-6.
- TF32: 532,754 flips between sparse and dense, almost all from the dense side (532,763 between dense and FP64, 179
  between sparse and FP64).
- The biases of layers followed by batch normalisation have an exact gradient of zero in training mode; both
  implementations return rounding noise of at most 3.8e-5, which is not counted as a relative deviation.
