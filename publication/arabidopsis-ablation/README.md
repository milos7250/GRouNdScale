# arabidopsis-ablation: which change gives which speed-up

Status: **run and collected (steps 0-6), paper updated 2026-10-29 (step 7: new subsection `sec:ablation` with `tab:arabidopsis-ablation`, Sparse Linear Layers and Model Compilation rewritten)**, see "Results" below. The instructions below are written for a Claude Code
session on the cluster login node (repository at `CODE_ROOT=/mnt/shared/scratch/mmicik/private/Geneformer/simulation/GRouNdScale`).
Work through the steps in order, record every deviation from the plan in the "Deviations" section at the end of this
file, and ask the user before changing the scope (variants, steps, GPU type).

## Goal

The Methods section of the paper (`paper.tex`, Sections "Sparse Linear Layers" and "Model Compilation") states how
much each improvement sped up Arabidopsis training (currently "over 6 times faster and 43 % less GPU memory" for the
sparse layers together with the dependency update, and "an additional 1.5 times speed-up" for compilation). Those
numbers come from old, unrecorded runs. This experiment re-measures them under one protocol, separating the
environment, the code update, the sparse layers, compilation and TF32 matmuls, so that every claim has a measured source.

Each variant trains the paper's Arabidopsis configuration (`paper-arabidopsis/causal_gan.cfg`, hyperparameters in the
paper's `tables/arabidopsis.cfg`) for 20 steps of the causal controller (CC) and 20 steps of the causal GAN, and records
step time and GPU memory of both phases, exactly as `publication/target-gene-num(-upstream)` does. Every variant is
measured once (user's decision, 2026-10-28), all on one node.

## Variants

The variants of the current code run the **current** GRouNdScale code (`HEAD` of the repository at the time of the
run), so that later improvements of the training loop, data loading and logging are present in all of them and do not
skew the comparison. They are all combinations of generator layers x compilation x matmul precision.

| Id | Code | Image | Generator layers | Compile | Matmuls | Entry point |
|---|---|---|---|---|---|---|
| `a-upstream` | `upstream/` at `2df087f` | original (Python 3.9, torch 1.13) | dense masked | no | FP32 (torch 1.13 default) | `../profile-memory-upstream.py` |
| `a2-upstream-newenv` | `upstream/`, unchanged, + `a2-pydeps` (tabulate, tensorflow) | groundscale (Python 3.12, torch 2.14) | dense masked | no | FP32 (torch default) | `../profile-memory-upstream.py` |
| `dense-eager-fp32` | `HEAD` + `dense_generator.py` | groundscale | dense masked (`MaskedLinear`) | no | FP32 | `launch.py` |
| `dense-eager-tf32` | `HEAD` + `dense_generator.py` | groundscale | dense masked | no | TF32 | `launch.py` |
| `dense-compiled-fp32` | `HEAD` + `dense_generator.py` | groundscale | dense masked | yes | FP32 | `launch.py` |
| `dense-compiled-tf32` | `HEAD` + `dense_generator.py` | groundscale | dense masked | yes | TF32 | `launch.py` |
| `sparse-eager-fp32` | `HEAD` | groundscale | sparse (`SparseLinear`) | no | FP32 | `launch.py` |
| `sparse-eager-tf32` | `HEAD` | groundscale | sparse | no | TF32 | `launch.py` |
| `sparse-compiled-fp32` | `HEAD` | groundscale | sparse | yes | FP32 | `launch.py` |
| `sparse-compiled-tf32` | `HEAD` | groundscale | sparse | yes | TF32 (the current default) | `launch.py` |

`a2-upstream-newenv` and the two `dense-compiled-*` variants may not run at all (the original code has never run in
the new environment; `MaskedLinear` is a custom autograd function that `torch.compile(fullgraph=True)` has to trace).
Their failure is recorded and reported, it does not cancel the other tasks.

What the comparisons isolate:

- **a-upstream -> a2-upstream-newenv**: the environment alone (Python, torch and all dependencies), same code, FP32 in
  both.
- **a2-upstream-newenv -> dense-eager-fp32**: every change of GRouNdScale outside the generator layers, compilation and
  TF32: the refactored training loop (e.g. `zero_grad(set_to_none=True)`), data loading, pre-allocated tensors,
  logging. If a2 fails, only a-upstream -> dense-eager-fp32 (environment and code update together) is available; say so
  in the paper.
- **dense -> sparse** (same compile and matmuls): the sparse layers alone.
- **eager -> compiled** (same layers and matmuls): compilation alone.
- **fp32 -> tf32** (same layers and compile): TF32 matmuls alone.
- The path of the paper's text: a-upstream -> dense-eager-tf32 (environment, code update and TF32) -> sparse-eager-tf32
  (sparse layers) -> sparse-compiled-tf32 (compilation).

### How the variants of the current code work

`launch.py` runs `../profile-memory.py` in the same process with the same arguments, after two in-process patches that
leave `src/` untouched:

- `GENERATOR_LAYERS=dense` imports `dense_generator.py`, which replaces `CausalGenerator._create_generator` and
  `_create_generator_block`. Each block uses `layers.masked_linear.MaskedLinear`, the dense masked layer of GRouNdGAN
  that is still in `src/` (functionally identical to `2df087f:src/layers/masked_linear.py`), built from `mask.toarray()`
  of the current CSR masks. As in GRouNdGAN, the hidden layers share one dense mask tensor (the CSR hidden mask is
  converted once per generator) and the weights are initialised with Xavier uniform / Kaiming normal and then masked.
  With the Arabidopsis configuration the hidden mask is H x H with H = G(k+1)w = 18496, about 1.3 GiB in 32-bit floats.
- The current code calls `torch.set_float32_matmul_precision("high")` (TF32) at the start of training. `TF32=0` turns
  that call into a call with `"highest"` (full FP32 matmuls); `TF32=1` leaves it alone.

`launch.py` also adds `generator_layers`, `float32_matmul_precision` and `cuda_matmul_allow_tf32` (read when the run
ends) and `python` to `settings` in `summary.json`, so every summary shows the precision that was actually in effect.

## Steps

### 0. Prerequisites

- Use one A100 80 GB node for all variants (user's decision), with the `#SBATCH` resources of
  `../target-gene-num-upstream/run.sh` (`--gpus=a100:1`, `--cpus-per-task=16`, `--mem=64G`), four tasks at a time.
  Pick a node with 4 free A100s with `sinfo` right before submitting and set `--nodelist` in `run.sh` (or override it on
  the `sbatch` command line).
- Data: `paper-arabidopsis/data/Col0_0h_scVI_{train,val,test}.h5ad` and `paper-arabidopsis/data/causal_graph.pkl`
  (422 TFs, 578 targets, k = 15). Do not re-run pre-processing or GRN creation. Both profilers expect
  `DATA_DIR/{train,validation,test}.h5ad`, so `data/` in this directory holds symlinks
  `train.h5ad -> ../../paper-arabidopsis/data/Col0_0h_scVI_train.h5ad` (likewise `validation`, `test`) and a copy of
  `causal_graph.pkl`. The symlinks resolve inside `$CODE_ROOT`, which the jobs bind.
- Both images are pinned by digest in `run.sh`; the digests and the `HEAD` commit are in the setup record. Do not commit
  to `src/` while the array runs: all variants of the current code must run the same commit (`settings.git` in every
  `summary.json` shows it).

### 1. Profiler change

`../profile-memory.py` writes `mode` as `compiled` or `eager`. One backward-compatible line lets runs carry the variant
id: `mode=os.environ.get("PROFILE_MODE") or ("compiled" if setup["compiled"] else "eager")`. Nothing else in the
profilers changes. `a-upstream` and `a2-upstream-newenv` use `../profile-memory-upstream.py` exactly as
`target-gene-num-upstream` does (its `mode` is always `upstream`; the run name is the variant id).

### 2. Config templates

Two templates in this directory, both filled in from the environment by the profilers. They contain the values of the
paper's `tables/arabidopsis.cfg` with the step counts reduced to 20:

- `[Preprocessing] library size = 1_000`; `[Data]` `${DATA_DIR}/{train,validation,test}.h5ad`,
  `number of genes = ${NUM_GENES}`, `causal graph = ${CAUSAL_GRAPH}`
- `[Model]` noise 1, depth 3, width 2, critic layers `752 368 176`, labeler layers `992 496 256`, latent dim 64,
  lambda 12
- `[Training]` batch size 2350, critic iterations 20, labeler and antilabeler training intervals 3,
  **maximum steps 20**, optimiser betas and learning rates of `tables/arabidopsis.cfg`
- `[CC Model]` generator layers `224 752 912`, critic layers `816 384 192`, latent dim 64, lambda 20
- `[CC Training]` batch size 358, critic iterations 8, **maximum steps 20**, CC betas and learning rates as above
- logging frequencies large enough that nothing is logged, plotted or saved inside the 20 steps, except the controller
  checkpoint at its last step, which the generator phase loads (`[CC Logging] save frequency = 20`)
- the same `random seed` in both templates

Files:

- `causal_gan.cfg` for the current code, with `compile modules = ${COMPILE_MODULES}`.
- `causal_gan_upstream.cfg` for the original code (a, a2), in the original format (no `compile modules`).

### 3. `run.sh`

A Slurm array with one task per variant (`--array=1-10%4`), in the order of the table above, modelled on
`../common.sh` and `../target-gene-num-upstream/run.sh`:

- `a-upstream`: `python3.9 ../profile-memory-upstream.py --template causal_gan_upstream.cfg ...` in the original image.
- `a2-upstream-newenv`: the same command with `python` in the groundscale image.
- `<layers>-<compile>-<matmuls>`: `GENERATOR_LAYERS=<layers> COMPILE_MODULES=<True|False> TF32=<0|1>
  PROFILE_MODE=<variant> python launch.py --config causal_gan.cfg ...` in the groundscale image.
- `APPTAINER_BIND` contains `$CODE_ROOT` and the real path of `upstream/`.
- Outputs to `results/<variant>/` (`summary.json`, `trace.json`, `nvidia-smi.csv`, the rendered `causal_gan.cfg`);
  everything else is deleted unless `KEEP_OUTPUTS=1` (also the `<OUT_DIR>_CC` directory that the original code writes).
- `scripts/common.sh` provides the cache settings; the compile cache is per job, so every compiled run compiles from
  scratch (the warm-up steps absorb it).
- An `oom` does not cancel the array (exit 0, as in the other experiments), nor does any failure of `a2-upstream-newenv`,
  `dense-compiled-fp32` and `dense-compiled-tf32`. Any other error cancels the pending tasks.

### 4. Things to keep in mind

- The un-compiled runs trace the TensorBoard graph before training (`log_tensorboard_graph`), which is in their phase
  peak, as in the other experiments. Report the steady-state torch peaks as well, which exclude it. The original code does not
  trace the graph.
- The nvidia-smi peak of the compiled runs contains the constant compile offset (about 6.2 GiB on an A100, see
  `../compile-overhead/`). Compare torch allocated as well as nvidia-smi, and do not attribute that offset to the model.
- With one measurement per variant there is no repeat to compare against. The steady-state step time is the mean of
  steps 5..20 (16 steps per phase); look at the per-step times in `trace.json` for outliers before reporting.
- Four tasks share the node's CPUs (4 x 16 of 128) and host memory bandwidth. All variants see the same kind of
  neighbours, but the order in which they run is not controlled.

### 5. Submit

```bash
cd "$CODE_ROOT/publication/arabidopsis-ablation"
sinfo -p gpu -N -O "NodeList:30,StateCompact:10,Gres:40,GresUsed:50"   # a node with gpu:a100:0 used, state idle
sbatch run.sh                                                          # or: sbatch --nodelist=<node> run.sh
```

Check each `summary.json`: `status` (`ok`, or the recorded error for the variants that may fail), F = 422, G = 578,
k = 15, B = 2350, d = 3, w = 2, the expected commit in `settings.git`, the expected Python/torch version, the expected
`float32_matmul_precision` (`highest` for fp32, `high` for tf32) and `generator_layers`, and plausible step counts
(21 steps per phase, steps 0..20). Expect `a-upstream` to be by far the slowest in the GAN phase (seconds per step). It
should fit in 80 GB: the dense layers here are smaller than in the PBMC "big" model that the original code trained in
77.5 GB (H = G(k+1)w = 18496 here, 28110 there).

### 6. Collect

Write `collect.py` (or extend `../collect-results.py` with this experiment) to produce `results.csv` with one row per
(variant, phase): status, `average_step_time_ms`, `peak_torch_allocated_mib`, `peak_torch_reserved_mib`,
`peak_nvidia_smi_mib`, and the steady-state peaks. For the generator and controller phases add:

- speed-up and memory ratio of each comparison listed under "What the comparisons isolate", and relative to
  `a-upstream`;
- extrapolated training time in hours for the paper's step counts (CC 200,000 steps, GAN 500,000 steps).

Add a "Results" section to this README with the summary table (as `publication/README.md` does), the container
digests, the commits, the failed variants with their errors, and any OOM. Sanity checks before reporting:

- `sparse-compiled-tf32` GAN step time should be in the range of the full run in `tables/arabidopsis-runtime.json` of
  the paper (`gan_time_per_step` 1477 ms, which also contains logging); a large mismatch means the configuration
  differs.
- The dense variants should need far more memory than the sparse ones, in line with the dense model of the paper's
  analytical section (`eq:peak-dense`); a large discrepancy between `dense-eager-fp32` and `a-upstream` /
  `a2-upstream-newenv` memory beyond the training-loop changes means the patch does not reproduce GRouNdGAN's layers
  (e.g. masks not shared).
- TF32 should mainly speed up the dense variants (large dense matmuls); a large TF32 effect on the sparse generator
  would need an explanation (the critic and labelers are dense in every variant).

### 7. Update the paper

The paper is in `~/Documents/GRouNdGAN Paper` on the user's laptop (not in this repository). Compile it with the latex
MCP server (`compile_latex` on `paper.tex`, `read_build_log` for warnings), not with latexmk directly. House style (see the project preferences): `\qty{...}{MiB}` / `\qty{...}{GiB}` inside inline math,
`\qty{...}{\percent}`, hours as `h`, `\num{}` only where thousands separators are needed (plain numbers otherwise),
`\tabref`/`\secref`/`\figref`, British spelling, "GRouNdGAN" for the original and "GRouNdScale" for the new
implementation.

Places to update in `paper.tex`:

1. **Section "Sparse Linear Layers"** (`\label{sec:sparse}`), the paragraph ending "...became over 6 times faster and
   used 43 % less GPU memory". Split it: the environment update (a -> a2, if a2 ran), the code update
   (a2 -> dense-eager-fp32, or a -> dense-eager-fp32 together with the environment), then the sparse layers alone
   (dense -> sparse). Keep the sentence that the sparse layers required the dependency update. State the combined
   effect only if it is useful.
2. **Section "Model Compilation"** (`\label{sec:compilation}`), "an additional 1.5 times speed-up": replace by
   eager -> compiled of the sparse model, per phase, and give the memory change in torch allocated memory, pointing to
   `\secref{sec:device-memory}` for the nvidia-smi offset. Mention the TF32 effect (fp32 -> tf32) where it fits. The
   second paragraph (the target-gene sweep at 11000 targets) stays.
3. **New table** with the ablation (rows: the variants; columns: CC step time (ms), GAN step time (ms),
   GAN torch peak (MiB), GAN nvidia-smi peak (MiB)), placed in the Methods section before "Sparse Linear Layers" or
   right after "Model Compilation", label `tab:arabidopsis-ablation`, referenced from both sections. Caption:
   configuration (Arabidopsis, \(F = 422\), \(G = 578\), \(k = 15\), \(B = 2350\), \(d = 3\), \(w = 2\)), 20 steps per
   phase, one run per variant, A100 80 GB, step time excludes the first 5 steps.
4. **Section "Investigation on the Arabidopsis thaliana Dataset"** (`\label{sec:arabidopsis}`): optionally one sentence
   pointing to the table, e.g. that training this configuration with GRouNdGAN would take X h instead of Y h.
5. **Summary** (`\label{sec:summary}`): only if the numbers change the story (e.g. if the environment update alone is
   a large part of the speed-up, say so).
6. **Section "Hyperparameter Optimisation"**: "hours/few days instead of weeks" should agree with the extrapolated
   times of a-upstream and sparse-compiled-tf32.

Then update the project doc `claude/scaling-chapters-update.md` (or a new `claude/arabidopsis-ablation.md`) with the
summary table, the commits and the container digests.

## Results

Jobs `37373475` (tasks 2-5) and `37373509` (tasks 1, 6-10), 2026-10-28, all on `n23-64-512-aragog` (A100-SXM4-80GB),
four at a time. GRouNdScale `b3ba67b` (`src_modified: false` in every run of the current code), images as in the setup
record. One run per variant; step times are the mean of steps 5..20. `results.csv` and `results-summary.csv` are written
by `collect.py`. All summaries have F = 422, G = 578, k = 15, B = 2350, d = 3, w = 2, 21 steps per phase, and the
expected `float32_matmul_precision` (`highest` / `high`) and `generator_layers`.

| Variant | CC step (ms) | GAN step (ms) | GAN torch peak (MiB) | GAN nvidia-smi peak (MiB) | GAN 500k steps (h) |
|---|---:|---:|---:|---:|---:|
| `a-upstream` | 104 | 7443 | 35171 | 37927 | 1034 |
| `a2-upstream-newenv` | 61 | 7154 | 32641 | 35077 | 994 |
| `dense-eager-fp32` | 41 | 7084 | 25644 | 27645 | 984 |
| `dense-eager-tf32` | 42 | 1561 | 25644 | 27645 | 217 |
| `dense-compiled-fp32` | 39 | 6524 | 23323 | 29881 | 906 |
| `dense-compiled-tf32` | 38 | 1229 | 23325 | 29873 | 171 |
| `sparse-eager-fp32` | 48 | 1915 | 35686 | 43113 | 266 |
| `sparse-eager-tf32` | 45 | 1884 | 35686 | 43113 | 262 |
| `sparse-compiled-fp32` | 43 | 1202 | 1868 | 7695 | 167 |
| `sparse-compiled-tf32` | 40 | 1152 | 1871 | 7521 | 160 |

The steady-state torch peaks of the GAN phase equal the phase peaks (the TensorBoard graph is not the peak). CC
extrapolations (200k steps): 5.8 h for `a-upstream`, 2.1-2.7 h for the current code.

GAN phase, speed-up (baseline / variant step time) and torch peak ratio (variant / baseline):

| Comparison | Speed-up | Torch peak |
|---|---:|---:|
| environment, FP32 (a-upstream -> a2-upstream-newenv) | 1.04 | 0.93 |
| code update, FP32 (a2-upstream-newenv -> dense-eager-fp32) | 1.01 | 0.79 |
| environment + code update, FP32 (a-upstream -> dense-eager-fp32) | 1.05 | 0.73 |
| TF32, dense eager / dense compiled | 4.54 / 5.31 | 1.00 / 1.00 |
| TF32, sparse eager / sparse compiled | 1.02 / 1.04 | 1.00 / 1.00 |
| sparse layers, eager FP32 / eager TF32 | 3.70 / 0.83 | 1.39 / 1.39 |
| sparse layers, compiled FP32 / compiled TF32 | 5.43 / 1.07 | 0.08 / 0.08 |
| compilation, dense FP32 / dense TF32 | 1.09 / 1.27 | 0.91 / 0.91 |
| compilation, sparse FP32 / sparse TF32 | 1.59 / 1.64 | 0.05 / 0.05 |
| paper path: a-upstream -> dense-eager-tf32 | 4.77 | 0.73 |
| paper path: dense-eager-tf32 -> sparse-eager-tf32 | 0.83 | 1.39 |
| paper path: sparse-eager-tf32 -> sparse-compiled-tf32 | 1.64 | 0.05 |
| all: a-upstream -> sparse-compiled-tf32 | 6.46 | 0.05 (nvidia-smi 0.20) |

Findings:

- The environment and code update together barely change the GAN step time at FP32 (1.05x) but lower the torch peak by
  27 %. Separately: the new environment gives 1.04x and 7 % less torch memory (a -> a2), the code update 1.01x and 21 %
  less (a2 -> dense-eager-fp32). Most of the speed-up of the dense model comes from TF32 (4.5-5.3x), which the current code enables and the
  original code (torch 1.13) does not. TF32 hardly matters for the sparse model (1.02-1.04x).
- Eager sparse layers are faster than dense ones only without TF32 (3.7x at FP32, 0.83x with TF32), and they need
  **more** memory: the generator update of the eager sparse model peaks at 34.9 GiB (critic iterations 10.7 GiB), against
  25.0 GiB (22.8 GiB) for the dense model, whose persistent weights, gradients and optimiser state are about 21 GiB.
- Compilation is what makes the sparse model small: 35.7 GiB -> 1.9 GiB torch peak (1.6x faster). Compiled sparse vs
  compiled dense with TF32: 1.07x faster, 92 % less torch memory.
- The paper's attribution ("over 6 times faster and 43 % less GPU memory" for sparse layers + dependency update, then
  1.5x for compilation) is not reproduced: a-upstream -> sparse-eager-tf32 is 3.95x faster with 1.5 % *more* torch
  memory, and compilation adds 1.64x and the whole memory reduction. The overall factor, 6.46x faster and 95 % less
  torch memory (80 % less nvidia-smi memory), is in the range of the paper's combined claim.
- Sanity checks: `sparse-compiled-tf32` 1152 ms per GAN step vs 1477 ms in `tables/arabidopsis-runtime.json` (which
  includes logging). GAN step times vary by 2.0-4.0 % (coefficient of variation over steps 5..20) in every variant; CC
  step times by 10-25 % (steps of about 40 ms, four jobs on the node), so CC comparisons are only rough.
- Why the original code needs more memory than `dense-eager-fp32` (same torch, same dense layers), from
  `diagnosis/` (allocator snapshot with allocation stacks at GAN step 6, jobs `37373688`/`37373690`):
  - Before a step, a2 holds 27236 MiB and dense-eager-fp32 21676 MiB. Optimiser state (AdamW with `amsgrad=True` in
    both, hence 3x the parameters, about 12.1 GiB), parameters (4.0 GiB), gradients (4.0 GiB) and masks plus the
    frozen controller (1.4 GiB) agree. The difference, 5473 MiB, is tensors that the model does not need: the masked
    weight copies `weight * mask` that `MaskedLinear` saves for backward (`masked_linear.py:22`, 3 x 1305 MiB for the
    hidden layers plus the input and output layers, 4026 MiB) and the activations of one generator forward pass
    (batch norm and matmul outputs, 2350 x 18496 floats each, about 1.4 GiB).
  - Cause: in the original `GAN._train_critic`, `fake = self.gen(fake_noise)` runs with gradient tracking (the critic
    only sees `fake.detach()`), and `self.tb_fake = fake` ("for tensorboard model graph") keeps `fake`, and with it the
    whole autograd graph of the generator, alive until the next critic iteration replaces it, so also between steps.
    The current code generates the critic's fake cells under `torch.no_grad()` (`training/gan.py:_critic_step`).
    `crit_loss.backward(retain_graph=True)` is not the cause; that graph only covers the small critic.
  - The peak of the original code (32.6 GiB) is in the critic stage: each iteration builds the new graph before the
    assignment to `self.tb_fake` frees the previous one, so two generator graphs (2 x 5.5 GiB) coexist with the
    gradients of the previous generator update. The current code peaks in the generator update instead (25.0 GiB), with
    one graph and the new gradients. The same holds for `a-upstream`, which has the same 26.6 GiB before a step.
- The environment lowers the peak of the generator update only (34.3 -> 30.6 GiB in a -> a2).

## Setup record

Prepared 2026-10-28 on the login node; nothing submitted yet.

- Code: GRouNdScale `HEAD` = `b3ba67bd8a6f034003dfd12dd2713336d51df185` (branch `dedup`), `src/` clean. The working tree
  has uncommitted changes outside `src/` (publication scripts, `scripts/common.sh`), which do not enter the runs. Do not
  commit to `src/` until the array has finished.
- Original code: `upstream/` -> `../upstream/GRouNdGAN` at `2df087f`, with the same local modifications as in
  `target-gene-num-upstream` (`settings.git.src_modified: true` there too): `causal_gan.py` moves the causal controller
  to the device, `gan.py` loads a training checkpoint only if it exists, `main.py` does not import `perturbation`, plus
  config/preprocessing/GRN changes that the training run does not use. None of them touches the training step.
  `a2-upstream-newenv` runs this same tree, so that a -> a2 changes nothing but the image; no compatibility fixes for
  the new environment were made.
- Images (digests from Docker Hub on 2026-10-28):
  - groundscale: `docker://milos7250/groundscale@sha256:e5732466d05309290f846c1631222a29227cfbc1a6d129c4d520367cc0f49fe7`
    (`:latest`, pushed 2026-10-22), Python 3.12.3, torch 2.14.0+cu132, `sparselinear` imports. The same torch as the
    `target-gene-num` runs.
  - original: `docker://yazdanz/groundgan@sha256:6f219dff74b1c8cc920495f62cdf8aed706c33960bfdb047034642a695dab2e7`
    (`:GRouNdGAN-toolkit`), Python 3.9.6, torch 1.13.1+cu117, `torch.backends.cuda.matmul.allow_tf32 = False` by
    default. Both images are in the apptainer cache.
  - The user site directory `~/.local/lib/python3.X/site-packages` is on `sys.path` in both images. For Python 3.12 it
    only holds pymupdf; for Python 3.9 it holds anndata 0.8.0, dill and multiprocess, which the original image therefore
    uses (as in `target-gene-num-upstream`). Left as is.
- Data: `data/{train,validation,test}.h5ad` and `data/causal_graph.pkl` link to `paper-arabidopsis/data`. Checked:
  F = 422, G = 578, k = 15 (every target has exactly 15 TFs, no TF is a target), 1000 genes, train/validation/test
  48150/10326/10326 cells, library size 1000.
- Hyperparameters: taken from `paper-arabidopsis/causal-gan/2/config.cfg` (the causal GAN trial behind
  `causal-gan-eval-trial2`) and `paper-arabidopsis/gan_hyperopt/56/config.cfg` (= `gan_hyperopt/best_trial`, the
  causal controller); they agree with the values listed in step 2. `tables/arabidopsis.cfg` of the paper is not on the
  cluster and was not compared directly. Random seed 07122309 (that of `paper-arabidopsis/causal_gan.cfg`) in both
  templates.
- TF32: `torch.set_float32_matmul_precision("high")` in `src/training/gan.py` is the only TF32 setting in `src/` and is
  called unconditionally, whether compiled or not; the original code has none.
- Logging: the current trainer runs all evaluation hooks (save, summary, UMAP, RF AUROC) at `step == max_steps` whatever
  the frequencies, but outside `_training_step` and after the measured phase window, as in the other publication
  experiments. `[CC Logging] save frequency = 20` is therefore only needed by the original code.
- Node: `n23-64-512-aragog` (4 x A100-SXM4-80GB, 128 CPUs, 396 GB) was idle with all 4 GPUs free at 2026-10-28 and is
  set in `run.sh`. Re-check with `sinfo` before submitting.
- `launch.py` checked on CPU for all four (layers, TF32) combinations: the precision after the trainer's
  `set_float32_matmul_precision("high")` call is `highest` for `TF32=0` and `high` for `TF32=1`, the generator block
  is replaced only for `dense`, and `profile-memory.py` picks up the wrapped `write_results`.

## Deviations

- Scope changed by the user on 2026-10-28: one measurement per variant instead of three repeats (no medians); all
  tasks on one node; the extra variant `a2-upstream-newenv` (original code, unchanged, in the groundscale image); and
  all combinations of dense/sparse x eager/compiled x FP32/TF32 instead of b-dense, c-eager and d-compiled. The old
  b-dense, c-eager and d-compiled are `dense-eager-tf32`, `sparse-eager-tf32` and `sparse-compiled-tf32`.
- `dense_generator.py` CPU check: the full `tests/resources/processed/` graph (G = 928, k = 14) gives H = 27840, i.e.
  2.9 GiB per dense hidden mask and weight, and was killed on the login node. The checks were run on its first 40 genes
  (G = 40, F = 50, k = 14, d = 3, w = 2, H = 1200) instead, all passing: every block has the same shape, the same
  number of non-zero weights after `reapply_mask()` and the same connectivity as the `SparseLinear` block; the three
  hidden masks share storage and the input mask is separate; a forward and backward pass run and the weight gradients
  are zero outside the mask in every block. As in GRouNdGAN, the masks are registered as non-trainable parameters; the
  current optimizers filter on `requires_grad`, so they are not optimised.
- Submission 2026-10-28: job `37373475` (`--array=1-10%4`). Task 1 (`a-upstream`) failed while reading the data:
  anndata 0.8.0 (from `~/.local`, see the setup record) cannot read `uns/log1p/base`, a `None` that the newer anndata
  of the pre-processing stored with encoding type `null` (`KeyError: ... IOSpec(encoding_type='null', ...)`). The
  `target-gene-num` data has no such element. Its non-zero exit cancelled the pending tasks 6-10, as `run.sh` intends
  for unexpected errors. Fix: `data-upstream/{train,validation,test}.h5ad` are copies of `data/` without
  `uns['log1p']['base']`, written with the current anndata (X, obs and var checked identical; the original image reads
  them, library sizes 1000), and `a-upstream` uses `DATA_DIR=data-upstream`. Tasks 1 and 6-10 were resubmitted as job
  `37373509` (`--array=1,6-10%4`) on the same node while tasks 3-5 of `37373475` were still running.
- Task 2 (`a2-upstream-newenv`, job `37373475`) ran as intended and recorded `status: error`,
  `ModuleNotFoundError: No module named 'tabulate'`: the original code does not import in the groundscale image without
  changes. Made to run at the user's request (2026-10-28), without changes to the original code or the profilers:
  - The original code imports `tabulate` (GRN creation) and `tensorflow`/`keras` (`evaluation/data_quality.py`,
    `evaluation/MMD.py`) at start-up; none is used in training. They and their missing dependencies (tensorflow-cpu
    2.21.0, keras 3.15.1, tabulate 0.10.0, astunparse, flatbuffers, gast, google_pasta, libclang, ml_dtypes, namex,
    opt_einsum, termcolor) were installed with the image's pip, `--no-deps --target a2-pydeps`, and only a2 runs with
    `PYTHONPATH=a2-pydeps`. h5py was left out (tensorflow asks for `<3.15`, the image has 3.16.0), so numpy 2.5.2,
    h5py, torch and anndata are still those of the image. Not in `~/.local`, which every run of the groundscale image
    would see.
  - Job `37373670` then crashed with a segmentation fault (no `summary.json`) when the first `AdamW` was created:
    `torch.optim.Optimizer` imports `torch._dynamo` and with it Triton, whose native library crashes when it is loaded
    after tensorflow's (reproduced on the login node: tensorflow then `torch._dynamo` exits with 139, the other order
    works; job `37373677` with `PYTHONFAULTHANDLER=1` gave the stack). `a2_launch.py` imports `torch._dynamo` first and
    then runs `../profile-memory-upstream.py` unchanged.
  - Job `37373678` (`--array=2`, `PYTHONFAULTHANDLER=1`, alone on the node) completed with `status: ok`. Unlike the
    other variants it did not share the node with three other tasks.
  - `a2-pyenv/` in this directory is a leftover of an abandoned attempt (a `--system-site-packages` venv cannot see
    the image's `/opt/venv` packages) and can be deleted.
- `dense_generator.py` no longer runs `profile-memory.py` itself; `launch.py` imports it and runs the profiler. The
  patch of the generator is unchanged. `ruff check` removed its then unused `# noqa: E402` comments.
