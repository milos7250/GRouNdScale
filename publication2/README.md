# publication2: memory experiments on one code version

Memory experiments on the current code, one result format, and both the compiled and the un-compiled sparse model, plus
the original (dense) implementation for the target-gene experiment. In each experiment one of the hyperparameters F (TFs),
G (target genes) and k (TFs per target) is varied and the others are held fixed. Training is started through `main()`;
the profiling scripts only wrap it with measurements.

All runs have finished, see [Results](#results).

## Layout

| Path | Purpose |
|---|---|
| `causal_gan.cfg` | One config for every run of the new implementation. B, d, w, n, the data, the causal graph and the compilation come from the environment. |
| `profile-memory.py` | Runs `main(train=True)` of the new implementation with measurements and writes the results of one run. |
| `memory_tools.py` | The measurement classes (torch and nvidia-smi memory, step time, `summary.json`) shared by both profilers. Written for Python 3.9 and torch 1.13 as well as the current environment. |
| `profile-memory-upstream.py` | The same measurements for the original implementation in `upstream/`, which runs in its own Python 3.9 image. |
| `common.sh` | Job body shared by the array experiments. Maps the Slurm array task to a (run, mode). |
| `grn_tools.py` | Builds a causal graph from a restricted GRN with the repository's `create_GRN`. |
| `collect-results.py` | Gathers every `summary.json` into `results.csv`. |
| `plot-results.py` | The four figures of the causal generator in `plots/`, see [Plots](#plots). |
| `tf-k/` | k = 1..20 on the 17789-gene dataset, F = 1000, G = 16789. 40 tasks, 4 at a time. |
| `target-gene-num/` | G = 1000..16000 and 16789 on the 17789-gene dataset, F = 1000, k = 15. 34 tasks, 4 at a time. |
| `target-gene-num-upstream/` | The same sizes on the original implementation: one job, sequential, stopping at the first run that runs out of memory. |
| `compile-overhead/` | Where the extra nvidia-smi memory of compiled runs comes from, and the largest G with 8 GiB free. See its README. |

Each experiment directory has `prepare-data.py` (creates the graphs and data, already run) except the upstream one, which
reads the data of `target-gene-num`, a `run.sh` (the Slurm array, or the sequential job), `data/` and
`results/<run>/<mode>/`. Modes are `compiled`, `eager` (the un-compiled sparse model) and `upstream`. Inside an array,
task `2i-1` is run `i` compiled and task `2i` is run `i` un-compiled. Runs are named `k<k>` and `g<G>`.

## Data

`data/base/` of `tf-k` and `target-gene-num` holds copies (not links) of the processed dataset and inferred GRN of the
`publication/size-compare` run with 17789 genes (the same copy in both). No preprocessing or GRN inference is re-run.
`prepare-data.py` derives the causal graphs from them, using `create_GRN` on the GRN restricted to fixed TFs:

- `tf-k`: F = 1000 TFs, the ones with the highest summed importance as in `limit-tfs.py` of `tf-num-experiment`, except
  that TFs which are the top-1 regulator of no target are swapped for the next ones by importance. Without the swap 4 of
  the 1000 TFs are unused at k = 1 and 1 at k = 2, F would change with k, and `create_GRN` would drop the genes of the
  unused TFs from the data. G = 16789 = 17789 - 1000 is every gene outside the TFs, the largest possible G. Each of
  them has at least 28 edges from the TFs, so each has exactly k regulators for every k up to 28, and k = 1..20 is
  run. F = 1000, G = 16789 and the 17789 genes are the same for every k. The earlier `tf-k-experiment` used a
  3000-gene dataset.
- `target-gene-num`: F = 1000 of the 1192 TFs, the ones with the highest summed importance, as in `limit-tfs.py` of
  `tf-num-experiment`. 1000 is the largest multiple of 100 for which every TF is still used at G = 1000 (at 1100 TFs only
  one of the random orders tried used all of them, 1099 and 1098 TFs were used under the dataset order and the importance
  order). Every gene outside these TFs has at least 15 edges from them, so G can be at most 17789 - 1000 = 16789. The
  targets are taken in a fixed random order, so a smaller run's targets are contained in a larger run's. The data is
  subset to the TFs and targets, as `create_GRN` does, and the 17 subsets in `data/processed/<G>/` take about 2.9 GB.
  Ordering by importance was rejected: it leaves 6 of 1000 TFs unused at G = 1000 and F would change between runs.

`prepare-data.py` asserts F, G and k of every graph. The `.h5ad` and `.csv` copies are git-ignored, the graphs tracked.

## Running

```bash
cd publication2/tf-k && sbatch run.sh                     # likewise target-gene-num and target-gene-num-upstream
python publication2/collect-results.py                     # results.csv, after the jobs finished
python publication2/plot-results.py                        # figures in publication2/plots; --log-y, --steady, --no-theory
```

`sbatch --array=1,2 run.sh` runs only the first run of an array. `KEEP_OUTPUTS=1` keeps checkpoints, UMAPs and
TensorBoard logs, which are otherwise deleted after the measurements are written. The scripts have no `--time` limit.

The container only mounts the working directory, so `common.sh` (and `target-gene-num-upstream/run.sh`) add the
repository to `APPTAINER_BIND`, and the upstream script also the real location of the `upstream/` link. Without it the
jobs cannot find `profile-memory.py` and `src/`.

## Result format

Every run writes three files to `results/<run>/<mode>/`.

`summary.json`: `experiment`, `run`, `mode`, `status` (`ok`, `oom`, `error`), `hyperparameters` (`B`, `F`, `G`, `k`, `d`,
`w`, `n`), `graph` (total genes, min/mean/max regulators per target) and, for `controller` and `generator`, the
measurements:

| Field | Meaning |
|---|---|
| `average_step_time_ms` | Mean wall time of a training step (critic iterations, generator, labelers), CUDA synchronized, steps >= `warmup_steps` (5) |
| `peak_torch_allocated_mib` | `torch.cuda.max_memory_allocated` from the start of the model initialization to the end of the last step |
| `peak_torch_reserved_mib` | The same for `max_memory_reserved` |
| `peak_nvidia_smi_mib` | Maximum `memory.used` of nvidia-smi in the same window, sampled every 0.2 s and at every step boundary |
| `steady_state` | The torch peaks restricted to steps >= `warmup_steps`, and the peak of each step stage (`critic`, `generator`, `labelers`) |

`B`, `d`, `w`, `n` are configured; `F` is the number of distinct TFs and `G` the number of targets in the causal graph;
`k` is the largest number of regulators of a target (`graph.k_min` shows when the graph is not uniform). `settings` holds
the GPU, the versions and `git` (the commit of the code and whether `src/` had uncommitted changes).

`trace.json`: for each phase the model initialization, optimizer initialization and, for un-compiled runs, the
TensorBoard graph tracing (allocated, reserved and nvidia-smi memory before and after, and its peak, parameter counts
of every module), and every training step (time, memory before and after, peak, nvidia-smi peak, stage peaks).
Peaks are in bytes there. `nvidia-smi.csv`: the raw samples (epoch seconds, MiB).

## Results

77 runs: 62 `ok`, 15 `oom`. Every `oom` is a run whose generator phase ran out of the 80 GB of the A100.

| Experiment | Compiled | Un-compiled | Original (dense) |
|---|---|---|---|
| `tf-k` (k = 1..20) | all 20 ok, generator torch peak 3.5 to 14.3 GiB | k = 1..12 ok (73 GiB at k = 12), k = 13..20 oom | not run |
| `target-gene-num` (17 sizes) | all 17 ok, 1.1 to 11.3 GiB | G = 1000..11000 ok, G = 12000 and up oom | G = 1000 and 2000 ok (19 and 72 GiB), G = 3000 oom, larger not run |

- `tf-k` at k = 15 and `target-gene-num` at G = 16789 are the same configuration. Their compiled results agree (11337.033
  MiB torch peak in both).
- The partial values of an `oom` run (the state at the crash, after 0 or 1 generator steps) are in the files but are not
  measurements.

## Plots

`plot-results.py` writes `memory-vs-k`, `memory-vs-G`, `time-vs-k` and `time-vs-G` (PNG and PDF) to `plots/`. They show
the causal generator only, with all curves of an experiment on one axis.

- Memory: torch allocated (solid), torch reserved (dashed) and nvidia-smi (dotted) for each mode, a line at the GPU
  memory (80 GiB, from the device name), and the analytical estimate of the peak as a thick pale line per mode. The
  estimates are the compiled and eager sparse equations of `publication/theoretical-size/compare-measurements.py` and,
  for the original implementation, the dense side of `compute_all` in `publication/theoretical-size/gpu_memory_model.py`.
  They are evaluated at every x of the experiment, so they continue past the GPU memory where the runs failed. They are
  for B = 1024, d = 2, w = 1, n = 1 and are skipped otherwise.
- Time: measured step times only. There is no analytical model of time.
- A run that ran out of memory has a red cross at the top of the panel instead of a point.
- The original implementation appears only in the `target-gene-num` figures (as a third series, green).

How the estimates fit: the eager sparse estimate follows the measurements closely (it is at most about 25% below them at
k = 1 and within about 3% from k = 5 up); the compiled estimate undershoots at small k and overshoots at k = 20 (its
constant was calibrated at k = 15); the dense estimate is about 15% below the measurements (16.1 against 19.1 GiB at
G = 1000, 62.5 against 71.9 GiB at G = 2000).

## Things to know

- The controller (`CC` sections) is trained first at batch size 128 for 20 steps, then the generator for 20 steps. The
  steps are numbered 0..20 and step 0 trains no critic. The controller depends on the total number of genes, not on
  B, d, w, n.
- The phase peaks include warm-up steps (compilation and autotuning may peak higher). The `steady_state` fields do not,
  and the per-step trace allows any other cut. nvidia-smi cannot be cut that way, because the caching allocator holds
  its reserved memory.
- Compiled nvidia-smi values are about 5.8 GiB higher than un-compiled ones at the same torch memory: the CUDA
  local-memory (stack) pool of the training process, grown by the Triton GEMM templates that max-autotune benchmarks and
  never shrunk. It is real memory that limits the model size on a small GPU; `compile-overhead/` measures it and shows
  how to avoid it (cuBLAS-only GEMM autotuning). The torch numbers are not affected.
- The window ends with the last training step. The evaluation hooks that `main()` runs afterwards (checkpoint, UMAP,
  Random Forest AUROC) are not measured, and the controller checkpoint they write is what the generator phase loads.
- Un-compiled runs trace the model graph for TensorBoard before training (`log_tensorboard_graph`), which allocates
  memory that training itself does not need. Its peak is its own event in `trace.json` and is in the phase peak.
- The un-compiled model needs far more memory (about 6.4 GiB per 1000 target genes at k = 15 in the measurements). A run
  that does not fit ends with `status: "oom"`, the exit code is 0 so the array continues, and the partial measurements
  are kept. A failure other than running out of memory exits non-zero, which cancels the pending array tasks.
- The compilation cache is per job (`scripts/common.sh` puts it in the job's temporary directory), so every compiled
  run compiles from scratch.
- `memory_tools.py` was factored out of `profile-memory.py` after the new-implementation runs, to share it with the
  upstream profiler. The code is moved, not changed, and the smoke test gives the same output structure, but the commit
  in those summaries predates it.

## Original implementation

`target-gene-num-upstream/` runs the `target-gene-num` data and graphs (read from `../target-gene-num/data`) through
`upstream/src/main.py --train`, with the config of `causal_gan_upstream.cfg` filled in from the environment. It needs the
image `docker://yazdanz/groundgan:GRouNdGAN-toolkit` and Python 3.9 (torch 1.13), so `memory_tools.py` and
`profile-memory-upstream.py` are written for it. Results are in `results/g<G>/upstream/` in the format above with
`mode: "upstream"`, and the summaries record Python 3.9.6 and the upstream commit.

- The original implementation has no step method: its trainers run the loop in `train()`. A step is everything from the
  first stage of an iteration (`_train_critic`, `_train_generator`, `_train_labelers`) to the end of its last stage, which
  is the same work as a step of the new implementation. Evaluation, logging and saving are not part of it.
- The sizes run one after the other in one job and stop after the first one that runs out of memory (`status: "oom"`),
  because the larger ones would too. The remaining sizes have no results.
- It has no `compile modules` and no sparse layer: it is the dense masked implementation.
- `main.py` of the original implementation does not import in the image of the new implementation (missing `tabulate` and
  `tensorflow`), which is another reason for its own image.
- Its `src/` had local modifications (`evaluation/grn_inference.py`, not on the training path), which `src_modified`
  records.
