# compile-overhead: the memory nvidia-smi sees beyond torch in compiled runs

In every compiled run of `target-gene-num`, nvidia-smi is 6353 MiB above torch's peak reserved memory, whatever G is.
In un-compiled runs the gap is 523 MiB (the CUDA context). These runs find where the extra ~5.8 GiB comes from and
whether it limits the model size on a small GPU.

## Setup

`run.sh` runs compiled `target-gene-num` configurations (same data, config and `main()` as `../target-gene-num`) as an
array of tasks `"<compile threads>:<free GiB>:<G>[:<options>]"`:

- free GiB: `ballast.py` first fills the A100 until `cudaMemGetInfo` reports exactly that much free memory (8 GiB,
  8190-8194 MiB measured), which simulates a small GPU. `full` runs without a ballast.
- options: `trim` resets the CUDA stack limit after each warm-up step 0..4; `aten` sets
  `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=ATEN` (matrix multiplications autotuned over cuBLAS only, no Triton GEMM
  templates); `expandable` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- `profile-lmem.py` is `../profile-memory.py` plus the stack limit of the context (`cuCtxGetLimit(CU_LIMIT_STACK_SIZE)`)
  at every snapshot of `trace.json` (`stack_size_bytes`), and the trim.
- A shell loop logs the memory of every process on the GPU every second to `processes.csv`.
- Pass tasks through the environment, `TASKS="8:8:1000:aten,trim ..." sbatch --array=1-N run.sh`; `sbatch --export`
  splits at the commas. `python summarize.py` prints one line per run.

`results/<variant>/g<G>/compiled/` has the usual `summary.json`, `trace.json`, `nvidia-smi.csv` plus `processes.csv`
and `ballast.json`. With a ballast, the nvidia-smi values include the ballast (about 72 GiB).

## Findings

1. **It is not a separate process.** `processes.csv` lists only the training process on the GPU; inductor's compile
   workers never touch the GPU. `TORCHINDUCTOR_COMPILE_THREADS` is irrelevant.
2. **It is the CUDA local-memory (stack) pool of the training process.** The per-thread stack limit is 1024 bytes after
   the model initialization and 26656 bytes after controller step 0 (28464 after generator step 0). The driver grows
   the local-memory pool to the stack size times the resident threads of the GPU (108 SMs x 2048 threads on the A100)
   whenever a kernel needs it and never shrinks it on its own: 26656 x 221184 = 5623 MiB, and resetting the limit
   released 5408 MiB in the controller and 5788 MiB in the generator. The kernels that need it are the Triton GEMM
   templates that max-autotune benchmarks (the masked linear layers are `mm`). The kernels chosen do not: after a reset
   the stack stays at 1024 bytes for steps 5..20, and with `aten` it never leaves 1024. After the reset nvidia-smi is
   torch reserved + about 565 MiB, as in un-compiled runs.
3. **It does limit the model size, and it is not released when needed.** The pool is taken at generator step 0, when
   the activations and the autotuning buffers are also live, so resetting it afterwards is too late. Results with
   8 GiB free:

| G | default | `trim` | `aten` | `aten,expandable` | full-GPU torch reserved |
|---|---|---|---|---|---|
| 1000 | ok | ok | ok | | 1338 MiB |
| 2000 | oom (step 3) | ok | ok | | 1960 MiB |
| 3000 | oom (step 0, `NoValidChoicesError`) | oom (step 0) | ok | | 2690 MiB |
| 9000 | oom (step 0) | oom (step 0) | ok | | 6496 MiB |
| 10000 | oom (step 0) | oom (step 0) | oom (step 11 in one run, step 2 in another) | ok | 7188 MiB |
| 11000 | oom (step 0) | oom (step 0) | oom (step 2) | oom (step 2) | 8572 MiB |
| 12000 | | | | oom (step 2) | 8560 MiB |

   The out-of-memory messages of the default runs report "this process has 7.6-8.0 GiB in use" with 1.4-1.8 GiB
   allocated by torch. The `aten` failure at G = 10000 had 783 MiB reserved but unallocated (fragmentation), which
   `expandable_segments` removes.
4. **The cuBLAS-only autotuning costs no speed here.** Generator step time with `aten` against the default (full GPU
   or 8 GiB free, `../target-gene-num`): 98 vs 96 ms at G = 1000, 149 vs 148 ms at 2000, 231 vs 228 ms at 3000, 493 vs
   505 ms at 8000, 625 vs 633 ms at 9000.

So on an 8 GiB A100 the default compiled model fits G = 1000 (the nvidia-smi bound), and with `aten` and
`expandable_segments` it fits G = 10000 (the torch bound: reserved + context). On a different GPU the pool scales with
its resident threads (SMs x max threads per SM), so it is smaller on smaller GPUs, e.g. about 1.9 GiB for 46 SMs x 1536
threads at the same stack size, but the stack size depends on the autotuning candidates, which depend on the GPU. This
is an estimate, not measured.

`results/threads8-freefull-trim/` trimmed only after step 4 (the first version of `profile-lmem.py`); every other trim
run trimmed after each of the steps 0..4.
