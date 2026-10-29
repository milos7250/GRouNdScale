# paper-grn-plots: GRN benchmark of the GRouNdGAN paper's PBMC-All simulation

The PR curve and top TF precision that were cropped from the GRouNdGAN paper for the *paper* row of the results figure
are of another dataset. This directory makes them again for PBMC-All, from the simulated cells and ground truth GRN that
the GRouNdGAN authors published, with the GRN benchmark of the original implementation (`upstream/src/main.py
--benchmark_grn`) in its own image.

## Layout

| Path | Purpose |
|---|---|
| `generated.h5ad` | Downloaded: 100000 simulated cells x 1000 genes, library-size normalised to 20000. "PBMC-ALL Expression Data (H5AD)" of the [benchmarking page](https://emad-combine-lab.github.io/GRouNdGAN/benchmarking.html). |
| `original-grn.csv` | Downloaded: the ground truth GRN imposed on that model, 14055 edges, 63 TFs, 937 targets with 15 TFs each. "PBMC-ALL Ground Truth GRN (CSV)" of the same page. |
| `run.sh` | The Slurm job (CPU only): `prepare.py`, `infer-grn.py`, then `--benchmark_grn`. |
| `prepare.py` | Writes `data/causal_graph.pkl` (pickled {target index: {TF indices}}, indices into the genes of `generated.h5ad`, the format of `--create_grn`) and `data/TFs.csv` (the 63 TFs) from `original-grn.csv`. |
| `infer-grn.py` | GRNBoost2 on the simulated cells with the regulators restricted to the TF list, written as the tab-separated TF, target, importance that the benchmark reads. |
| `results/` | Output on all cells: `simulated-grn.tsv` (inferred GRN), `benchmark.cfg`, `benchmark.log` (AUPRC and precision at k), `ground truth GRN.csv`, `PR-curve.png`, `top-TF-precision.png`. |
| `results-<N>-cells/` | The same on a random subset of N cells (`N_CELLS=<N>`). |
| `logs/` | Slurm logs. |

The data is PBMC-All: its 1000 genes and 63 TFs are those of `../paper-v4/data/processed/PBMC` (in another order),
and 11052 of the 14055 edges are in the causal graph of `../paper-v4`, which was inferred separately from the same data.

## Running

```bash
cd publication/paper-grn-plots && sbatch run.sh          # all 100000 cells, results/
N_CELLS=10000 sbatch run.sh                               # 10000 random cells, results-10000-cells/
```

An inferred GRN that already exists in the output directory is reused, so a second run only redoes the benchmark.
Delete `simulated-grn.tsv` to run GRNBoost2 again. The figures for the paper are `PR-curve.png` and
`top-TF-precision.png` of the output directory; they go to `figures/paper/` of the manuscript.

## How it works

- Every step runs in `docker://yazdanz/groundgan:GRouNdGAN-toolkit` (Python 3.9, arboreto 0.1.6, dask 2021.12, pandas
  1.3), like `target-gene-num-upstream/`. The benchmark is the `upstream/` checkout, including its local change to
  `get_fake_grn` (gene names read as strings).
- GRNBoost2 is called as in `--create_grn`: `grnboost2(cells, tf_names=TFs, seed=1)` with the 63 TFs of the ground
  truth. The only change is the dask cluster, one single-threaded worker per allocated CPU, because arboreto's default
  local cluster sizes itself to the whole node. GRNBoost2 infers regulators for all 1000 genes; the benchmark keeps the
  edges from the 63 TFs to the 937 targets and adds every missing TF-target pair with importance 0, so recall reaches 1.
- `--benchmark_grn` reads the ground truth as a causal graph, not as a CSV, and maps its indices to names with the genes
  of `[Data] train`. `generated.h5ad` is given there, so the indices of `prepare.py` refer to the right genes. `run.sh`
  checks that the ground truth GRN the benchmark writes back is `original-grn.csv`.
- The plots load the Atkinson Hyperlegible font from `Atkinson_Hyperlegible/` relative to the working directory, so
  `run.sh` links that directory of `upstream/` here; without it matplotlib falls back to DejaVu Sans.
- Baselines: precision at k of a random predictor is k / 63, and the PR baseline is the density 14055 / (63 x 937) =
  0.238.

## Things to know

- The GRouNdGAN and GRouNdScale benchmarks in the paper infer the GRN from 10000 generated cells. By default this one
  uses all 100000 published cells, which gives GRNBoost2 more data. `N_CELLS=10000` makes the comparison like for like.
- `run.sh` uses the `short` partition with 32 CPUs and 64 GiB like `scripts/create-grn.sh`. The memory need is far
  lower: arboreto sends each worker the TF columns (100000 x 63) and one target column per task.
- The scripts were tested on synthetic data in a Python 3.9 environment with the image's versions of arboreto, dask,
  distributed, pandas, numpy, scikit-learn, anndata and scanpy, with the benchmark run from the upstream sources, not in
  the image itself.
