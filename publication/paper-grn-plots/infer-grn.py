#!/usr/bin/env python3.9
"""
Infers a GRN from simulated cells with GRNBoost2, with the regulators restricted to a list of TFs, and writes it in the
format that the original implementation's GRN benchmark (`main.py --benchmark_grn`) reads: tab separated, columns TF,
target and importance, ordered by decreasing importance.

Written for the image of the original implementation (Python 3.9, arboreto 0.1.6, dask 2021.12), whose
`main.py --create_grn` runs GRNBoost2 the same way (`grnboost2(..., tf_names=TFs, seed=1)`). The only difference is the
dask cluster: one single-threaded worker per allocated CPU instead of arboreto's default local cluster, which sizes
itself to the whole node.
"""

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", type=Path, required=True, help="simulated cells (.h5ad)")
    parser.add_argument("--tfs", type=Path, required=True, help="TF list (tab separated, column 'Symbol')")
    parser.add_argument("--out", type=Path, required=True, help="output: inferred GRN (.tsv)")
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())),
        help="dask workers (default: SLURM_CPUS_PER_TASK, else the CPU count)",
    )
    parser.add_argument("--seed", type=int, default=1, help="GRNBoost2 seed (default: 1, as in --create_grn)")
    parser.add_argument("--n-cells", type=int, help="infer from a random subset of this many cells (default: all)")
    parser.add_argument("--subset-seed", type=int, default=0, help="seed of the random subset (default: 0)")
    args = parser.parse_args()

    import anndata as ad
    import numpy as np
    import pandas as pd
    from arboreto.algo import grnboost2
    from distributed import Client, LocalCluster
    from scipy import sparse

    cells = ad.read_h5ad(args.cells)
    if args.n_cells is not None and args.n_cells < cells.n_obs:
        rng = np.random.default_rng(args.subset_seed)
        cells = cells[np.sort(rng.choice(cells.n_obs, args.n_cells, replace=False))]
    x = cells.X.toarray() if sparse.issparse(cells.X) else np.asarray(cells.X)
    expression = pd.DataFrame(x, columns=cells.var_names)

    tfs = pd.read_csv(args.tfs, sep="\t", dtype=str, keep_default_na=False)["Symbol"].tolist()
    missing = set(tfs) - set(expression.columns)
    if missing:
        raise ValueError(f"{len(missing)} TFs are not in {args.cells}, e.g. {sorted(missing)[:5]}")

    print(
        f"GRNBoost2 on {expression.shape[0]} cells x {expression.shape[1]} genes with {len(tfs)} TFs, "
        f"{args.workers} workers, seed {args.seed}",
        flush=True,
    )
    with LocalCluster(
        n_workers=args.workers,
        threads_per_worker=1,
        processes=True,
        dashboard_address=None,
        local_directory=os.environ.get("TMPDIR"),
    ) as cluster, Client(cluster) as client:
        grn = grnboost2(expression, tf_names=tfs, client_or_address=client, seed=args.seed, verbose=True)

    grn = grn.sort_values("importance", ascending=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    grn.to_csv(args.out, sep="\t", index=False)
    print(f"Saved {len(grn)} edges to {args.out}")


if __name__ == "__main__":
    main()
