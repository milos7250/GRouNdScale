#!/usr/bin/env python3
"""Creates the data of the target-gene-num experiment from data/base, the 17789-gene dataset and its GRN.

F is fixed and G is varied. The GRN is restricted to a selection of the TFs, here the F = 1000 TFs with the highest
summed importance. The targets are the genes outside these TFs that have at least k = 15 edges from them, which is
16789 = 17789 - 1000 of them, so 16789 is the largest possible G. The first G of the targets in a fixed random order are
used, so the targets of a smaller run are contained in a larger one. The order is random because ordering by importance
leaves TFs unused at small G, which would change F between runs: with G = 1000, 994 of the 1000 TFs are used under that
order and all 1000 under a random one.

`create_GRN` takes the top-k edges per target among the selected TFs, writes the causal graph and subsets the dataset to
the TFs and targets of the graph. F and G are checked on every result.

Run from publication/target-gene-num: python prepare-data.py
"""

import pickle
import shutil
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from grn_tools import build_graph

BASE = HERE / "data" / "base"
NUM_TFS = 1000
K = 15
SEED = 781759803  # the random seed of the experiments
SIZES = [*range(1000, 16001, 1000), 16789]
FILES = ("train.h5ad", "validation.h5ad", "test.h5ad")


def main() -> None:
    grn = pd.read_csv(BASE / "inferred_grnboost2.csv", index_col=0)  # sorted by importance within each target
    genes = list(ad.read_h5ad(BASE / "train.h5ad", backed="r").var_names)

    tf_importance = grn.groupby("TF")["importance"].sum().sort_values(ascending=False)
    tfs = sorted(tf_importance.index[:NUM_TFS])
    edges = grn[grn["TF"].isin(tfs)]
    edges_per_target = edges.groupby("target").size()
    candidates = [g for g in genes if g not in set(tfs) and edges_per_target.get(g, 0) >= K]
    assert len(candidates) == len(genes) - NUM_TFS and max(SIZES) == len(candidates)
    targets = list(np.random.default_rng(SEED).permutation(candidates))
    print(f"{len(tfs)} of {tf_importance.size} TFs, {len(candidates)} possible targets")

    for size in SIZES:
        out = HERE / "data" / "processed" / str(size)
        out.mkdir(parents=True, exist_ok=True)
        for name in FILES:
            shutil.copy(BASE / name, out / name)  # create_GRN subsets these in place
        chosen = set(targets[:size])
        build_graph(
            edges[edges["target"].isin(chosen)], tfs, K, *(out / name for name in FILES), out / "causal_graph.pkl"
        )

        with (out / "causal_graph.pkl").open("rb") as handle:
            graph = pickle.load(handle)
        used = set().union(*graph.values())
        n_genes = ad.read_h5ad(out / "train.h5ad", backed="r").n_vars
        assert (len(used), len(graph), n_genes) == (NUM_TFS, size, NUM_TFS + size), (len(used), len(graph), n_genes)
        assert {len(v) for v in graph.values()} == {K}
        print(f"G={size}: F={len(used)}, genes in the data={n_genes}")


if __name__ == "__main__":
    main()
