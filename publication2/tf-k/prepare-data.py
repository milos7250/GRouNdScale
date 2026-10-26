#!/usr/bin/env python3
"""Creates the causal graphs of the tf-k experiment, k = 1..20, from data/base, the 17789-gene dataset and its GRN.

F and G must not change with k. F = 1000 TFs are used, G = 16789 targets, which is every gene outside the TFs and so
the largest possible G. Every one of these targets has at least 28 edges from the TFs, so each has exactly k regulators
for every k up to 28 and all of k = 1..20 are possible. Every k takes the top-k edges per target among the TFs.

The TFs are the 1000 with the highest summed importance, as in limit-tfs.py of publication/tf-num-experiment, except
that TFs which are the top-1 regulator of no target are swapped for the next TFs by importance until none is left.
Without it 4 of the 1000 TFs are unused at k = 1 and 1 at k = 2, which would change F and, as `create_GRN` drops the
genes of unused TFs from the data, the dataset. The graphs are checked for this.

Run from publication2/tf-k: python prepare-data.py
"""

import pickle
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from grn_tools import build_graph

BASE = HERE / "data" / "base"
NUM_TFS = 1000
KS = range(1, 21)


def main() -> None:
    grn = pd.read_csv(BASE / "inferred_grnboost2.csv", index_col=0)  # sorted by importance within each target
    by_importance = list(grn.groupby("TF")["importance"].sum().sort_values(ascending=False).index)
    tfs, spare = by_importance[:NUM_TFS], by_importance[NUM_TFS:]
    while True:
        edges = grn[grn["TF"].isin(tfs) & ~grn["target"].isin(tfs)]
        top1 = set(edges.groupby("target", sort=False).head(1)["TF"])
        unused = [tf for tf in tfs if tf not in top1]
        print(f"{len(unused)} of {len(tfs)} TFs are the top-1 regulator of no target")
        if not unused:
            break
        assert len(spare) >= len(unused), "Not enough TFs left to replace the unused ones"
        tfs = [tf for tf in tfs if tf in top1] + spare[: len(unused)]
        spare = spare[len(unused) :]
    tfs = sorted(tfs)
    edges = grn[grn["TF"].isin(tfs)]
    n_targets = edges.loc[~edges["target"].isin(tfs), "target"].nunique()
    n_genes = grn["target"].nunique()
    per_target = edges[~edges["target"].isin(tfs)].groupby("target").size()
    assert n_targets == n_genes - NUM_TFS and per_target.min() >= max(KS)
    print(f"{len(tfs)} TFs, {n_targets} targets with {per_target.min()}-{per_target.max()} edges from the TFs")

    for k in KS:
        path = HERE / "data" / "graphs" / f"k{k}.pkl"
        build_graph(edges, tfs, k, BASE / "train.h5ad", BASE / "validation.h5ad", BASE / "test.h5ad", path)
        with path.open("rb") as handle:
            graph = pickle.load(handle)
        used = set().union(*graph.values())
        assert (len(used), len(graph)) == (NUM_TFS, n_targets), f"k={k}: F={len(used)}, G={len(graph)}"
        assert {len(v) for v in graph.values()} == {k}, f"k={k}: not every target has {k} regulators"
        print(f"k={k}: F={len(used)}, G={len(graph)}")


if __name__ == "__main__":
    main()
