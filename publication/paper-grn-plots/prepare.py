#!/usr/bin/env python3.9
"""
Builds the inputs of the original implementation's GRN benchmark (`main.py --benchmark_grn`) from a simulated dataset
and its ground truth GRN as published on the GRouNdGAN website:

- the causal graph, pickled {target index: {TF indices}} with indices into the genes of the simulated cells, the format
  that `main.py --create_grn` writes and `--benchmark_grn` reads,
- the TF list for GRNBoost2 (the TFs of the ground truth GRN, one column "Symbol", as `[GRN Preparation] TFs`).

Written for the image of the original implementation (Python 3.9, anndata 0.7.8, pandas 1.3.0).
"""

import argparse
import pickle
from pathlib import Path

import anndata as ad
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", type=Path, required=True, help="simulated cells (.h5ad), only the gene names are read")
    parser.add_argument("--grn", type=Path, required=True, help="ground truth GRN (.csv, columns TF and target)")
    parser.add_argument("--causal-graph", type=Path, required=True, help="output: pickled causal graph")
    parser.add_argument("--tfs", type=Path, required=True, help="output: TF list")
    args = parser.parse_args()

    genes = ad.read_h5ad(args.cells, backed="r").var_names
    if not genes.is_unique:
        raise ValueError(f"{args.cells} has duplicate gene names")

    # keep_default_na=False keeps gene names such as "NA" as strings
    grn = pd.read_csv(args.grn, dtype=str, keep_default_na=False)
    if grn.shape[1] != 2:
        raise ValueError(f"{args.grn} must have two columns (TF, target), it has {grn.shape[1]}")
    grn.columns = ["TF", "target"]

    if grn.duplicated().any():
        raise ValueError(f"{args.grn} has {grn.duplicated().sum()} duplicate edges")
    missing = (set(grn["TF"]) | set(grn["target"])) - set(genes)
    if missing:
        raise ValueError(f"{len(missing)} genes of the GRN are not in {args.cells}, e.g. {sorted(missing)[:5]}")
    tfs = set(grn["TF"])
    targets = set(grn["target"])
    if tfs & targets:
        raise ValueError(f"{len(tfs & targets)} TFs are also targets; the causal graph has no edges between TFs")

    regulators = grn.groupby("target")["TF"].apply(set)
    causal_graph = {genes.get_loc(target): {genes.get_loc(tf) for tf in tfs_} for target, tfs_ in regulators.items()}

    # the causal graph maps back to exactly the edges of the GRN
    edges = {(genes[tf], genes[target]) for target, tfs_ in causal_graph.items() for tf in tfs_}
    assert edges == set(zip(grn["TF"], grn["target"]))

    args.causal_graph.parent.mkdir(parents=True, exist_ok=True)
    with open(args.causal_graph, "wb") as fp:
        pickle.dump(causal_graph, fp, protocol=pickle.HIGHEST_PROTOCOL)
    args.tfs.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Symbol": sorted(tfs)}).to_csv(args.tfs, sep="\t", index=False)

    k = regulators.apply(len)
    print(f"Genes in the cells: {len(genes)}")
    print(f"TFs: {len(tfs)}, targets: {len(targets)}, genes in neither: {len(genes) - len(tfs) - len(targets)}")
    print(f"Edges: {len(grn)}, density: {len(grn) / (len(tfs) * len(targets)):.4f}, TFs per target: {k.min()} to {k.max()}")
    print(f"Saved the causal graph to {args.causal_graph} and the TF list to {args.tfs}")


if __name__ == "__main__":
    main()
