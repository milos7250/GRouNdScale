"""Builds the causal graph of a run from a restricted GRN with the repository's own `create_GRN`.

The experiments fix the TFs (and, in target-gene-num, the targets) by restricting the inferred GRN to them before the
top-k edges per target are taken. `create_GRN` then returns the causal graph in the index space of the dataset, and
subsets the h5ad files when the graph does not cover every gene.
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from custom_parser import get_configparser
from preprocessing import create_GRN


def build_graph(edges: pd.DataFrame, tfs: list[str], k: int, train: Path, validation: Path, test: Path, graph: Path):
    """
    Runs `create_GRN` on the edges (columns TF, target, importance) with the given TF list and k.

    The h5ad files are only rewritten when the graph does not cover every gene, so pass copies if that can happen.
    """
    graph.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        grn_csv, tf_list = Path(tmp) / "grn.csv", Path(tmp) / "tfs.tsv"
        edges.to_csv(grn_csv)  # same layout as the GRN inferred by create_GRN
        pd.DataFrame({"Symbol": tfs}).to_csv(tf_list, sep="\t", index=False)

        cfg = get_configparser()
        cfg.read_dict({
            "GRN Preparation": {"TFs": str(tf_list), "Inferred GRN": str(grn_csv), "k": str(k), "strategy": "top"},
            "Data": {
                "train": str(train),
                "validation": str(validation),
                "test": str(test),
                "causal graph": str(graph),
            },
        })
        create_GRN(cfg)
