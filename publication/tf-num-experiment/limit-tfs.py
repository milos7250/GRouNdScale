#!/usr/bin/env python3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loggers import setup_logger
from preprocessing.grn_accessor import GRNAccessor

logger = setup_logger(Path(__file__).stem)


def limit_tfs(grn: pd.DataFrame, num_tfs: int) -> pd.DataFrame:
    """
    Filters the GRN into a bipartite graph like `GRNAccessor.to_bipartite`, but keeps only the top `num_tfs` TFs with
    the highest summed importance of their outgoing edges. TFs are first classified as in `to_bipartite` (genes whose
    summed importance as TF exceeds their summed importance as target) and then ranked by their TF importance sum.
    Genes that are not among the selected TFs end up on the target side of the bipartite graph.

    Args:
        grn (pd.DataFrame): GRN DataFrame with 'TF', 'target', and 'importance' columns.
        num_tfs (int): Maximum number of top TFs to keep.

    Returns:
        pd.DataFrame: Filtered bipartite GRN.
    """
    gene_names = pd.Index(grn["TF"].tolist() + grn["target"].tolist()).unique().sort_values()
    importances = pd.DataFrame(
        {
            "TF": grn.groupby("TF")["importance"].sum(),
            "target": grn.groupby("target")["importance"].sum(),
        },
        index=gene_names,
    ).fillna(0)
    # candidates = importances[importances["TF"] > importances["target"]]
    candidates = importances[importances["TF"] > 0]
    logger.info(f"Found {len(candidates)} TF candidates.")
    top_tfs = candidates["TF"].sort_values(ascending=False).head(num_tfs).index
    if len(top_tfs) < num_tfs:
        logger.warning(f"Requested {num_tfs} TFs, but only {len(top_tfs)} are available.")
    filtered = grn[grn["TF"].isin(top_tfs) & ~grn["target"].isin(top_tfs)]
    retained_tfs = filtered["TF"].nunique()
    logger.info(
        f"Bipartite filtering retained {retained_tfs} of the {len(top_tfs)} selected TFs "
        f"({len(top_tfs) - retained_tfs} regulate only other selected TFs)."
    )
    targets = filtered["target"].unique()
    logger.info(f"Filtered GRN to bipartite graph with {retained_tfs} TFs and {len(targets)} targets.")
    return filtered.reset_index(drop=True)


if __name__ == "__main__":
    try:
        import rich_click as click
    except ImportError:
        import click

    @click.command(context_settings={"show_default": True, "help_option_names": ["-h", "--help"]})
    @click.option(
        "--input", "-i", type=click.Path(exists=True, path_type=Path), required=True, help="Input GRN CSV file."
    )
    @click.option(
        "--output", "-o", type=click.Path(path_type=Path), required=True, help="Output filtered bipartite GRN CSV file."
    )
    @click.option(
        "--num-tfs",
        "-n",
        type=click.IntRange(min=1),
        required=True,
        help="Number of top TFs with the highest importance sum to keep.",
    )
    @click.option("--tf-col", type=str, default="TF", help="Column name for TFs in the input CSV.")
    @click.option("--target-col", type=str, default="target", help="Column name for targets in the input CSV.")
    @click.option(
        "--importance-col",
        type=str,
        default="importance",
        help="Column name for importance in the input CSV. If not present, set to empty string "
        "and the script will use the order of edges as importance.",
    )
    def cli(input: Path, output: Path, num_tfs: int, tf_col: str, target_col: str, importance_col: str) -> None:
        """Filters a GRN CSV file into a bipartite graph limited to the top NUM_TFS TFs by importance sum."""
        col_names = {"TF": tf_col, "target": target_col}
        if importance_col != "":
            col_names["importance"] = importance_col
        grn = GRNAccessor.from_csv(input, col_names=col_names).grn._obj
        filtered = limit_tfs(grn, num_tfs)
        output.parent.mkdir(parents=True, exist_ok=True)
        filtered.to_csv(output, index=False)
        logger.info(f"Saved processed GRN to {output}")

    cli()
