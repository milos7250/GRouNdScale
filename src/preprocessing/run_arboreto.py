#!/usr/bin/env python
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import pandas as pd
    import scanpy as sc


def infer_grn(adata: "sc.AnnData", tf_names: Iterable[str] | Literal["all"] = "all", seed: int = 777) -> "pd.DataFrame":
    """
    Infer a gene regulatory network (GRN) using GRNBoost2.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix containing single-cell data.
    tf_names : Iterable[str] | Literal["all"], optional
        List of transcription factor (TF) names to consider for GRN inference. If "all", all genes will be considered as TFs. Default is "all".
    seed : int, optional
        Random seed for reproducibility. Default is 777.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the inferred GRN.
    """
    import numpy as np
    import pandas as pd
    from arboreto.algo import grnboost2
    from scipy import sparse

    x = adata.X.toarray() if sparse.issparse(adata.X) else adata.X  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
    X = np.array(x)
    adata_df = pd.DataFrame(X, columns=adata.var_names)
    grn_df = (
        grnboost2(adata_df, tf_names=tf_names, seed=seed, verbose=True)  # pyright: ignore[reportArgumentType]
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )

    return grn_df


def main():
    try:
        import rich_click as click
    except ImportError:
        import click

    @click.command()
    @click.argument(
        "h5ad_file",
        type=click.Path(exists=True, dir_okay=False),
        help="Path to the input h5ad file containing single-cell data.",
    )
    @click.argument(
        "tf_names",
        default="all",
        help="Path to a tab- or comma-separated file containing TF names (in column with the name 'Symbol') "
        "or 'all' to use all genes as TFs.",
    )
    @click.option("--out", default="grnboost2_grn.csv", help="Path to save the inferred GRN CSV file.")
    @click.option("--seed", default=777, help="Random seed for reproducibility.")
    def run_infer_grn(h5ad_file, tf_names, out, seed):
        """Runs GRNBoost2 inference on the provided h5ad file and saves the inferred GRN to a CSV file."""
        import pandas as pd
        import scanpy as sc

        adata = sc.read_h5ad(h5ad_file)
        tf_names_list = None
        if tf_names != "all":
            # Sniff the delimiter so the same TF list works here and in [GRN Preparation]/TFs.
            tf_names_df = pd.read_csv(tf_names, sep=None, engine="python")
            tf_names_list = tf_names_df["Symbol"].tolist()
        else:
            tf_names_list = "all"

        grn_df = infer_grn(adata, tf_names=tf_names_list, seed=seed)
        grn_df.to_csv(out)
        print(f"GRN inference completed. Results saved to {out}")

    run_infer_grn()


if __name__ == "__main__":
    main()
