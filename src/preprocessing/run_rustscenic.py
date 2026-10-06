#!/usr/bin/env python
import os
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import pandas as pd
    import scanpy as sc


class EnvironmentManager:
    def __init__(self, new_env_vars: dict[str, str]):
        self.new_env_vars = new_env_vars
        self.old_env_vars = {}

    def __enter__(self):
        # Store old environment variables and set new ones
        self.old_env_vars = os.environ.copy()
        os.environ.update(self.new_env_vars)

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore old environment variables
        os.environ.clear()
        os.environ.update(self.old_env_vars)


def infer_grn(adata: "sc.AnnData", tf_names: Iterable[str] | Literal["all"] = "all", seed: int = 777) -> "pd.DataFrame":
    """
    Infer a gene regulatory network (GRN) using RustScenic.

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
    import scanpy as sc
    from rustscenic.grn import infer

    from loggers import setup_logger

    # Log-transform the data
    anndata_grn = adata.copy()
    sc.pp.normalize_total(anndata_grn, target_sum=1e4)
    sc.pp.log1p(anndata_grn)

    # Infer the GRN using RustScenic
    tfs = anndata_grn.var_names if tf_names == "all" else tf_names
    n_threads = os.environ.get("OMP_NUM_THREADS", "8")
    setup_logger(__name__).info(f"Starting rustscenic GRN inference using {n_threads} threads.")
    with EnvironmentManager({
        "RAYON_NUM_THREADS": n_threads,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONNOUSERSITE": "1",
    }):
        grn_df = (
            infer(anndata_grn, tf_names=tfs, seed=seed)
            .sort_values(by="importance", ascending=False)
            .reset_index(drop=True)
        )
    del anndata_grn

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
    @click.option("--out", default="rustscenic_grn.csv", help="Path to save the inferred GRN CSV file.")
    @click.option("--seed", default=777, help="Random seed for reproducibility.")
    def run_infer_grn(h5ad_file, tf_names, out, seed):
        """Runs RustScenic inference on the provided h5ad file and saves the inferred GRN to a CSV file."""
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
