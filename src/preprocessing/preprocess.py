import warnings
from collections import Counter
from configparser import ConfigParser
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix

from loggers import setup_logger
from randomness import random_seed


def preprocess(cfg: ConfigParser) -> None:
    """
    Apply preprocessing steps.

    Parameters
    ----------
    cfg
        Parser for config file containing preprocessing params.
    """

    # Configure logger
    logger = setup_logger(__name__)

    logger.info("Loading data...")
    if cfg.get("Preprocessing", "10x") == "True":
        anndata = sc.read_10x_mtx(cfg.get("Preprocessing", "raw"), make_unique=True, gex_only=True)

    else:
        anndata = sc.read_h5ad(cfg.get("Preprocessing", "raw"))

    logger.info("Shuffling data...")
    original_order = np.arange(anndata.n_obs)  # Store the original cell order
    np.random.default_rng(random_seed).shuffle(original_order)  # Shuffle the indices
    shuffled_order = original_order.copy()
    del original_order

    # Apply the shuffled order to the AnnData object
    anndata = anndata[shuffled_order].copy()

    # clustering
    logger.info("Clustering data...")
    ann_clustered = anndata.copy()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*The function filter_genes_dispersion is deprecated and will be removed in the future.*",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore", message=r".*zero-centering a sparse array/matrix densifies it.*", category=UserWarning
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*The function louvain is deprecated and will be removed in the future. Use :func:`scanpy.tl.leiden` instead.*",
            category=FutureWarning,
        )
        sc.pp.recipe_zheng17(ann_clustered)
        sc.tl.pca(ann_clustered, n_comps=50)
        sc.pp.neighbors(ann_clustered, n_pcs=50, random_state=random_seed)
        sc.tl.louvain(
            ann_clustered, resolution=float(cfg.get("Preprocessing", "louvain res")), random_state=random_seed
        )
    anndata.obs["cluster"] = ann_clustered.obs["louvain"]
    del ann_clustered

    # get cluster ratios
    cells_per_cluster = Counter(anndata.obs["cluster"])
    cluster_ratios = dict()
    for key, value in cells_per_cluster.items():
        cluster_ratios[key] = value / anndata.shape[0]
    anndata.uns["cluster_ratios"] = cluster_ratios
    anndata.uns["clusters_no"] = len(cluster_ratios)

    # filtering
    logger.info("Filtering data...")
    cells_no, genes_no = anndata.shape
    while True:
        sc.pp.filter_cells(anndata, min_genes=int(cfg.get("Preprocessing", "min genes")))
        sc.pp.filter_genes(anndata, min_cells=int(cfg.get("Preprocessing", "min cells")))
        if anndata.shape == (cells_no, genes_no):
            break
        cells_no, genes_no = anndata.shape

    # library-size normalization
    logger.info("Subsetting highly variable genes...")
    anndata.layers["normalized"] = sc.pp.normalize_total(
        anndata, target_sum=int(cfg.get("Preprocessing", "library size")), inplace=False
    )["X"]  # pyright: ignore[reportOptionalSubscript]

    if cfg.get("Preprocessing", "annotations", fallback=None) is not None:
        annotations = pd.read_csv(cfg.get("Preprocessing", "annotations"), delimiter="\t", index_col=["barcodes"])
        anndata.obs["celltype"] = annotations.loc[anndata.obs_names, "celltype"].values

    # identify highly variable genes
    sc.pp.log1p(anndata, layer="normalized")  # logarithmize the data
    hvgs = sc.pp.highly_variable_genes(
        anndata, layer="normalized", n_top_genes=int(cfg.get("Preprocessing", "highly variable number")), inplace=False
    )["highly_variable"]  # pyright: ignore[reportOptionalSubscript]

    del anndata.layers["normalized"]
    del anndata.uns["log1p"]
    anndata = anndata[:, hvgs].copy()  # only keep highly variable genes

    cells_no, genes_no = anndata.shape
    while True:
        sc.pp.filter_cells(anndata, min_genes=int(cfg.get("Preprocessing", "min genes")))
        sc.pp.filter_genes(anndata, min_cells=int(cfg.get("Preprocessing", "min cells")))
        if anndata.shape == (cells_no, genes_no):
            break
        cells_no, genes_no = anndata.shape
    anndata.uns["cells_no"] = anndata.shape[0]
    anndata.uns["genes_no"] = anndata.shape[1]
    sc.pp.normalize_total(anndata, target_sum=int(cfg.get("Preprocessing", "library size")))

    # sort genes by name (not needed)
    sorted_genes = np.sort(anndata.var_names)
    anndata = anndata[:, sorted_genes].copy()

    val_size = int(cfg.get("Preprocessing", "validation set size"))
    test_size = int(cfg.get("Preprocessing", "test set size"))

    anndata.X = csr_matrix(anndata.X)

    logger.info("Saving datasets...")
    Path(cfg.get("Data", "train")).parent.mkdir(parents=True, exist_ok=True)
    anndata[:val_size].copy().write_h5ad(cfg.get("Data", "validation"))
    Path(cfg.get("Data", "validation")).parent.mkdir(parents=True, exist_ok=True)
    anndata[val_size : test_size + val_size].copy().write_h5ad(cfg.get("Data", "test"))
    Path(cfg.get("Data", "test")).parent.mkdir(parents=True, exist_ok=True)
    anndata[test_size + val_size :].copy().write_h5ad(cfg.get("Data", "train"))

    logger.info("Successfully preprocessed and saved dataset.")
    logger.info(
        f"Train set ({anndata[test_size + val_size :].shape[0]} cells, {anndata.shape[1]} genes): {cfg.get('Data', 'train')}"
    )
    logger.info(f"Validation set ({val_size} cells, {anndata.shape[1]} genes): {cfg.get('Data', 'validation')}")
    logger.info(f"Test set ({test_size} cells, {anndata.shape[1]} genes): {cfg.get('Data', 'test')}")
