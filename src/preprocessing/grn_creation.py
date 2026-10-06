import pickle
from collections.abc import Iterable
from configparser import ConfigParser
from itertools import chain
from pathlib import Path
from typing import TypeVar

import pandas as pd
import scanpy as sc

from loggers import setup_logger
from randomness import random_seed

from .grn_accessor import GRNAccessor

_T = TypeVar("_T")


# Code tries to avoid using sets to preserve order where possible for reproducibility
def unique_list(seq: Iterable[_T], /) -> list[_T]:
    """Returns a list of unique elements, preserving the original order."""
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


def create_GRN(cfg: ConfigParser) -> None:
    """
    Infers a GRN using the backend selected by ``[GRN Preparation]/method``
    (``grnboost2`` or ``rustscenic``) and uses it to construct a causal graph to impose onto GRouNdScale.

    Parameters
    ----------
    cfg
        Parser for config file containing GRN creation params.
    """
    # Configure logger
    logger = setup_logger(__name__)

    real_cells = sc.read_h5ad(cfg.get("Data", "train"))
    real_cells_val = sc.read_h5ad(cfg.get("Data", "validation"))
    real_cells_test = sc.read_h5ad(cfg.get("Data", "test"))
    if (
        real_cells.uns.get("GRouNdScale_was_subsetted") is not None
        or real_cells_val.uns.get("GRouNdScale_was_subsetted") is not None
        or real_cells_test.uns.get("GRouNdScale_was_subsetted") is not None
    ):
        raise ValueError(
            "The provided training dataset appears to have been subsetted by the create_GRN method "
            "previously. To avoid inconsistencies, please re-run the preprocessing step to obtain "
            "the full dataset."
        )

    # find TFs that are in highly variable genes
    gene_names = real_cells.var_names
    if cfg.get("GRN Preparation", "TFs", fallback=None) is None:
        TFs = "all"
    else:
        TFs = pd.read_csv(cfg.get("GRN Preparation", "TFs"), sep="\t")["Symbol"]
        TFs = [tf for tf in TFs if tf in gene_names]

    # infer GRN if it doesn't already exist
    if not Path(cfg.get("GRN Preparation", "Inferred GRN")).exists():
        grn_method = cfg.get("GRN Preparation", "method", fallback="grnboost2").lower()
        if grn_method == "grnboost2":
            from .run_arboreto import infer_grn
        elif grn_method == "rustscenic":
            from .run_rustscenic import infer_grn
        else:
            raise ValueError(
                f"GRN inference method {grn_method} not supported. Please choose 'grnboost2' or 'rustscenic'."
            )

        logger.info(f"Starting GRN inference using {grn_method} with {len(TFs) if TFs != 'all' else 'all'} TFs.")
        inferred_grn = infer_grn(real_cells, tf_names=TFs, seed=random_seed)
        inferred_grn.to_csv(cfg.get("GRN Preparation", "Inferred GRN"))
        logger.info(f"Successfully saved GRN inferred by {grn_method} to {cfg.get('GRN Preparation', 'Inferred GRN')}")
    else:
        logger.info(f"Using already existing GRN at {cfg.get('GRN Preparation', 'Inferred GRN')}")

    # read GRN csv output, group TFs regulating genes, sort by importance
    real_grn = GRNAccessor.from_csv(Path(cfg.get("GRN Preparation", "Inferred GRN")))

    # When using GRNBoost2 without a predefined TF list, all genes are considered as both potential TFs and targets.
    # This leads to most genes occuring as both TFs and targets in the inferred GRN. As we remove self-regulatory edges
    # in favour of causal edges from TFs to targets, we would end up with most genes being used as TFs (~95% of genes)
    # and very few targets. To mitigate this, we select TFs as those genes with higher total outgoing importance than
    # incoming importance, and then remove any targets that are also TFs.
    if TFs == "all":
        real_grn = real_grn.grn.to_bipartite()

    causal_graph = dict(real_grn.groupby("target")["TF"].apply(list))

    k = int(cfg.get("GRN Preparation", "k"))

    if cfg.get("GRN Preparation", "strategy") == "top":
        logger.info(f"Creating top {k} GRN from top TFs")
        causal_graph = {
            gene: unique_list(tfs)[:k]  # to sample the top k edges
            for (gene, tfs) in causal_graph.items()
        }
    elif cfg.get("GRN Preparation", "strategy") == "pos ctr":
        logger.info("Creating positive control GRN from even indexed top TFs (top 1, 3, 5, ...)")
        causal_graph = {
            gene: unique_list(tfs)[0:k:2]  # sample even indices
            for (gene, tfs) in causal_graph.items()
        }

    elif cfg.get("GRN Preparation", "strategy") == "neg ctr":
        logger.info("Creating negative control GRN from odd indexed top TFs (top 2, 4, 6, ...)")
        causal_graph = {
            gene: unique_list(tfs)[1:k:2]  # sample odd indices
            for (gene, tfs) in causal_graph.items()
            if len(tfs) > 1
        }
    else:
        raise ValueError("GRN preparation strategy not valid")

    # get gene, TF names
    tfs = unique_list(chain.from_iterable(causal_graph.values()))

    # delete targets that are also regulators
    causal_graph = {k: v for (k, v) in causal_graph.items() if k not in tfs}

    # handle genes with no regulators
    missing_genes = [gene for gene in gene_names if gene not in causal_graph.keys() and gene not in tfs]
    include_no_reg = cfg.getboolean("GRN Preparation", "include genes with no regulators", fallback=False)
    if missing_genes and include_no_reg:
        logger.info(f"Included {len(missing_genes)} targets with no regulators in the causal graph")
        causal_graph |= {gene: [] for gene in missing_genes}  # include genes with no regulators
    elif missing_genes:
        logger.warning(
            f"Excluding {len(missing_genes)} targets with no regulators from the causal graph. "
            "To include them, set 'include genes with no regulators' to True in the config."
        )
    else:
        logger.info("All targets have regulators; no targets with no regulators included in the causal graph.")

    # targets sorted by original order, TFs per target sorted by importance
    causal_graph = dict(sorted(causal_graph.items(), key=lambda item: gene_names.to_list().index(item[0])))

    targets = unique_list(causal_graph.keys())
    genes = tfs + targets
    genes = sorted(genes, key=lambda x: gene_names.to_list().index(x))  # sort genes by original order

    if not genes == gene_names.to_list():
        # overwrite train, validation, and test datasets when some genes were excluded from the dataset
        real_cells.uns["GRouNdScale_was_subsetted"] = True
        real_cells_val.uns["GRouNdScale_was_subsetted"] = True
        real_cells_test.uns["GRouNdScale_was_subsetted"] = True
        real_cells[:, genes].write_h5ad(cfg.get("Data", "train"))
        real_cells_val[:, genes].write_h5ad(cfg.get("Data", "validation"))
        real_cells_test[:, genes].write_h5ad(cfg.get("Data", "test"))
        logger.warning(
            "The provided training dataset has been subsetted to only include genes present in the causal graph. Please "
            "adjust the config to use appropriate number of genes."
        )

    # print causal graph info
    possible_edges = len(tfs) * len(targets)
    imposed_edges = len(list(chain.from_iterable(causal_graph.values())))
    logger.info(
        "Causal graph info:\n"
        + pd.DataFrame([
            ("``TFs``", len(tfs)),
            ("``Targets``", len(targets)),
            ("Genes", len(genes)),
            ("Possible Edges", possible_edges),
            ("Imposed Edges", imposed_edges),
            ("GRN density Edges", f"{imposed_edges / possible_edges * 100 if possible_edges > 0 else 0:.1f}%"),
        ]).to_string(index=False, header=False)
    )

    # convert gene names to numerical indices in the final gene list (which matches the possibly subsetted dataset)
    gene_indices = {gene: index for index, gene in enumerate(genes)}
    causal_graph = {gene_indices[gene]: {gene_indices[tf] for tf in tfs} for (gene, tfs) in causal_graph.items()}

    # save causal graph
    with open(cfg.get("Data", "causal graph"), "wb") as fp:
        pickle.dump(causal_graph, fp, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f"Successfully saved GRouNdScale causal graph to {cfg.get('Data', 'causal graph')}")
