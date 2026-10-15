import json
from configparser import ConfigParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import sparse
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.model_selection import train_test_split
from umap import UMAP

from evaluation import MMD
from evaluation.lisi import compute_lisi
from loggers import setup_logger
from randomness import random_seed

logger = setup_logger("evaluate")


def read_datasets(cfg: ConfigParser) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """
    Load real and simulated (fake) gene expression datasets.

    Parameters
    ----------
    cfg
        Parser for config file containing program params.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        A tuple containing:
        - real_cells.X : NumPy array of real gene expression data
        - fake_cells.X : NumPy array of simulated gene expression data, truncated to match real data row count
    """
    test_cells_path = cfg.get("Data", "test")

    fake_cells_path = cfg.get("Evaluation", "simulated data path", fallback="")
    if not fake_cells_path:  # Fall back to generation path
        fake_cells_path = cfg.get("Generation", "generation path", fallback="")
    if not fake_cells_path:  # Fall back on default save dir if generation path is also empty
        fake_cells_path = Path(cfg.get("EXPERIMENT", "output directory")) / "simulated.h5ad"

    real_cells = sc.read_h5ad(test_cells_path)
    fake_cells = sc.read_h5ad(fake_cells_path)

    if not isinstance(real_cells.X, sparse.csr_matrix):  # pyright: ignore[reportUnknownMemberType]
        raise TypeError("The real data matrix is not in sparse csr format. Please preprocess the data accordingly.")
    else:
        real_cells = real_cells.X
    if not isinstance(fake_cells.X, sparse.csr_matrix):  # pyright: ignore[reportUnknownMemberType]
        raise TypeError("The fake data matrix is not in sparse csr format. Please preprocess the data accordingly.")
    else:
        fake_cells = fake_cells.X

    no_of_cells = int(min(real_cells.shape[0], fake_cells.shape[0]))  # pyright: ignore[reportOptionalSubscript,reportUnknownArgumentType]
    real_cells = real_cells[:no_of_cells, :]
    fake_cells = fake_cells[:no_of_cells, :]

    return real_cells, fake_cells


def get_UMAP_embeddings(
    real: np.ndarray | sparse.csr_matrix, fake: np.ndarray | sparse.csr_matrix
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute UMAP embeddings for real and fake cell data.

    Parameters
    ----------
    real
        A NumPy array of real cell data with shape (n_real_cells, n_features).
    fake
        A NumPy array of fake/generated cell data with shape (n_fake_cells, n_features).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the 2D UMAP embeddings of the real and fake data,
        in the form (real_embedding, fake_embedding).
    """
    umap = UMAP(random_state=42, min_dist=0.0, n_jobs=1)
    umap.fit(real)  # ensure UMAP is fitted only once to preserve comparability
    real_embedding = np.array(umap.transform(real))
    fake_embedding = np.array(umap.transform(fake))

    return real_embedding, fake_embedding


def plot_UMAP(real_embedding: np.ndarray, fake_embedding: np.ndarray) -> tuple[Figure, Figure, Figure]:
    """
    Perform UMAP embedding on real and fake cell data and save a scatter plot.

    Parameters
    ----------
    real
        A NumPy array of real cell data with shape (n_real_cells, n_features).
    fake
        A NumPy array of fake/generated cell data with shape (n_fake_cells, n_features).
    output_dir

    Returns
    -------
    tuple[Figure, Figure, Figure]
        A tuple containing the matplotlib Figure objects for the scatter plot, hexbin plot, and histogram
        difference plot, respectively.
    """
    extent = np.array([
        [
            min(min(real_embedding[:, 0]), min(fake_embedding[:, 0])),
            max(max(real_embedding[:, 0]), max(fake_embedding[:, 0])),
        ],
        [
            min(min(real_embedding[:, 1]), min(fake_embedding[:, 1])),
            max(max(real_embedding[:, 1]), max(fake_embedding[:, 1])),
        ],
    ])
    margin = np.array(extent[:, 1] - extent[:, 0]) * 0.05  # 5% margin
    extent[:, 0] -= margin[0]
    extent[:, 1] += margin[1]

    plt.clf()
    scatter_fig = plt.figure(figsize=(5, 5))

    # Interweave points for cycled colors
    points = np.empty(
        (real_embedding.shape[0] + fake_embedding.shape[0], real_embedding.shape[1]), dtype=real_embedding.dtype
    )
    points[0::2] = real_embedding
    points[1::2] = fake_embedding
    colors = np.array(["blue", "red"] * (points.shape[0] // 2 + 1))[: points.shape[0]]

    plt.scatter(
        points[:, 0],
        points[:, 1],
        c=colors,
        s=3,
        edgecolor="none",
    )

    # Create artificial legend handles for the scatter plot
    a1 = plt.scatter(
        [],
        [],
        c="blue",
        label="real",
        s=3,
        edgecolor="none",
    )
    a2 = plt.scatter(
        [],
        [],
        c="red",
        label="generated",
        s=3,
        edgecolor="none",
    )
    plt.legend(handles=[a1, a2], loc="lower left", ncol=2, fontsize=8, bbox_to_anchor=(0, 0)).set(zorder=5)

    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.gca().set_axisbelow(True)
    plt.xlim(extent[0, 0], extent[0, 1])
    plt.ylim(extent[1, 0], extent[1, 1])
    plt.title("UMAP Projection of Real and Generated Cells")

    hexbin_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1: Axes
    ax2: Axes

    ax1.hexbin(
        real_embedding[:, 0],
        real_embedding[:, 1],
        mincnt=1,
        linewidths=0.0,
        extent=extent.flatten(),  # pyright: ignore[reportArgumentType]
        cmap="Reds",
    )
    ax1.set_title("Real Cells")
    plt.colorbar(ax1.collections[0], ax=ax1)

    ax2.hexbin(
        fake_embedding[:, 0],
        fake_embedding[:, 1],
        mincnt=1,
        linewidths=0.0,
        extent=extent.flatten(),  # pyright: ignore[reportArgumentType]
        cmap="Reds",
    )
    ax2.set_title("Generated Cells")
    plt.colorbar(ax2.collections[0], ax=ax2)
    plt.suptitle("UMAP Histograms")

    H_real, xedges, yedges = np.histogram2d(real_embedding[:, 0], real_embedding[:, 1], bins=80, range=extent)
    H_fake, _, _ = np.histogram2d(fake_embedding[:, 0], fake_embedding[:, 1], bins=80, range=extent)
    H_real[H_real == 0] = np.nan
    H_rel = H_real / (H_real + H_fake)  # relative density difference
    X, Y = np.meshgrid(xedges, yedges)
    v_bound = np.nanmax(np.abs(H_rel - 0.5))

    hist_rel_abun_fig = plt.figure(figsize=(5, 5))
    plt.pcolormesh(X, Y, H_rel.T, shading="auto", cmap="coolwarm", vmin=0.5 - v_bound, vmax=0.5 + v_bound)

    plt.title("UMAP Histogram Relative Abundance of Real Cells")

    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)  # shrink fig so cbar is visible
    # make new ax object for the cbar
    cbar_ax = hist_rel_abun_fig.add_axes((0.87, 0.15, 0.02, 0.7))  # x, y, width, height
    plt.colorbar(cax=cbar_ax)

    return scatter_fig, hexbin_fig, hist_rel_abun_fig


def compute_distances(
    real_cells: np.ndarray | sparse.csr_matrix, fake_cells: np.ndarray | sparse.csr_matrix, axis: int = 0
) -> tuple[float, float]:
    """
    Compute Euclidean and Cosine distances between the mean expression profiles of real and fake cells.


    Parameters
    ----------
    real_cells
        A NumPy array representing real cell data (cells × features).
    fake_cells
        A NumPy array representing fake cell data (cells × features).
    axis
        Axis along which to compute the mean expression (default is 0, meaning across cells), by default 0

    Returns
    -------
    tuple[float, float]
        A tuple containing:
        - Euclidean distance between the mean expression profiles.
        - Cosine distance between the mean expression profiles.
    """

    # calculate mean expression across cells
    fake_mean_expression = np.asarray(fake_cells.mean(axis=axis).reshape(1, -1))
    real_mean_expression = np.asarray(real_cells.mean(axis=axis).reshape(1, -1))
    return (
        euclidean_distances(fake_mean_expression, real_mean_expression).item(),
        cosine_distances(fake_mean_expression, real_mean_expression).item(),
    )


def compute_RF_AUROC(
    real_cells: np.ndarray | sparse.csr_matrix,
    fake_cells: np.ndarray | sparse.csr_matrix,
    n_components: int = 50,
) -> tuple[float, Figure]:
    """
    Compute the AUROC and plot the ROC curve of a Random Forest classifier distinguishing real from fake cells.

    Parameters
    ----------
    real_cells
        A NumPy array representing real cell data (cells x features).
    fake_cells
        A NumPy array representing fake/generated cell data (cells x features).
    n_components
        Number of principal components to retain during PCA, by default 50

    Returns
    -------
    tuple[float, Figure]
        The area under the ROC curve (AUROC) for the classifier and the matplotlib Figure object containing the ROC plot.
    """
    # create labels for real and fake data (real = 1, fake = 0)
    y_real = np.ones(real_cells.shape[0])  # pyright: ignore[reportOptionalSubscript]
    y_fake = np.zeros(fake_cells.shape[0])  # pyright: ignore[reportOptionalSubscript]

    # perform PCA
    pca = PCA(n_components=n_components, random_state=random_seed)

    # split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        sparse.vstack((real_cells, fake_cells), format="csr"),
        np.hstack((y_real, y_fake)),
        test_size=0.3,
        random_state=random_seed,
        shuffle=True,
    )

    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # train and test RF
    rf = RandomForestClassifier(n_estimators=1000, n_jobs=8, random_state=random_seed)
    rf.fit(X_train, y_train)
    preds = rf.predict_proba(X_test)
    fpr, tpr, _ = metrics.roc_curve(y_test, preds[:, 1])
    roc_auc = float(metrics.roc_auc_score(y_test, preds[:, 1]))

    # plot ROC
    plt.figure(figsize=(5, 5))
    plt.title("Receiver Operating Characteristic")
    plt.plot(fpr, tpr, "b", label=f"AUC = {roc_auc:0.2f}")
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")

    return roc_auc, plt.gcf()


def evaluate(cfg: ConfigParser) -> dict[str, float]:
    """
    Assess the data quality of the simulated dataset.

    - Umap plots of real vs simulated cells (jointly embedded and plotted).
    - Euclidean and Cosine distances between the centroids of real cells and simulated cells.
        The centroid cell was obtained by calculating the mean along the gene axis (across all simulated or real cells).
    - Area under the receiver operating characteristic curve of a random forest in distinguishing real cells from simulated ones.
        We perform dimensionality reduction using PCA to extract the top 50 PCs of each cell as the input features to the RF model.
        The RF model is composed of 1000 trees and the Gini impurity was used to measure the quality of a split.
    - Maximum mean discrepancy (MMD) to estimate the proximity of high-dimensional distributions of real and simulated cells without
      creating centroids.
    - Mean integration local inverse Simpson's Index (miLISI).
        iLISI captures the effective number of datatypes (real or simulated) to which datapoints of its local neighborhood belong.

    As a “control” (and to enable calibration of these scores), we also calculated these metrics using two halves of the reference test set.

    Parameters
    ----------
    cfg
        Parser for config file containing program params.

    Returns
    -------
    dict[str, float]
        The computed metrics, keyed by name. Also written to ``evaluation_results.json`` in the output directory.
    """
    output_dir = Path(cfg.get("EXPERIMENT", "output directory"))
    plt.rcParams.update({"savefig.dpi": 600})
    results = {}

    real_cells, fake_cells = read_datasets(cfg)

    # Split the test set into 2 to compute the control metrics
    num_rows = real_cells.shape[0]  # pyright: ignore[reportOptionalSubscript]
    half = num_rows // 2
    real_cells_ctr1 = real_cells[:half, :]
    real_cells_ctr2 = real_cells[half:, :]

    if cfg.getboolean("Evaluation", "plot umap") or cfg.getboolean("Evaluation", "compute miLISI"):
        real_embedding, fake_embedding = get_UMAP_embeddings(real_cells, fake_cells)
        scatter_fig, hexbin_fig, hist_diff_fig = plot_UMAP(real_embedding, fake_embedding)

        umap_path = output_dir / "UMAP"
        umap_path.mkdir(parents=True, exist_ok=True)

        scatter_fig.savefig(umap_path / "UMAP Scatter.png")
        scatter_fig.savefig(umap_path / "UMAP Scatter.pdf")
        scatter_fig.axes[0].set_rasterization_zorder(1.1)
        scatter_fig.savefig(umap_path / "UMAP Scatter Rasterized.pdf")
        hexbin_fig.savefig(umap_path / "UMAP Histogram.png")
        hexbin_fig.savefig(umap_path / "UMAP Histogram.pdf")
        hist_diff_fig.savefig(umap_path / "UMAP Histogram Difference.png")
        hist_diff_fig.savefig(umap_path / "UMAP Histogram Difference.pdf")
        plt.close("all")

    if cfg.getboolean("Evaluation", "compute euclidean distance") or cfg.getboolean(
        "Evaluation", "compute cosine distance"
    ):
        euclidean, cosine = compute_distances(real_cells, fake_cells)
        euclidean_ctr, cosine_ctr = compute_distances(real_cells_ctr1, real_cells_ctr2)

        logger.info(f"Euclidean distance (real vs fake): {euclidean}")
        logger.info(f"Euclidean distance (control): {euclidean_ctr}")

        logger.info(f"Cosine distance (real vs fake): {cosine}")
        logger.info(f"Cosine distance (control): {cosine_ctr}")

        results |= {
            "euclidean_distance_real_vs_fake": euclidean,
            "euclidean_distance_control": euclidean_ctr,
            "cosine_distance_real_vs_fake": cosine,
            "cosine_distance_control": cosine_ctr,
        }

    if cfg.getboolean("Evaluation", "compute rf auroc"):
        rf_auroc, fig = compute_RF_AUROC(
            real_cells,
            fake_cells,
        )
        fig.savefig(output_dir / "RF.png")
        fig.savefig(output_dir / "RF.pdf")
        plt.close(fig)
        logger.info(f"RF ROC plot saved to {output_dir / 'RF.png'}")
        logger.info(f"RF AUROC: {rf_auroc}")
        results |= {"rf_auroc": rf_auroc}

    if cfg.getboolean("Evaluation", "compute MMD"):
        mmd_real_vs_fake = MMD.MMD(real_cells).compute(real_cells, fake_cells)
        mmd_control = MMD.MMD(real_cells).compute(real_cells_ctr1, real_cells_ctr2)
        logger.info(f"MMD (real vs fake): {mmd_real_vs_fake}")
        logger.info(f"MMD (control): {mmd_control}")
        results |= {
            "mmd_real_vs_fake": mmd_real_vs_fake,
            "mmd_control": mmd_control,
        }

    if cfg.getboolean("Evaluation", "compute miLISI"):
        umap_real_ctr1, umap_real_ctr2 = get_UMAP_embeddings(real_cells_ctr1, real_cells_ctr2)

        umap_coords = np.vstack((real_embedding, fake_embedding))  # pyright: ignore[reportPossiblyUnboundVariable]
        umap_coords_ctr = np.vstack((umap_real_ctr1, umap_real_ctr2))

        metadata = pd.DataFrame(
            ["real"] * real_cells.shape[0] + ["generated"] * fake_cells.shape[0],  # pyright: ignore[reportOptionalSubscript]
            columns=["type"],
        )
        metadata_ctr = pd.DataFrame(
            ["ctr1"] * umap_real_ctr1.shape[0] + ["ctr2"] * umap_real_ctr2.shape[0],
            columns=["type"],
        )

        lisis = compute_lisi(umap_coords, metadata, ["type"])
        lisis_ctr = compute_lisi(umap_coords_ctr, metadata_ctr, ["type"])
        lisis = np.mean(lisis)
        lisis_ctr = np.mean(lisis_ctr)
        logger.info(f"miLISI (real vs fake): {lisis}")
        logger.info(f"miLISI (control): {lisis_ctr}")
        results |= {
            "miLISI_real_vs_fake": float(lisis),
            "miLISI_control": float(lisis_ctr),
        }

    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def aggregate_evaluations(runs: list[dict[str, float]], output_path: Path) -> None:
    """
    Report the mean and sample standard deviation of each metric across repeated evaluations.

    Parameters
    ----------
    runs
        Metrics returned by `evaluate`, one dictionary per run. All runs must contain the same keys.
    output_path
        Path of the JSON file the summary is written to.
    """
    logger.info(f"Aggregating results from {len(runs)} runs...")
    
    metrics: dict[str, dict[str, float | list[float]]] = {}
    for key in runs[0]:
        values = np.array([run[key] for run in runs], dtype=float)
        mean, std = float(values.mean()), float(values.std(ddof=1))
        metrics[key] = {"mean": mean, "std": std, "values": values.tolist()}
        logger.info(f"{key}: {mean:.6g} +/- {std:.6g}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"n_runs": len(runs), "metrics": metrics}, f, indent=2)
    logger.info(f"Evaluation range summary saved to {output_path}")
