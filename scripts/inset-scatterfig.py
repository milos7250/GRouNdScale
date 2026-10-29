#!/usr/bin/env python3
"""Plot UMAP embeddings of an arbitrary number of cell groups, showing the smaller of two disjoint regions as a zoomed inset to reduce whitespace."""

import sys
from pathlib import Path

# Add parent folder to pythonpath so `src` is importable when run as a script.
sys.path.insert(0, Path(__file__).resolve().parent.parent.as_posix())

from src.loggers import setup_logger

logger = setup_logger("inset_scatterfig")

from typing import TYPE_CHECKING

import rich_click as click

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.figure import Figure
    from scipy import sparse


_DEFAULT_COLORS: tuple[str, ...] = (
    "blue",
    "red",
    "green",
    "orange",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
)
_CORNER_LOCATIONS: tuple[str, ...] = ("upper right", "upper left", "lower right", "lower left")
_LEGEND_LOCATIONS: tuple[str, ...] = (
    "best",
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "right",
    "center left",
    "center right",
    "lower center",
    "upper center",
    "center",
)
_INSET_LOCATIONS: tuple[str, ...] = ("best", "upper right", "upper left", "lower right", "lower left", "center")


def _category_colors(n: int) -> list[str]:
    """Return ``n`` distinct colour names, cycling the default palette if ``n`` exceeds it."""
    if n <= len(_DEFAULT_COLORS):
        return list(_DEFAULT_COLORS[:n])
    logger.warning("More than %d cell groups requested; colours will cycle.", len(_DEFAULT_COLORS))
    return [_DEFAULT_COLORS[i % len(_DEFAULT_COLORS)] for i in range(n)]


def read_datasets(cells_paths: list[Path], max_cells: int | None = None, seed: int = 42) -> list["sparse.csr_matrix"]:
    """
    Load gene expression datasets from H5AD files.

    Each file is read into a sparse matrix and all datasets are reduced to a common cell count so
    every group contributes the same number of cells. The common count is the smallest row count,
    further capped by ``max_cells`` if given. Cells are randomly subsampled (without replacement,
    keeping their original order) rather than taking the first rows, so ordering in the H5AD files
    does not bias the sample.

    Parameters
    ----------
    cells_paths
        Paths to the H5AD files, one per cell group.
    max_cells
        Optional upper limit on the number of cells per group.
    seed
        Seed for the random subsampling.

    Returns
    -------
    list["sparse.csr_matrix"]
        Sparse matrices (one per input file), each reduced to the common cell count.
    """
    import numpy as np
    import scanpy as sc
    from scipy import sparse

    matrices = [sparse.csr_matrix(sc.read_h5ad(p).X) for p in cells_paths]
    no_of_cells = int(min(m.shape[0] for m in matrices))
    if max_cells is not None:
        no_of_cells = min(no_of_cells, max_cells)
    logger.info("Using %d cells per group", no_of_cells)

    rng = np.random.default_rng(seed)
    datasets = []
    for m in matrices:
        if m.shape[0] > no_of_cells:
            idx = np.sort(rng.choice(m.shape[0], size=no_of_cells, replace=False))
            m = m[idx, :]
        datasets.append(m)
    return datasets


def get_UMAP_embeddings(datasets: list["sparse.csr_matrix"]) -> list["np.ndarray"]:
    """
    Compute 2D UMAP embeddings for each dataset.

    UMAP is fitted once on the first dataset and used to transform every dataset, so the embeddings
    are comparable across groups. Put the reference (e.g. real) dataset first.

    Parameters
    ----------
    datasets
        Sparse matrices of shape (n_cells, n_features), one per group.

    Returns
    -------
    list["np.ndarray"]
        2D UMAP embeddings, one array per dataset, in the same order as the input.
    """
    import numpy as np
    from umap import UMAP

    umap = UMAP(random_state=42, min_dist=0.0, n_jobs=1)
    umap.fit(datasets[0])  # fit only once on the first dataset to preserve comparability
    return [np.array(umap.transform(d)) for d in datasets]


def _interleave(points_per_category: list["np.ndarray"]) -> tuple["np.ndarray", "np.ndarray"]:
    """
    Round-robin interleave points from each category so colours cycle on the scatter plot.

    Parameters
    ----------
    points_per_category
        List of ``(n_i, 2)`` arrays, one per category in registration order.

    Returns
    -------
    tuple["np.ndarray", "np.ndarray"]
        ``(interleaved_points, category_index)`` where ``interleaved_points`` has shape
        ``(sum(n_i), 2)`` and ``category_index`` maps each row back to its source category.
    """
    import numpy as np

    counts = [int(p.shape[0]) for p in points_per_category]
    total = sum(counts)
    if total == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=np.int64)
    width = int(points_per_category[0].shape[1])
    dtype = np.result_type(*points_per_category)
    out = np.empty((total, width), dtype=dtype)
    cat = np.empty(total, dtype=np.int64)
    idx = [0] * len(points_per_category)
    pos = 0
    while pos < total:
        progressed = False
        for k in range(len(points_per_category)):
            if idx[k] < counts[k]:
                out[pos] = points_per_category[k][idx[k]]
                cat[pos] = k
                idx[k] += 1
                pos += 1
                progressed = True
        if not progressed:
            break
    return out, cat


def split_into_regions(points: "np.ndarray", centres: list[tuple[float, float]] | None = None) -> "np.ndarray":
    """
    Split 2D points into two spatial regions: the larger (more points) labelled ``0`` (main) and the
    rest labelled ``1`` (inset).

    If ``centres`` are supplied, each point is assigned to its nearest centre; the largest resulting
    group becomes the main region and all other points the inset region. Otherwise the two regions
    are found by splitting at the largest gap along the best of three 1D projections (the first
    principal component and the x / y axes), choosing the projection whose split yields the most
    separated groups. This gap-based split is more robust than density clustering (HDBSCAN), which
    tends to over-segment a region into several adjacent sub-clusters so that the "two largest
    clusters" both lie in the same region. Callers should additionally verify the two regions are
    well-separated before insetting.

    Parameters
    ----------
    points
        ``(n, 2)`` array of 2D embeddings. Not modified.
    centres
        Optional list of ``(x, y)`` region centres for manual assignment.

    Returns
    -------
    "np.ndarray"
        Integer labels of shape ``(n,)`` with values in ``{0, 1}``.
    """
    import numpy as np

    n = points.shape[0]
    if n < 2:
        return np.zeros(n, dtype=np.int64)

    if centres is not None and len(centres) >= 2:
        cen = np.asarray(centres, dtype=float)
        assign = np.argmin(np.linalg.norm(points[:, None, :] - cen[None, :, :], axis=2), axis=1)
    else:
        assign = _largest_gap_split(points)

    counts = np.bincount(assign)
    main_label = int(np.argmax(counts))
    return np.where(assign == main_label, 0, 1).astype(np.int64)


def _largest_gap_split(points: "np.ndarray") -> "np.ndarray":
    """Split points into two groups at the largest gap along the best of PC1 / x / y axes.

    The "best" axis is the one whose largest-gap split produces the most separated (by bounding-box
    gap) pair of groups, so the main/outlier separation is chosen directly rather than via an
    arbitrary projection.
    """
    import numpy as np
    from sklearn.decomposition import PCA

    pc1 = PCA(n_components=1).fit_transform(points).ravel()
    candidates = [pc1, points[:, 0], points[:, 1]]
    best_gap = -np.inf
    best_labels = np.zeros(points.shape[0], dtype=np.int64)
    for proj in candidates:
        order = np.argsort(proj)
        s = proj[order]
        gaps = np.diff(s)
        if gaps.size == 0:
            continue
        i = int(np.argmax(gaps))
        labels = np.where(proj <= s[i], 0, 1)
        g0 = points[labels == 0]
        g1 = points[labels == 1]
        if g0.shape[0] == 0 or g1.shape[0] == 0:
            continue
        gap = _bbox_gap(g0, g1)
        if gap > best_gap:
            best_gap = gap
            best_labels = labels
    return best_labels


def _bbox_gap(region0: "np.ndarray", region1: "np.ndarray") -> float:
    """Maximum spatial gap between the bounding boxes of two point sets (0 if they overlap)."""
    b0 = _region_bbox(region0, margin_frac=0.0)
    b1 = _region_bbox(region1, margin_frac=0.0)
    gap_x = max(b0[0, 0], b1[0, 0]) - min(b0[0, 1], b1[0, 1])
    gap_y = max(b0[1, 0], b1[1, 0]) - min(b0[1, 1], b1[1, 1])
    return float(max(gap_x, gap_y))


def _region_bbox(points: "np.ndarray", margin_frac: float = 0.05) -> "np.ndarray":
    """
    Bounding box of ``points`` plus a symmetric fractional margin.

    Returns
    -------
    "np.ndarray"
        Array of shape ``(2, 2)`` as ``[[xmin, xmax], [ymin, ymax]]``.
    """
    import numpy as np

    if points.shape[0] == 0:
        return np.array([[0.0, 1.0], [0.0, 1.0]])
    extent = np.array([[points[:, 0].min(), points[:, 0].max()], [points[:, 1].min(), points[:, 1].max()]], dtype=float)
    margin = (extent[:, 1] - extent[:, 0]) * margin_frac
    extent[:, 0] -= margin[0]
    extent[:, 1] += margin[1]
    return extent


def _regions_well_separated(region0: "np.ndarray", region1: "np.ndarray", min_gap_frac: float = 0.05) -> bool:
    """
    Return True if the two regions' bounding boxes are separated by a meaningful gap.

    Used to avoid insetting when a single blob is over-segmented into adjacent pieces, since
    insetting only reduces whitespace when the two regions are genuinely disjoint.
    """
    import numpy as np

    if region0.shape[0] == 0 or region1.shape[0] == 0:
        return False
    b0 = _region_bbox(region0, margin_frac=0.0)
    b1 = _region_bbox(region1, margin_frac=0.0)
    gap_x = max(b0[0, 0], b1[0, 0]) - min(b0[0, 1], b1[0, 1])
    gap_y = max(b0[1, 0], b1[1, 0]) - min(b0[1, 1], b1[1, 1])
    overall = np.vstack([region0, region1])
    overall_extent = float(max(overall[:, 0].max() - overall[:, 0].min(), overall[:, 1].max() - overall[:, 1].min()))
    if overall_extent <= 0:
        return False
    return max(gap_x, gap_y) > min_gap_frac * overall_extent


def _sibling_inset_axes(
    fig: "Figure", main_ax, corner: str, inset_width: float, inset_height: float, pad: float = 0.02
):
    """
    Create a *sibling* inset axes (via ``fig.add_axes``) at ``corner`` of the main axes.

    A sibling (top-level) axes is used rather than ``inset_axes`` (which parents the inset under the
    main axes) for two reasons: (1) a child inset corrupts rasterization of the main axes' artists in
    the PDF backend, squishing rasterized points into a corner; (2) a sibling axes is drawn *after*
    the main axes, so its opaque white patch reliably covers the main axes' grid/points, giving the
    inset a clean white background. The position is computed directly (no locatable locator), so it
    is known before draw.
    """
    pos = main_ax.get_position()
    iw = inset_width * (pos.x1 - pos.x0)
    ih = inset_height * (pos.y1 - pos.y0)
    if corner == "upper right":
        left, bottom = pos.x1 - iw - pad, pos.y1 - ih - pad
    elif corner == "upper left":
        left, bottom = pos.x0 + pad, pos.y1 - ih - pad
    elif corner == "lower right":
        left, bottom = pos.x1 - iw - pad, pos.y0 + pad
    elif corner == "lower left":
        left, bottom = pos.x0 + pad, pos.y0 + pad
    else:  # center
        left = pos.x0 + (pos.x1 - pos.x0 - iw) / 2
        bottom = pos.y0 + (pos.y1 - pos.y0 - ih) / 2
    return fig.add_axes([left, bottom, iw, ih])


def _orient_inset_ticks(ax_inset, corner: str) -> None:
    """
    Put the inset's ticks on the sides that face the main axes interior.

    For a corner placement the interior-facing sides are those toward the main axes centre, e.g. an
    upper-right inset gets x ticks on the bottom and y ticks on the left. This keeps the inset's
    tick labels inside the main axes and away from the main axes' outer labels.
    """
    if corner == "upper right":
        ax_inset.xaxis.tick_bottom()
        ax_inset.yaxis.tick_left()
    elif corner == "upper left":
        ax_inset.xaxis.tick_bottom()
        ax_inset.yaxis.tick_right()
    elif corner == "lower right":
        ax_inset.xaxis.tick_top()
        ax_inset.yaxis.tick_left()
    elif corner == "lower left":
        ax_inset.xaxis.tick_top()
        ax_inset.yaxis.tick_right()
    else:  # center: fall back to the default bottom/left placement
        ax_inset.xaxis.tick_bottom()
        ax_inset.yaxis.tick_left()


def _best_inset_corner(main_points: "np.ndarray") -> tuple[str, dict[str, int]]:
    """
    Return the corner of the main axes containing the fewest points so the inset covers little data.

    Returns
    -------
    tuple[str, dict[str, int]]
        The chosen corner and a mapping from each corner name to the number of main-region points
        that fall in it.
    """

    counts: dict[str, int] = dict.fromkeys(_CORNER_LOCATIONS, 0)
    if main_points.shape[0] == 0:
        return "upper right", counts
    xs = main_points[:, 0]
    ys = main_points[:, 1]
    x_mid = (xs.min() + xs.max()) / 2.0
    y_mid = (ys.min() + ys.max()) / 2.0
    counts = {
        "upper right": int(((xs >= x_mid) & (ys >= y_mid)).sum()),
        "upper left": int(((xs < x_mid) & (ys >= y_mid)).sum()),
        "lower right": int(((xs >= x_mid) & (ys < y_mid)).sum()),
        "lower left": int(((xs < x_mid) & (ys < y_mid)).sum()),
    }
    corner = min(_CORNER_LOCATIONS, key=lambda loc: (counts[loc], _CORNER_LOCATIONS.index(loc)))
    return corner, counts


def _scatter_interleaved(ax, points_per_category: list["np.ndarray"], colors: list[str]) -> None:
    """Interleave the category points and scatter them on ``ax`` with cycled colours."""
    import numpy as np

    pts, cat = _interleave(points_per_category)
    color_array = np.array(colors)[cat]
    ax.scatter(pts[:, 0], pts[:, 1], c=color_array, s=3, edgecolor="none")


def _add_legend(ax, location: str, names: list[str], colors: list[str]) -> None:
    """Attach the shared legend (one handle per cell group) to ``ax``."""
    handles = [ax.scatter([], [], c=colors[i], label=names[i], s=3, edgecolor="none") for i in range(len(names))]
    ax.legend(handles=handles, loc=location, ncol=1, fontsize=8).set(zorder=5)


def plot_UMAP_inset(
    embeddings: list["np.ndarray"],
    names: list[str],
    colors: list[str],
    legend_location: str = "lower left",
    inset_loc: str = "best",
    inset_width: float = 0.35,
    inset_height: float = 0.35,
    centres: list[tuple[float, float]] | None = None,
) -> "Figure":
    """
    Scatter the UMAP embeddings with the smaller of two disjoint regions shown as a zoomed inset.

    The main axes are cropped to the larger region's bounding box, eliminating the whitespace
    that the gap between the two regions would otherwise create. The smaller region is plotted on
    an inset axes zoomed to its own bounding box and placed in the least-dense corner of the main
    axes. If two disjoint regions cannot be detected, a plain full-extent UMAP scatter is produced
    instead.

    Parameters
    ----------
    embeddings
        2D UMAP embeddings, one array per cell group.
    names
        Legend name per cell group.
    colors
        Colour name per cell group.
    legend_location
        Location of the shared legend on the main axes. Defaults to ``"lower left"``.
    inset_loc
        Placement of the inset axes. ``"best"`` picks the corner with the fewest main points.
        Defaults to ``"best"``.
    inset_width
        Inset width as a fraction of the main axes width. Defaults to ``0.35``.
    inset_height
        Inset height as a fraction of the main axes height. Defaults to ``0.35``.
    centres
        Optional list of ``(x, y)`` region centres. If given, each point is assigned to its nearest
        centre instead of the automatic largest-gap split.

    Returns
    -------
    Figure
        A matplotlib Figure object for the scatter plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    all_points = np.vstack(embeddings)
    region = split_into_regions(all_points, centres=centres)
    has_inset = bool((region == 0).any() and (region == 1).any())
    if has_inset and not _regions_well_separated(all_points[region == 0], all_points[region == 1]):
        has_inset = False

    if has_inset:
        main_mask = region == 0
        inset_mask = region == 1
        logger.info(
            "Inset identified: main region %d points (midpoint %s), smaller region %d points (midpoint %s)",
            int(main_mask.sum()),
            all_points[main_mask].mean(axis=0).tolist(),
            int(inset_mask.sum()),
            all_points[inset_mask].mean(axis=0).tolist(),
        )
    else:
        logger.warning("Two well-separated disjoint regions were not detected; producing a plain UMAP plot.")

    plt.clf()
    fig = plt.figure(figsize=(5, 5))
    ax = fig.subplots()

    if not has_inset:
        _scatter_interleaved(ax, embeddings, colors)
        extent = _region_bbox(all_points)
        ax.set_xlim(extent[0, 0], extent[0, 1])
        ax.set_ylim(extent[1, 0], extent[1, 1])
        ax.grid(True, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_title("UMAP Projection of Cells")
        _add_legend(ax, legend_location, names, colors)
        return fig

    sizes = [int(emb.shape[0]) for emb in embeddings]
    starts = [0]
    for s in sizes[:-1]:
        starts.append(starts[-1] + s)
    region_per_category = [region[starts[i] : starts[i] + sizes[i]] for i in range(len(embeddings))]
    main_per_category = [embeddings[i][region_per_category[i] == 0] for i in range(len(embeddings))]
    inset_per_category = [embeddings[i][region_per_category[i] == 1] for i in range(len(embeddings))]
    main_points = np.vstack(main_per_category)
    inset_points = np.vstack(inset_per_category)
    main_extent = _region_bbox(main_points)
    inset_extent = _region_bbox(inset_points)

    ax.set_xlim(main_extent[0, 0], main_extent[0, 1])
    ax.set_ylim(main_extent[1, 0], main_extent[1, 1])
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    _scatter_interleaved(ax, main_per_category, colors)
    ax.set_title("UMAP Projection of Cells")

    best_corner, corner_counts = _best_inset_corner(main_points)
    corner = best_corner if inset_loc == "best" else inset_loc
    logger.info("Inset corner point counts: %s; placing inset at '%s'", corner_counts, corner)
    # Sibling inset axes (drawn on top of the main axes) with an opaque white patch so the main
    # grid/points are hidden behind the inset.
    ax_inset = _sibling_inset_axes(fig, ax, corner, inset_width, inset_height)
    ax_inset.patch.set_facecolor("white")
    ax_inset.patch.set_alpha(1.0)
    _scatter_interleaved(ax_inset, inset_per_category, colors)
    ax_inset.set_xlim(inset_extent[0, 0], inset_extent[0, 1])
    ax_inset.set_ylim(inset_extent[1, 0], inset_extent[1, 1])
    # Inset ticks on the sides facing the main axes interior (so labels stay inside and do not
    # overlap the main axes' outer labels).
    _orient_inset_ticks(ax_inset, corner)
    ax_inset.grid(True, linestyle="--", linewidth=0.5)
    ax_inset.set_axisbelow(True)
    ax_inset.tick_params(labelsize=6)

    _add_legend(ax, legend_location, names, colors)
    return fig


def main(
    cells_paths: list[Path],
    names: list[str],
    output_dir: Path,
    legend_location: str = "best",
    inset_loc: str = "best",
    inset_width: float = 0.35,
    inset_height: float = 0.35,
    centres: list[tuple[float, float]] | None = None,
    max_cells: int | None = None,
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "savefig.dpi": 600,
        "legend.facecolor": (1.0, 1.0, 1.0, 0.0),
    })

    if len(names) != len(cells_paths):
        raise ValueError(f"Number of names ({len(names)}) must match number of input files ({len(cells_paths)}).")
    colors = _category_colors(len(names))

    logger.info("Reading datasets")
    datasets = read_datasets(cells_paths, max_cells=max_cells)

    logger.info("Computing UMAP embeddings")
    embeddings = get_UMAP_embeddings(datasets)

    logger.info("Plotting UMAP inset scatter plot")
    scatter_fig = plot_UMAP_inset(
        embeddings,
        names,
        colors,
        legend_location=legend_location,
        inset_loc=inset_loc,
        inset_width=inset_width,
        inset_height=inset_height,
        centres=centres,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    scatter_fig.savefig(output_dir / "UMAP Inset Scatter.png", transparent=False, facecolor=(1, 1, 1, 0))
    scatter_fig.savefig(output_dir / "UMAP Inset Scatter.pdf", transparent=False, facecolor=(1, 1, 1, 0))
    # Rasterize only the dense scatter collections (per-artist) for a compact PDF. Using
    # set_rasterization_zorder instead mis-renders the inset (child axes), squishing the
    # rasterized points into the lower-left corner.
    for ax in scatter_fig.axes:
        for coll in ax.collections:
            coll.set_rasterized(True)
    scatter_fig.savefig(output_dir / "UMAP Inset Scatter Rasterized.pdf", transparent=False, facecolor=(1, 1, 1, 0))
    logger.info(f"UMAP inset scatter plots saved to '{output_dir}'")
    plt.close("all")


def _parse_centres(spec: str) -> list[tuple[float, float]]:
    """
    Parse a cluster-centres specification of the form ``"x1,y1;x2,y2[;...]"``.

    Centres are separated by ``;`` and each centre is an ``x,y`` pair. Whitespace is ignored. At
    least two centres are required.
    """
    centres: list[tuple[float, float]] = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        coords = [c.strip() for c in part.split(",")]
        if len(coords) != 2:
            raise click.UsageError(f"Each cluster centre must be 'x,y' (got '{part}').")
        try:
            centres.append((float(coords[0]), float(coords[1])))
        except ValueError as exc:
            raise click.UsageError(f"Cluster centre coordinates must be numeric (got '{part}').") from exc
    if len(centres) < 2:
        raise click.UsageError("--cluster-centres must provide at least two 'x,y' centres separated by ';'.")
    return centres


@click.command()
@click.argument(
    "files",
    nargs=-1,
    required=True,
    type=click.Path(
        exists=True,
        dir_okay=False,
        file_okay=True,
        readable=True,
        path_type=Path,
    ),
)
@click.option(
    "--names",
    type=str,
    required=True,
    help="Comma-separated legend names, one per input file (e.g. 'real,original generated,improved generated').",
)
@click.option(
    "--out",
    type=click.Path(dir_okay=True, file_okay=False, writable=True, path_type=Path),
    required=True,
    help="Directory where the scatter plots will be saved.",
)
@click.option(
    "--legend-location",
    type=click.Choice(list(_LEGEND_LOCATIONS), case_sensitive=False),
    default="best",
    show_default=True,
    help="Location of the shared legend on the main axes.",
)
@click.option(
    "--inset-loc",
    type=click.Choice(list(_INSET_LOCATIONS), case_sensitive=False),
    default="best",
    show_default=True,
    help="Placement of the zoomed inset. 'best' picks the corner with the fewest main-region points.",
)
@click.option(
    "--inset-width",
    type=click.FloatRange(0.05, 1.0),
    default=0.35,
    show_default=True,
    help="Inset width as a fraction of the main axes width.",
)
@click.option(
    "--inset-height",
    type=click.FloatRange(0.05, 1.0),
    default=0.35,
    show_default=True,
    help="Inset height as a fraction of the main axes height.",
)
@click.option(
    "--cluster-centres",
    type=str,
    default=None,
    help="Optional manual region centres as 'x1,y1;x2,y2' (semicolon-separated). Each point is "
    "assigned to its nearest centre; the largest group is the main region, the rest the inset. "
    "Use when automatic detection fails.",
)
@click.option(
    "--max-cells",
    type=click.IntRange(min=1),
    default=None,
    help="Maximum number of cells per group, randomly subsampled (seed 42). Defaults to the smallest input file's cell count.",
)
def cli(
    files: tuple[Path, ...],
    names: str,
    out: Path,
    legend_location: str,
    inset_loc: str,
    inset_width: float,
    inset_height: float,
    cluster_centres: str | None,
    max_cells: int | None,
) -> None:
    """
    Plot UMAP embeddings of an arbitrary number of cell groups (one H5AD file each), showing the smaller of two disjoint regions as a zoomed inset to reduce whitespace. UMAP is fitted on the first file and used to transform all files.
    """
    name_list = [n.strip() for n in names.split(",")]
    if len(name_list) != len(files):
        raise click.UsageError(f"--names must provide {len(files)} comma-separated names (got {len(name_list)}).")
    centres = _parse_centres(cluster_centres) if cluster_centres else None
    main(
        list(files),
        name_list,
        out,
        legend_location,
        inset_loc,
        inset_width,
        inset_height,
        centres=centres,
        max_cells=max_cells,
    )


if __name__ == "__main__":
    cli()
