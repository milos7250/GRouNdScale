import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from loggers import setup_logger

logger = setup_logger(__name__)


class MMD:
    """
    Maximum Mean Discrepancy (MMD) class for computing distribution similarity
    between real and generated samples using Gaussian kernels.
    """

    def __init__(self, real_cells: np.ndarray | csr_matrix):
        """
        Initialize the MMD class with scale and weight parameters based on the median
        nearest neighbor distance among real cells.

        Parameters
        ----------
        real_cells
            A NumPy array representing real cell data (cells x features).
        """
        n_neighbors = 25
        n_resamples = 20
        med = np.empty(n_resamples)

        logger.info("Calculating MMD scale parameters...")
        for ii in range(n_resamples):
            # bootstrap: sample all cells with replacement
            sample = real_cells[np.random.randint(real_cells.shape[0], size=real_cells.shape[0]), :]  # pyright: ignore[reportOptionalSubscript]
            # each cell is its own nearest neighbour, so query one extra neighbour and drop it
            nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(sample)
            distances, _ = nbrs.kneighbors(sample)
            med[ii] = np.median(distances[:, 1:])

        med = np.median(med)
        scales = np.array([med / 2, med, med * 2], dtype=np.float32)
        weights = np.ones(len(scales), dtype=np.float32)

        # Reshape for broadcasting
        self.scales = scales.reshape(-1, 1, 1)
        self.weights = weights.reshape(-1, 1, 1)

    def squared_distance(
        self,
        x: csr_matrix,
        y: csr_matrix,
    ) -> np.ndarray:
        """
        Compute pairwise squared Euclidean distances between rows of X and Y.

        Parameters
        ----------
        x
            Input matrix of shape (n, d).
        y
            Input matrix of shape (m, d).

        Returns
        -------
        csr_matrix
            A matrix of shape (n, m) representing squared distances.
        """
        # X is nxd, Y is mxd, returns nxm matrix of all pairwise Euclidean distances squared
        return pairwise_distances(x, y) ** 2

    def gaussian_kernel(
        self,
        a: csr_matrix,
        b: csr_matrix,
    ) -> np.ndarray:
        """
        Compute the multi-scale Gaussian kernel between two datasets.

        Parameters
        ----------
        a
            Input matrix of shape (n, d).
        b
            Input matrix of shape (m, d).

        Returns
        -------
        np.ndarray
            A matrix of shape (n, m) representing the Gaussian kernel matrix.
        """
        numerator = self.squared_distance(a, b)[np.newaxis, :, :]  # shape: (1, n, m)
        kernel = np.sum(self.weights * np.exp(-numerator / (self.scales**2)), axis=0)
        return kernel

    def compute(
        self,
        a: csr_matrix,
        b: csr_matrix,
    ) -> float:
        """
        Compute the Maximum Mean Discrepancy (MMD) between two samples.

        Parameters
        ----------
        a
            First sample of shape (n, d).
        b
            Second sample of shape (m, d).

        Returns
        -------
        float
            The MMD score between the two distributions.
        """
        logger.info("Calculating MMD score...")
        return (
            self.gaussian_kernel(a, a).mean()
            + self.gaussian_kernel(b, b).mean()
            - 2 * self.gaussian_kernel(a, b).mean()
        ).item()
