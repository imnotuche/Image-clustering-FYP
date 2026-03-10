import numpy as np
import torch
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import euclidean_distances


class PseudoLabelGenerator:
    """
    Generates pseudo-labels from HDBSCAN for use as training targets.

    The key design decision here is noise reassignment:
    HDBSCAN marks sparse/outlier images as noise (-1). Rather than discarding
    them (which wastes training data and biases the training set), we assign
    each noise image to its nearest cluster centroid using Euclidean distance
    in the UMAP-reduced space. This ensures every image gets a pseudo-label
    and contributes to training.

    Centroids are the mean of all non-noise points in each cluster.
    """

    def __init__(self, min_cluster_size: int = 50, min_samples: int = 5):
        self.min_cluster_size = min_cluster_size
        self.min_samples      = min_samples
        self.clusterer        = None
        self.centroids        = None   # (K, n_components) — saved for inference

    def fit_predict(self, reduced_features: np.ndarray) -> np.ndarray:
        """
        Run HDBSCAN on UMAP-reduced features, then reassign noise points.

        Args:
            reduced_features: np.ndarray of shape (N, n_components)

        Returns:
            pseudo_labels: np.ndarray of shape (N,) — no -1 values
        """
        print(f"Running HDBSCAN (min_cluster_size={self.min_cluster_size}, "
            f"min_samples={self.min_samples})...")

        self.clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            store_centers='centroid',
            copy=True
        )

        raw_labels = self.clusterer.fit_predict(reduced_features)

        n_clusters = len(set(raw_labels)) - (1 if -1 in raw_labels else 0)
        n_noise    = int(np.sum(raw_labels == -1))
        print(f"HDBSCAN found {n_clusters} clusters, {n_noise} noise points "
              f"({n_noise / len(raw_labels) * 100:.1f}%)")

        if n_clusters == 0:
            raise RuntimeError(
                "HDBSCAN found 0 clusters. Try lowering min_cluster_size."
            )

        # Compute centroids as mean of non-noise points per cluster
        self.centroids = self.clusterer.centroids_   # (K, n_components)

        # Noise reassignment
        pseudo_labels = raw_labels.copy()
        noise_mask    = raw_labels == -1

        if noise_mask.any():
            noise_features = reduced_features[noise_mask]           # (M, n_components)
            distances       = euclidean_distances(noise_features, self.centroids)  # (M, K)
            nearest_cluster = np.argmin(distances, axis=1)          # (M,)
            pseudo_labels[noise_mask] = nearest_cluster
            print(f"Reassigned {n_noise} noise points to nearest centroids.")

        assert (pseudo_labels >= 0).all(), "Noise reassignment failed — -1 labels remain."
        print(f"Pseudo-labels ready. {n_clusters} unique labels, "
            f"{len(pseudo_labels)} total images.")

        return pseudo_labels   # (N,) — all non-negative

    def get_centroids(self) -> np.ndarray:
        if self.centroids is None:
            raise RuntimeError("fit_predict must be called before get_centroids.")
        return self.centroids
