import numpy as np
import joblib
import os
import torch
import umap


class DimensionReducer:
    """
    Wraps UMAP for dimensionality reduction.

    - fit_transform: used during training. Fits UMAP on the training embeddings
    and saves the fitted model to disk.
    - transform: used at inference. Loads the saved model and projects new
    embeddings into the existing coordinate system without refitting.

    Why this split matters: if you refit UMAP on new data, the coordinate
    system changes and the cluster centroids from training become meaningless.
    transform preserves the coordinate system so new images land in the same
    space as the training images.
    """

    def __init__(self, n_components: int = 50, n_neighbors: int = 30,
                min_dist: float = 0.1, model_path: str = "./models/umap_model.pkl"):
        self.n_components = n_components
        self.n_neighbors  = n_neighbors
        self.min_dist     = min_dist
        self.model_path   = model_path
        self.reducer      = None

    def fit_transform(self, features) -> np.ndarray:
        """
        Fit UMAP on features and return reduced array.
        Saves fitted model to self.model_path.

        Args:
            features: torch.Tensor or np.ndarray of shape (N, D)

        Returns:
            np.ndarray of shape (N, n_components)
        """
        if isinstance(features, torch.Tensor):
            features = features.numpy()

        print(f"Fitting UMAP: {features.shape[1]}d -> {self.n_components}d "
            f"(n_neighbors={self.n_neighbors}, min_dist={self.min_dist})...")

        self.reducer = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric='cosine',
            random_state=42,
            verbose=True
        )

        reduced = self.reducer.fit_transform(features)

        # Save fitted model
        joblib.dump(self.reducer, self.model_path)
        print(f"UMAP model saved to {self.model_path}")

        return reduced  # (N, n_components)

    def transform(self, features) -> np.ndarray:
        """
        Project new features into the existing UMAP coordinate system.
        Loads model from disk if not already loaded in memory.

        Args:
            features: torch.Tensor or np.ndarray of shape (N, D)

        Returns:
            np.ndarray of shape (N, n_components)
        """
        if isinstance(features, torch.Tensor):
            features = features.numpy()

        if self.reducer is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"No UMAP model found at {self.model_path}. "
                    "Run fit_transform on training data first."
                )
            print(f"Loading UMAP model from {self.model_path}...")
            self.reducer = joblib.load(self.model_path)

        print("Projecting into existing UMAP space...")
        return self.reducer.transform(features)  # (N, n_components)
