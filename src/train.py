import numpy as np
from sklearn.cluster import HDBSCAN
from dimension_reduction import get_reduced_features

def train_hdbscan(
    raw_features,
    min_cluster_size=15,
    min_samples=5,
    embedding_dim=50,
    umap_model_path="./models/umap_model.pkl"
):
    """
    Steps:
    1. Reduce 2048d ResNet features to 50d via UMAP
    2. Fit HDBSCAN on the reduced features
    3. Save both models for inference
    
    Returns: labels (N,) where -1 means noise/unclustered
    """

    # Step 1: Reduce dimensions
    reduced_features = get_reduced_features(
        raw_features=raw_features,
        n_dims=embedding_dim,
        umap_model_path=umap_model_path
    )

    # Step 2: Fit HDBSCAN
    print("--- Fitting HDBSCAN... ---")
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom',  # excess of mass - finds stable clusters
        store_centers='centroid',         # needed for inference on new points
        copy=True
    )

    labels = clusterer.fit_predict(reduced_features)

    # Report what was found
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    print(f"--- HDBSCAN found {n_clusters} clusters ---")
    print(f"--- {n_noise} images labelled as noise ({n_noise/len(labels)*100:.1f}%) ---")

    return labels, reduced_features, clusterer