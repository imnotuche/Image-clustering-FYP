import umap
import joblib
import numpy as np

def get_reduced_features(raw_features, n_dims=50, umap_model_path="./models/umap_model.pkl"):
    """
    Reduces high-dimensional ResNet50 embeddings (2048d) down to n_dims
    using UMAP, preserving neighbourhood structure.
    
    raw_features: numpy array or torch tensor of shape (N, 2048)
    n_dims: target dimensions (50 for HDBSCAN, 2 for visualisation)
    """
    print(f"--- Reducing dimensions via UMAP: 2048 -> {n_dims}... ---")
    
    if hasattr(raw_features, 'numpy'):
        raw_features = raw_features.cpu().numpy()
    
    reducer = umap.UMAP(
        n_components=n_dims,
        n_neighbors=30,      # how many neighbours to consider per point
        min_dist=0.1,        # how tightly points can be packed together
        metric='cosine',     # cosine works well for normalized ResNet embeddings
        random_state=42,
        verbose=True
    )
    
    reduced = reducer.fit_transform(raw_features)
    
    # Save the fitted UMAP model so inference can reuse it
    joblib.dump(reducer, umap_model_path)
    print(f"UMAP model saved to {umap_model_path}")
    
    return reduced