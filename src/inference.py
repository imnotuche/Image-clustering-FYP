import torch
import numpy as np
import joblib
from sklearn.metrics.pairwise import euclidean_distances

def run_inference(dataloader, feature_extractor, device, umap_model_path, hdbscan_model, limit=None):
    """
    For new unseen images:
    1. Extract ResNet50 embeddings
    2. Transform with saved UMAP (reuses the fitted reducer, does not refit)
    3. Assign to nearest cluster centroid from saved HDBSCAN
    
    Returns: predictions, raw_images, strengths
    -1 in predictions means the image was too far from any cluster centroid (noise)
    """

    # Load saved models
    print("Loading UMAP and HDBSCAN models...")
    umap_reducer = joblib.load(umap_model_path)
    clusterer = hdbscan_model

    # Get cluster centroids saved during training
    # store_centers='centroid' must have been set during fit
    centroids = clusterer.centroids_

    feature_extractor.to(device)
    feature_extractor.eval()

    all_images = []
    all_raw_feats = []
    total_count = 0

    print("Extracting features from new images...")
    with torch.no_grad():
        for images, _ in dataloader:
            if limit is not None and total_count >= limit:
                break

            images = images.to(device)
            raw_feats = feature_extractor(images)

            all_images.append(images.cpu())
            all_raw_feats.append(raw_feats.cpu().numpy())
            total_count += images.size(0)

    final_images = torch.cat(all_images)
    if limit is not None:
        final_images = final_images[:limit]

    raw_feats_np = np.concatenate(all_raw_feats)
    if limit is not None:
        raw_feats_np = raw_feats_np[:limit]

    # Transform with saved UMAP (transform not fit_transform)
    print("Projecting into UMAP space...")
    reduced = umap_reducer.transform(raw_feats_np)

    # Assign each point to its nearest centroid
    print("Assigning to nearest cluster centroids...")
    distances = euclidean_distances(reduced, centroids)
    nearest_cluster = np.argmin(distances, axis=1)
    nearest_distance = np.min(distances, axis=1)

    # Points that are very far from all centroids get flagged as noise
    # Threshold: anything beyond 3x the median distance is noise
    noise_threshold = np.median(nearest_distance) * 3
    predictions = np.where(nearest_distance <= noise_threshold, nearest_cluster, -1)

    # Strength: inverse of distance, normalised to 0-1
    max_dist = np.max(nearest_distance)
    strengths = 1 - (nearest_distance / (max_dist + 1e-8))
    strengths[predictions == -1] = 0.0

    n_noise = int(np.sum(predictions == -1))
    print(f"--- {len(predictions)} images processed ---")
    print(f"--- {n_noise} flagged as noise ({n_noise/len(predictions)*100:.1f}%) ---")

    return predictions, final_images, strengths