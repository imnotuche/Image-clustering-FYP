import torch
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import euclidean_distances

from feature_extractor import DinoFeatureExtractor
from embedding_model import ProjectionHead
from dimension_reduction import DimensionReducer


def run_inference(
    projection_head: ProjectionHead,
    dimension_reducer: DimensionReducer,
    min_cluster_size: int = 50,
    min_samples: int = 5,
    device: str = 'cpu',
    # Option A: pass pre-extracted DINO embeddings directly
    dino_features: torch.Tensor = None,
    raw_images: torch.Tensor = None,
    # Option B: pass a dataloader + extractor and extract on the fly
    dataloader = None,
    dino_extractor: DinoFeatureExtractor = None,
    limit: int = None
) -> tuple:
    """
    Clusters a batch of images using the trained projection head.

    Two usage modes:

    Mode A — embeddings already extracted (preferred, avoids redundant work):
        run_inference(
            projection_head=head,
            dimension_reducer=reducer,
            dino_features=features,   # (N, 384) tensor
            raw_images=images,        # (N, C, H, W) tensor  [optional]
        )

    Mode B — raw images, extract on the fly:
        run_inference(
            projection_head=head,
            dimension_reducer=reducer,
            dataloader=loader,
            dino_extractor=dino,
        )

    Pipeline:
        DINO embeddings (384d)
        -> ProjectionHead  -> (N, 128)
        -> UMAP transform  -> (N, 50)
        -> fresh HDBSCAN   -> cluster labels

    HDBSCAN is fit fresh on each batch so the system adapts to whatever
    categories are present in the user's images.

    Returns:
        predictions: np.ndarray (N,) — cluster labels (-1 = noise)
        raw_images:  torch.Tensor (N, C, H, W) — original images (None if not provided)
        strengths:   np.ndarray (N,) — confidence scores 0-1
    """
    device_obj = torch.device(device)
    projection_head = projection_head.to(device_obj).eval()

    # -- Get DINO features ----------------------------------------------------
    if dino_features is not None:
        # Mode A: already have embeddings, nothing to extract
        features_np = dino_features.numpy() if isinstance(dino_features, torch.Tensor) else dino_features
        if limit is not None:
            features_np = features_np[:limit]
            if raw_images is not None:
                raw_images = raw_images[:limit]
        print(f"Using {len(features_np)} pre-extracted DINO embeddings.")

    elif dataloader is not None and dino_extractor is not None:
        # Mode B: extract from dataloader
        print("Extracting DINO features from images...")
        all_images   = []
        all_features = []
        total_count  = 0

        for images, _ in dataloader:
            if limit is not None and total_count >= limit:
                break
            dino_feats = dino_extractor.extract_batch(images)   # (B, 384)
            all_images.append(images.cpu())
            all_features.append(dino_feats)
            total_count += images.size(0)

        raw_images  = torch.cat(all_images, dim=0)
        features_np = torch.cat(all_features, dim=0).numpy()

        if limit is not None:
            raw_images  = raw_images[:limit]
            features_np = features_np[:limit]

        print(f"Extracted features for {len(features_np)} images.")

    else:
        raise ValueError(
            "Provide either (dino_features) or (dataloader + dino_extractor)."
        )

    # -- Projection head ------------------------------------------------------
    features_t   = torch.tensor(features_np, dtype=torch.float32).to(device_obj)
    with torch.no_grad():
        proj_features = projection_head(features_t).cpu().numpy()   # (N, 128)

    # -- UMAP transform -------------------------------------------------------
    reduced = dimension_reducer.transform(proj_features)   # (N, 50)

    # -- Fresh HDBSCAN on this batch ------------------------------------------
    print("Fitting HDBSCAN on this batch...")
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom',
        store_centers='centroid',
        copy=True
    )
    predictions = clusterer.fit_predict(reduced)   # (N,)

    n_clusters = len(set(predictions)) - (1 if -1 in predictions else 0)
    n_noise    = int(np.sum(predictions == -1))
    print(f"Clusters found: {n_clusters}")
    print(f"Noise images:   {n_noise} ({n_noise / len(predictions) * 100:.1f}%)")

    # -- Confidence scores ----------------------------------------------------
    centroids    = clusterer.centroids_
    distances    = euclidean_distances(reduced, centroids)
    nearest_dist = np.min(distances, axis=1)
    max_dist     = np.max(nearest_dist) + 1e-8
    strengths    = 1.0 - (nearest_dist / max_dist)
    strengths[predictions == -1] = 0.0

    return predictions, raw_images, strengths