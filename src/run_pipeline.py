"""
run_pipeline.py — Training pipeline

Stages:
    1. Extract DINO ViT-S/8 embeddings from STL-10 training set
    2. Reduce 384d -> 50d with UMAP
    3. Generate HDBSCAN pseudo-labels + noise reassignment
    4. Mine nearest neighbours
    5. Train MLP projection head with neighbour agreement loss
    6. Save trained model + evaluate on training data

Run:
    python run_pipeline.py
"""

import os
import torch
import numpy as np
from torchvision import datasets

from data_manager import DataManager
from feature_extractor import DinoFeatureExtractor
from dimension_reduction import DimensionReducer
from pseudo_label_generator import PseudoLabelGenerator
from train import mine_neighbours, train_projection_head
from evaluate import evaluate_clustering
import joblib


def main():

    # --- Config --------------------------------------------------------------

    DEVICE           = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE       = 64
    MIN_CLUSTER_SIZE = 50
    MIN_SAMPLES      = 5
    UMAP_DIMS        = 50
    UMAP_NEIGHBOURS  = 30
    KNN_K            = 10
    EPOCHS           = 50
    LR               = 1e-4
    TEMPERATURE      = 0.1

    os.makedirs('./results', exist_ok=True)
    os.makedirs('./models', exist_ok=True)

    print(f"Device: {DEVICE}")
    print("=" * 55)

    # --- Step 1: Data loading ------------------------------------------------

    print("\n[1/5] Setting up DataManager and DataLoader...")

    stl10_manager = DataManager(
        path='./data/stl10',
        batch_size=BATCH_SIZE,
        device=DEVICE
    )

    train_loader = stl10_manager.get_loader(
        source=datasets.STL10,
        train=True,
        shuffle=False   # keep order consistent with label indexing
    )

    # --- Step 2: DINO feature extraction -------------------------------------

    print("\n[2/5] Extracting DINO ViT-S/8 features...")

    dino = DinoFeatureExtractor(device=DEVICE)

    train_features, train_labels = dino.get_or_create_embeddings(
        dataloader=train_loader,
        data_manager=stl10_manager,
        save_name='stl10_dino_train_embeddings'
    )

    print(f"Features shape: {train_features.shape}")   # (5000, 384)
    print(f"Labels shape:   {train_labels.shape}")     # (5000,)

    # --- Step 3: UMAP dimensionality reduction --------------------------------

    print(f"\n[3/5] Reducing {train_features.shape[1]}d -> {UMAP_DIMS}d with UMAP...")

    reducer = DimensionReducer(
        n_components=UMAP_DIMS,
        n_neighbors=UMAP_NEIGHBOURS,
        min_dist=0.1,
        model_path='./models/stl10_umap.pkl'
    )

    reduced_features = reducer.fit_transform(train_features)   # (N, 50)
    print(f"Reduced shape: {reduced_features.shape}")

    # --- Step 4: HDBSCAN pseudo-labels + noise reassignment ------------------

    print("\n[4/5] Generating pseudo-labels with HDBSCAN + noise reassignment...")

    plg = PseudoLabelGenerator(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES
    )

    pseudo_labels = plg.fit_predict(reduced_features)   # (N,) no -1 values

    pseudo_labels_tensor = torch.tensor(pseudo_labels, dtype=torch.long)
    torch.save(pseudo_labels_tensor, './results/pseudo_labels.pt')
    print(f"Pseudo-labels saved. Unique clusters: {len(np.unique(pseudo_labels))}")

    # --- Step 5: Mine neighbours + train projection head ---------------------

    print(f"\n[5/5] Mining {KNN_K} nearest neighbours per image...")

    neighbour_idx = mine_neighbours(train_features.numpy(), k=KNN_K)

    print("\nTraining MLP projection head...")

    projection_head = train_projection_head(
        embeddings=train_features,
        pseudo_labels=pseudo_labels_tensor,
        neighbour_idx=neighbour_idx,
        epochs=EPOCHS,
        batch_size=256,
        lr=LR,
        temperature=TEMPERATURE,
        device=DEVICE
    )

    stl10_manager.store_model(model=projection_head.state_dict(), name="stl10_projection_head")

    # --- Refit UMAP on projection head outputs -------------------------------
    # CRITICAL: the inference UMAP must be fitted on projection head outputs
    # (128d), not raw DINO features (384d). If we use the training UMAP here
    # we'd be passing 128d vectors into a reducer fitted on 384d — garbage out.

    print("\n[Post-train] Fitting inference UMAP on projection head outputs...")
    import umap as umap_lib

    projection_head.eval()
    with torch.no_grad():
        proj_train_feats = projection_head(train_features).numpy()   # (N, 128)

    inference_reducer = umap_lib.UMAP(
        n_components=50,
        n_neighbors=UMAP_NEIGHBOURS,
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    inference_reducer.fit(proj_train_feats)

    joblib.dump(inference_reducer, './models/stl10_inference_umap.pkl')
    print("Inference UMAP saved to ./models/stl10_inference_umap.pkl")

    # --- Training evaluation -------------------------------------------------
    # Two conditions compared side by side on training data:
    #
    # Baseline : raw DINO -> existing UMAP -> fresh HDBSCAN
    # Proposed : DINO -> projection head -> fresh UMAP -> fresh HDBSCAN
    #
    # Both use fresh HDBSCAN so the comparison is fair.
    # Labels are compared against STL-10 ground truth.

    from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
    from sklearn.metrics.pairwise import euclidean_distances
    import umap as umap_lib

    print("\n[Eval] Evaluating on training data...")

    true_labels_np = train_labels.numpy()

    # -- Baseline: DINO + existing UMAP + fresh HDBSCAN ----------------------
    print("\n  Running baseline (DINO + UMAP + HDBSCAN)...")
    baseline_clusterer = SklearnHDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES, metric="euclidean"
    )
    baseline_labels = baseline_clusterer.fit_predict(reduced_features)
    baseline_labels_clean = baseline_labels.copy()
    noise_mask = baseline_labels == -1
    if noise_mask.any() and hasattr(baseline_clusterer, "centroids_"):
        dists = euclidean_distances(reduced_features[noise_mask], baseline_clusterer.centroids_)
        baseline_labels_clean[noise_mask] = np.argmin(dists, axis=1)
    baseline_metrics = evaluate_clustering(
        features=reduced_features,
        predicted_labels=baseline_labels_clean,
        true_labels=true_labels_np
    )
    baseline_k = len(set(baseline_labels_clean))

    # -- Proposed: projection head -> fresh UMAP -> fresh HDBSCAN ------------
    print("\n  Running proposed (DINO -> projection head -> UMAP -> HDBSCAN)...")
    projection_head.eval()
    with torch.no_grad():
        proj_features = projection_head(train_features).numpy()   # (N, 128)

    proj_reducer = umap_lib.UMAP(
        n_components=50, n_neighbors=UMAP_NEIGHBOURS,
        min_dist=0.1, metric="cosine", random_state=42
    )
    reduced_proj = proj_reducer.fit_transform(proj_features)

    proposed_clusterer = SklearnHDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES, metric="euclidean"
    )
    proposed_labels = proposed_clusterer.fit_predict(reduced_proj)
    proposed_labels_clean = proposed_labels.copy()
    noise_mask = proposed_labels == -1
    if noise_mask.any() and hasattr(proposed_clusterer, "centroids_"):
        dists = euclidean_distances(reduced_proj[noise_mask], proposed_clusterer.centroids_)
        proposed_labels_clean[noise_mask] = np.argmin(dists, axis=1)
    proposed_metrics = evaluate_clustering(
        features=reduced_proj,
        predicted_labels=proposed_labels_clean,
        true_labels=true_labels_np
    )
    proposed_k = len(set(proposed_labels_clean))

    # -- Print comparison -----------------------------------------------------
    print("\n" + "=" * 55)
    print("TRAINING EVALUATION (ground truth: 10 clusters)")
    print("=" * 55)
    print(f"{'Metric':<20} {'Baseline':>10} {'Proposed':>10}")
    print("-" * 42)
    print(f"{'Silhouette':<20} {baseline_metrics.get('silhouette', float('nan')):>10.4f} {proposed_metrics.get('silhouette', float('nan')):>10.4f}")
    print(f"{'NMI':<20} {baseline_metrics.get('nmi', float('nan')):>10.4f} {proposed_metrics.get('nmi', float('nan')):>10.4f}")
    print(f"{'ARI':<20} {baseline_metrics.get('ari', float('nan')):>10.4f} {proposed_metrics.get('ari', float('nan')):>10.4f}")
    print(f"{'Clusters found':<20} {baseline_k:>10} {proposed_k:>10}")
    print("=" * 55)
    print("Run test_pipeline.py for held-out test set evaluation.")


if __name__ == '__main__':
    main()