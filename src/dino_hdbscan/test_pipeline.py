"""
test_pipeline.py — Evaluation on STL-10 test set

Loads trained projection head and UMAP model, runs inference on the
STL-10 test set using pre-extracted DINO embeddings (no redundant
DINO forward pass), and reports NMI, ARI, Silhouette Score.

Run AFTER run_pipeline.py:
    python test_pipeline.py
"""

import os
import torch
import numpy as np
from torchvision import datasets

from dino_hdbscan.data_manager import DataManager
from dino_hdbscan.feature_extractor import DinoFeatureExtractor
from dino_hdbscan.dimension_reduction import DimensionReducer
from dino_hdbscan.embedding_model import ProjectionHead
from dino_hdbscan.inference import run_inference
from dino_hdbscan.evaluate import evaluate_clustering, plot_clusters, plot_cluster_gallery
from dino_hdbscan.auto_min_cluster import auto_min_cluster_size


def main():

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 64
    MIN_SAMPLES = 5
    
    stl10_manager = DataManager(
        path='./data/stl10',
        batch_size=BATCH_SIZE,
        device=DEVICE
    )
    
    UMAP_MODEL_PATH = './models/stl10_inference_umap.pkl'
    HEAD_WEIGHTS = stl10_manager.load_model(name="stl10_projection_head", device=DEVICE, path=True)

    os.makedirs('./results', exist_ok=True)

    print(f"Device: {DEVICE}")
    print("=" * 55)

    for path in [UMAP_MODEL_PATH, HEAD_WEIGHTS]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing trained model: {path}\n"
                "Run run_pipeline.py first."
            )

    # --- Load trained models -------------------------------------------------

    print("\n[1/4] Loading trained models...")

    dino = DinoFeatureExtractor(device=DEVICE)

    reducer = DimensionReducer(model_path=UMAP_MODEL_PATH)

    projection_head = ProjectionHead(input_dim=384, hidden_dim=256, output_dim=128)
    projection_head.load_state_dict(
        torch.load(HEAD_WEIGHTS, map_location='cpu', weights_only=True)
    )
    projection_head.eval()
    print("All models loaded.")

    # --- Load STL-10 test set + extract embeddings ---------------------------

    print("\n[2/4] Loading STL-10 test set...")

    test_loader = stl10_manager.get_loader(
        source=datasets.STL10,
        train=False,
        shuffle=False
    )

    print("\n[3/4] Extracting DINO features for test set...")

    test_features, test_labels = dino.get_or_create_embeddings(
        dataloader=test_loader,
        data_manager=stl10_manager,
        save_name='stl10_dino_test_embeddings'
    )

    print(f"Test features: {test_features.shape}")
    print(f"Test labels:   {test_labels.shape}")

    # --- Run inference using pre-extracted embeddings ------------------------
    # Pass dino_features directly — no redundant DINO forward pass

    print("\n[4/4] Running inference...")

    INFERENCE_LIMIT = 1000
    MIN_CLUSTER_SIZE = auto_min_cluster_size(n=INFERENCE_LIMIT)

    # Slice labels and features to match the limit upfront
    if INFERENCE_LIMIT is not None:
        test_features_inf = test_features[:INFERENCE_LIMIT]
        test_labels_inf   = test_labels[:INFERENCE_LIMIT]
    else:
        test_features_inf = test_features
        test_labels_inf   = test_labels

    predictions, _, strengths = run_inference(
        projection_head=projection_head,
        dimension_reducer=reducer,
        dino_features=test_features_inf,
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        device=DEVICE
    )

    # --- Metrics -------------------------------------------------------------

    # Project features through head for silhouette score geometry
    projection_head_gpu = projection_head.to(DEVICE)
    with torch.no_grad():
        proj_feats = projection_head_gpu(test_features_inf.to(DEVICE)).cpu().numpy()

    reduced_test = reducer.transform(proj_feats)

    metrics = evaluate_clustering(
        features=reduced_test,
        predicted_labels=predictions,
        true_labels=test_labels_inf.numpy()
    )

    n_clusters = len(set(predictions)) - (1 if -1 in predictions else 0)
    n_noise    = int(np.sum(predictions == -1))

    print("\n" + "=" * 55)
    print("RESULTS — STL-10 Test Set")
    print("=" * 55)
    print(f"Silhouette Score : {metrics.get('silhouette', float('nan')):.4f}")
    print(f"NMI              : {metrics.get('nmi', float('nan')):.4f}")
    print(f"ARI              : {metrics.get('ari', float('nan')):.4f}")
    print(f"Clusters found   : {n_clusters}  (ground truth: 10)")
    print(f"Noise images     : {n_noise} ({n_noise / len(predictions) * 100:.1f}%)")
    print("=" * 55)

    torch.save({
        'predictions': predictions,
        'true_labels': test_labels.numpy(),
        'metrics':     metrics,
        'n_clusters':  n_clusters,
        'n_noise':     n_noise
    }, './results/test_results.pt')
    print("\nResults saved to ./results/test_results.pt")

    # --- Visualisations ------------------------------------------------------

    plot_clusters(reduced_test, predictions, title="Test Set — Predicted Clusters")

    unique_clusters = sorted(set(predictions) - {-1})
    print(f"\nGenerating galleries for {len(unique_clusters)} clusters...")

    # Collect raw images up to INFERENCE_LIMIT for gallery display
    all_images  = []
    total_so_far = 0
    for imgs, _ in test_loader:
        remaining = (INFERENCE_LIMIT - total_so_far) if INFERENCE_LIMIT is not None else imgs.size(0)
        all_images.append(imgs[:remaining])
        total_so_far += imgs[:remaining].size(0)
        if INFERENCE_LIMIT is not None and total_so_far >= INFERENCE_LIMIT:
            break
    raw_images = torch.cat(all_images, dim=0)

    for c_id in unique_clusters:
        plot_cluster_gallery(raw_images, predictions, strengths, cluster_id=c_id)


if __name__ == '__main__':
    main()