import torch
import numpy as np
from data_manager import DataManager
from feature_extractor import FeatureExtractor
from train import train_hdbscan
from utils import evaluate_clustering, plot_clusters
from dimension_reduction import get_reduced_features
from torchvision import datasets

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting experiment on: {device}")

    # --- SETUP ---
    stl10_manager = DataManager(path="./data/stl10", batch_size=128, device=device)

    # Change the save_path to wherever your embeddings are stored
    raw_features, labels = stl10_manager.load__embedding("stl10_embeddings")

    print(f"Raw features shape: {raw_features.shape}")  # Should be (50000, 2048)

    # --- STEP 2: UMAP + HDBSCAN ---
    predicted_labels, reduced_features, clusterer = train_hdbscan(
        raw_features=raw_features,
        min_cluster_size=50,
        min_samples=5,
        embedding_dim=2,
        umap_model_path="./models/umap_model.pkl",
    )

    stl10_manager.store_model(model=clusterer, name="stl10_hdbscan_model")
    
    # --- STEP 3: Evaluate ---
    # Only evaluate on non-noise points for silhouette
    # (silhouette doesn't make sense for noise points)
    mask = predicted_labels != -1
    
    if mask.sum() < 2:
        print("Too many noise points to evaluate. Try lowering min_cluster_size.")
        return

    print("Evaluating clusters...")
    metrics = evaluate_clustering(
        features=reduced_features[mask],
        predicted_labels=predicted_labels[mask],
        true_labels=labels.numpy()[mask],
        sample_size=min(5000, mask.sum())
    )

    print("\n--- Experiment Results ---")
    print(f"Silhouette Score: {metrics['silhouette']:.4f}")
    print(f"NMI:              {metrics['nmi']:.4f}")
    print(f"ARI:              {metrics['ari']:.4f}")

    n_clusters = len(set(predicted_labels)) - (1 if -1 in predicted_labels else 0)
    n_noise = np.sum(predicted_labels == -1)
    print(f"Clusters found:   {n_clusters}")
    print(f"Noise images:     {n_noise} ({n_noise/len(predicted_labels)*100:.1f}%)")
    

    # --- STEP 4: Visualise (optional) ---
    # Reduce to 2D just for the plot
    features_2d = get_reduced_features(
        raw_features=raw_features[mask] if hasattr(raw_features, '__getitem__') else raw_features,
        n_dims=2,
        umap_model_path="./models/umap_2d_model.pkl"
    )
    plot_clusters(features_2d, predicted_labels[mask], title="HDBSCAN Clusters (UMAP 2D)")

    # --- STEP 5: Save results ---
    results = {
        'predictions': predicted_labels,
        'labels': labels.numpy(),
        'metrics': metrics
    }
    torch.save(results, "./results/hdbscan_experiment_results.pt")
    print("Results saved to ./results/hdbscan_experiment_results.pt")

if __name__ == "__main__":
    main()
