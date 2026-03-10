import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                            normalized_mutual_info_score)
from sklearn.manifold import TSNE


def evaluate_clustering(
    features:         np.ndarray,
    predicted_labels: np.ndarray,
    true_labels:      np.ndarray = None,
    sample_size:      int        = 5000
) -> dict:
    """
    Computes clustering quality metrics.

    Silhouette Score is an internal metric — no ground truth needed.
    NMI and ARI are external metrics — require ground truth labels.

    Args:
        features:         (N, D) reduced embeddings
        predicted_labels: (N,)   cluster assignments
        true_labels:      (N,)   ground truth class labels (optional)
        sample_size:      max samples for silhouette (expensive on large N)

    Returns:
        dict with keys: 'silhouette', 'nmi' (if true_labels), 'ari' (if true_labels)
    """
    results = {}

    # Silhouette requires at least 2 clusters and no noise-only sets
    unique = np.unique(predicted_labels)
    valid  = unique[unique != -1]

    if len(valid) >= 2:
        # Only evaluate on non-noise points
        mask = predicted_labels != -1
        n    = int(mask.sum())
        s    = min(sample_size, n)
        results['silhouette'] = silhouette_score(
            features[mask], predicted_labels[mask], sample_size=s
        )
    else:
        results['silhouette'] = float('nan')
        print("Warning: fewer than 2 clusters — silhouette score undefined.")

    if true_labels is not None:
        # Use only non-noise points for external metrics too
        mask = predicted_labels != -1
        results['nmi'] = normalized_mutual_info_score(
            true_labels[mask], predicted_labels[mask]
        )
        results['ari'] = adjusted_rand_score(
            true_labels[mask], predicted_labels[mask]
        )

    return results


def plot_clusters(features: np.ndarray, labels: np.ndarray, title: str = "Clusters"):
    """
    2D t-SNE scatter plot coloured by cluster label.
    """
    print("Running t-SNE (this may take a few minutes)...")
    tsne       = TSNE(n_components=2, perplexity=30, max_iter=500, random_state=42)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        features_2d[:, 0], features_2d[:, 1],
        c=labels, cmap='tab20', alpha=0.6, s=5
    )
    plt.colorbar(scatter)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"./results/{title.replace(' ', '_')}.png", dpi=150)
    plt.show()
    print(f"Plot saved to ./results/")


def plot_cluster_gallery(
    images:      torch.Tensor,
    predictions: np.ndarray,
    strengths:   np.ndarray,
    cluster_id:  int,
    num_samples: int = 10
):
    """
    Shows the top-N most confident images for a given cluster.

    Args:
        images:      (N, C, H, W) raw image tensors
        predictions: (N,) cluster label per image
        strengths:   (N,) confidence score per image (0-1)
        cluster_id:  which cluster to visualise
        num_samples: how many images to show
    """
    mask    = predictions == cluster_id
    indices = np.where(mask)[0]

    if len(indices) == 0:
        print(f"No images found for Cluster {cluster_id}")
        return

    # Sort by strength descending
    sorted_order = np.argsort(strengths[indices])[::-1]
    top_indices  = indices[sorted_order[:num_samples]]

    fig, axes = plt.subplots(1, len(top_indices), figsize=(15, 3))
    if len(top_indices) == 1:
        axes = [axes]

    for i, idx in enumerate(top_indices):
        img = images[idx].permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        axes[i].imshow(img)
        axes[i].set_title(f"{strengths[idx]*100:.1f}%", fontsize=9)
        axes[i].axis('off')

    axes[0].set_ylabel(f"Cluster {cluster_id}", fontsize=11, fontweight='bold')
    fig.suptitle(
        f"Top {len(top_indices)} images — Cluster {cluster_id}",
        fontsize=13, y=1.02
    )
    plt.tight_layout()
    plt.savefig(f"./results/cluster_{cluster_id}_gallery.png",
                dpi=120, bbox_inches='tight')
    plt.show()
