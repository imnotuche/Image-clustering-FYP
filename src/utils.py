import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch

def evaluate_clustering(features, predicted_labels, true_labels=None, sample_size=5000):
    """
    Calculates key metrics to evaluate how well the images were grouped.
    """
    results = {}
    
    # 1. Silhouette Score: Measures how similar an image is to its own 
    # cluster compared to other clusters. Range: [-1, 1]
    # (Higher is better - indicates clear separation)
    results['silhouette'] = silhouette_score(features, predicted_labels, sample_size=sample_size)
    
    # 2. External Metrics: If we have the ground truth (CIFAR-10 labels)
    if true_labels is not None:
        # ARI: Measures the similarity between two assignments
        results['ari'] = adjusted_rand_score(true_labels, predicted_labels)
        # NMI: Measures the agreement between the labels
        results['nmi'] = normalized_mutual_info_score(true_labels, predicted_labels)
        
    return results

def plot_clusters(features, labels, title):
    print("Running t-SNE (this might take a minute)...")
    # Reduce 50-dim to 2-dim for plotting
    tsne = TSNE(n_components=2, perplexity=30, max_iter=300)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(title)
    plt.show()
    
def plot_cluster_gallery(images, predictions, cluster_id, num_samples=10):
    """
    Shows a row of images that the model assigned to a specific cluster.
    """
    # Find the indices of images belonging to the chosen cluster
    idxs = np.where(predictions == cluster_id)[0]
    
    if len(idxs) == 0:
        print(f"No images found in Cluster {cluster_id}")
        return

    # Pick a random subset if there are too many
    if len(idxs) > num_samples:
        idxs = np.random.choice(idxs, num_samples, replace=False)

    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(idxs):
        # Get the image and convert from (C, H, W) to (H, W, C) for plotting
        img = images[idx].permute(1, 2, 0).numpy()
        
        # Denormalize for display (mapping back to 0-1 range)
        img = (img - img.min()) / (img.max() - img.min())
        
        plt.subplot(1, len(idxs), i + 1)
        plt.imshow(img)
        plt.axis('off')
        if i == 0:
            plt.ylabel(f"Cluster {cluster_id}", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()