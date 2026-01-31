import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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


def plot_cluster_gallery(images, q_probabilities, cluster_id, num_samples=10):
    """
    Shows the top-N images the model is MOST confident about for a specific cluster.
    """
    # 1. Get the confidence scores for the specific cluster column
    # q_probabilities shape is (N, 10)
    cluster_scores = q_probabilities[:, cluster_id]

    # 2. Get indices of the highest scores (sorted descending)
    # argsort goes small to large, so we take the last 'num_samples' and reverse them
    idxs = np.argsort(cluster_scores)[-num_samples:][::-1]
    
    # Check if we actually have images (should always be true if data exists)
    if len(idxs) == 0:
        print(f"No images found for Cluster {cluster_id}")
        return

    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(idxs):
        # Get the image and convert from (C, H, W) to (H, W, C)
        img = images[idx].permute(1, 2, 0).numpy()
        
        # Denormalize for display
        img = (img - img.min()) / (img.max() - img.min())
        
        # Get the confidence percentage for the title
        confidence = cluster_scores[idx] * 100
        
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img)
        plt.title(f"{confidence:.1f}%", fontsize=10) # Show how sure the model is
        plt.axis('off')
        if i == 0:
            plt.ylabel(f"Cluster {cluster_id}", fontsize=12, fontweight='bold')
    
    plt.suptitle(f"Top {num_samples} Most Confident Images for Cluster {cluster_id}", fontsize=14, y=1.05)
    plt.tight_layout()
    plt.show()