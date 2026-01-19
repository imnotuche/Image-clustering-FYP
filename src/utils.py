import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

def evaluate_clustering(features, predicted_labels, true_labels=None):
    """
    Calculates key metrics to evaluate how well the images were grouped.
    """
    results = {}
    
    # 1. Silhouette Score: Measures how similar an image is to its own 
    # cluster compared to other clusters. Range: [-1, 1]
    # (Higher is better - indicates clear separation)
    results['silhouette'] = silhouette_score(features, predicted_labels)
    
    # 2. External Metrics: If we have the ground truth (CIFAR-10 labels)
    if true_labels is not None:
        # ARI: Measures the similarity between two assignments
        results['ari'] = adjusted_rand_score(true_labels, predicted_labels)
        # NMI: Measures the agreement between the labels
        results['nmi'] = normalized_mutual_info_score(true_labels, predicted_labels)
        
    return results