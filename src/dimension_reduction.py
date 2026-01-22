import os
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def get_reduced_features(raw_features, save_path, n_dims=50):
    
    #check if reduced dimensions for the dataset already exists
    if os.path.exists(save_path):
        print(f"--- Loading reduced features from {save_path} ---")
        return torch.load(save_path, weights_only=False)

    # 2. If not, we do the math
    print("--- Reducing dimensions via PCA... ---")
    features = raw_features.numpy()
    
    # Scale first (Crucial for SVD/PCA)
    features_scaled = StandardScaler().fit_transform(features)
    
    # Run Randomized SVD (via PCA)
    pca = PCA(n_components=n_dims, random_state=42)
    reduced = pca.fit_transform(features_scaled)
    
    # 3. Save it so we never do this again
    torch.save(features, save_path)
    print(f"--- Saved reduced features to {save_path} ---")
    return features