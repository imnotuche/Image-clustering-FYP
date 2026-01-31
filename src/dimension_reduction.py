import os
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

def get_reduced_features(raw_features, save_path, n_dims=50, pca_model_path="./models/pca_model.pkl"):
    
    #check if reduced dimensions for the dataset already exists
    if os.path.exists(save_path):
        print(f"--- Loading reduced features from {save_path} ---")
        return torch.load(save_path, weights_only=False)

    # 2. If not, we do the math
    print("--- Reducing dimensions via PCA... ---")
    features_scaled = raw_features.cpu().numpy()
    
    # Run Randomized SVD (via PCA)
    pca = PCA(n_components=n_dims, random_state=42)
    reduced = pca.fit_transform(features_scaled)
    
    #save pca model
    joblib.dump(pca, pca_model_path) 
    print(f"PCA model saved to {pca_model_path}")
    
    # 3. Save it so we never do this again
    torch.save(reduced, save_path)
    print(f"--- Saved reduced features to {save_path} ---")
    return reduced