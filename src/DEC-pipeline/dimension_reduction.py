from sklearn.decomposition import PCA
import joblib

def get_reduced_features(raw_features, n_dims=50, pca_model_path="./models/pca_model.pkl"):

    print("--- Reducing dimensions via PCA... ---")
    features_scaled = raw_features.cpu().numpy()
    
    # Run Randomized SVD (via PCA)
    pca = PCA(n_components=n_dims, random_state=42)
    reduced = pca.fit_transform(features_scaled)
    
    #save pca model
    joblib.dump(pca, pca_model_path) 
    print(f"PCA model saved to {pca_model_path}")
    
    return reduced