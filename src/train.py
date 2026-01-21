import torch
import torch.optim as optim
import numpy as np
from sklearn.cluster import KMeans
from dec_model import DEC

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def train_dec(dataloader, n_clusters=10, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DEC(n_clusters=n_clusters).to(device)
    
    # --- STAGE 1: FEATURE CACHING & K-MEANS ---
    # We run the images through ResNet ONCE to get the 'clouds'
    print("Extracting features and initializing clusters with K-Means...")
    features_list = []
    with torch.no_grad():
        for images in dataloader:
            features = model.feature_extractor(images.to(device))
            features_list.append(features.cpu())
    
    all_features = torch.cat(features_list).numpy()
    
    # Find the 'densest clouds' to avoid the Zero Loss problem
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(all_features)
    
    # Set the model's cluster centers to the K-Means centers
    initial_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float).to(device)
    model.clustering_layer.centers.data = initial_centers
    
    # --- STAGE 2: DEC TRAINING ---
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.KLDivLoss(reduction='batchmean')

    model.train()
    print("Starting Deep Clustering refinement...")
    
    for epoch in range(epochs):
        total_loss = 0
        for images, _ in dataloader:
            images = images.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            q = model(images)
            
            # Sharpen the distribution
            p = target_distribution(q).detach()
            
            # Calculate how far we are from our 'sharpened' goal
            loss = criterion(q.log(), p)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    return model