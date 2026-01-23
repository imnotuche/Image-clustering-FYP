import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.cluster import KMeans
from dec_model import DEC
from feature_extractor import FeatureExtractor
from dimension_reduction import get_reduced_features

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def train_dec(dataloader, n_clusters=10, embedding_dim=50, epochs=50):
    
    #raw embeddings and dimensionally reduced embeddings path
    raw_embeddings_path="./embeddings/cifar10_embeddings.pt"
    reduced_embeddings_path="./embeddings/cifar10_reduced.pt"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DEC(n_clusters=n_clusters, embedding_dim=embedding_dim).to(device)
    
    #extract embeddings and reduce dimensions
    extractor=FeatureExtractor()
    raw_features, labels = extractor.get_or_create_embeddings(dataloader=dataloader, save_path=raw_embeddings_path)
    reduced_features=get_reduced_features(raw_features=raw_features, save_path=reduced_embeddings_path, n_dims=embedding_dim)
    
    # Ensure reduced_features is a tensor
    if isinstance(reduced_features, np.ndarray):
        reduced_features = torch.from_numpy(reduced_features).float()
        
    all_features = reduced_features.numpy()
    
    # Find the 'densest clouds' to avoid the Zero Loss problem
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(all_features)
    
    # Set the model's cluster centers to the K-Means centers
    initial_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float).to(device)
    model.clustering_layer.centers.data = initial_centers
    
    # --- STAGE 2: DEC TRAINING ---
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.KLDivLoss(reduction='batchmean')

        
    feature_dataset = TensorDataset(reduced_features)
    feature_loader = DataLoader(feature_dataset, batch_size=128, shuffle=True)
    
    model.train()
    print("Starting Deep Clustering refinement...")
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch in feature_loader:
            batch_data = batch[0].to(device)
            
            optimizer.zero_grad()
            
            q = model(batch_data) 
            
            p = target_distribution(q).detach()
            loss = criterion(q.log(), p)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    return model