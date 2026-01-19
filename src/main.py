import torch
from data_loader import get_dataloader
from train import train_dec
from utils import evaluate_clustering

def main():
    # 1. Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting experiment on: {device}")
    
    # 2. Load the Data (CIFAR-10)
    # We only need the training loader for this unsupervised task
    train_loader = get_dataloader(batch_size=64)
    
    # 3. Train the Model
    # This runs the Feature Extraction + DEC Clustering loop
    print("Training DEC model...")
    trained_model = train_dec(train_loader, n_clusters=10, epochs=10)
    
    # 4. Final Evaluation
    print("Evaluating clusters...")
    trained_model.eval()
    all_features = []
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in train_loader:
            # Extract features and get cluster predictions
            features = trained_model.feature_extractor(images)
            q = trained_model.clustering_layer(features)
            
            # The cluster with the highest probability is our prediction
            preds = torch.argmax(q, dim=1)
            
            all_features.append(features)
            all_predictions.append(preds)
            all_labels.append(labels)

    # Convert lists to large arrays for the utility functions
    features_cat = torch.cat(all_features).cpu().numpy()
    preds_cat = torch.cat(all_predictions).cpu().numpy()
    labels_cat = torch.cat(all_labels).cpu().numpy()

    # 5. Calculate Metrics
    metrics = evaluate_clustering(features_cat, preds_cat, labels_cat)
    
    print("\n--- Experiment Results ---")
    print(f"Silhouette Score: {metrics['silhouette']:.4f}")
    print(f"NMI: {metrics['nmi']:.4f}")
    print(f"ARI: {metrics['ari']:.4f}")

if __name__ == "__main__":
    main()