import torch
import numpy as np
from data_manager import DataManager
from train import train_dec
from utils import evaluate_clustering
from torchvision import datasets

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting experiment on: {device}")
    
    manager=DataManager(path="./data/cifar10", batch_size=128, device=device)
    
    train_loader=manager.get_loader(
        source=datasets.CIFAR10,
        train=True,
        shuffle=False
    )
    
    # 2. Train the Model
    # Note: Ensure train_dec returns the model AND the processed features/labels
    print("Training DEC model...")
    trained_model = train_dec(train_loader, n_clusters=10, embedding_dim=50, epochs=10)
    
    # 3. Load the Reduced Features for Evaluation
    # Since train_dec saved them to disk, let's grab them
    print("Loading reduced features for evaluation...")
    # Using weights_only=False because of the NumPy data inside the .pt file
    data = torch.load("./embeddings/cifar10_reduced.pt", weights_only=False)
    
    # If you saved it as a dict, unpack it. If just a tensor, use it directly.
    if isinstance(data, dict):
        features_cat = data['features']
        labels_cat = data['labels']
    else:
        features_cat = data
        # You'll need to get labels from your extractor or original loader
        # Let's assume you returned labels from train_dec or saved them
        labels_cat = torch.load("./embeddings/cifar10_embeddings.pt", weights_only=False)['labels']

    # 4. Final Evaluation
    print("Evaluating clusters...")
    trained_model.eval()
    
    if isinstance(features_cat, np.ndarray):
        features_cat = torch.from_numpy(features_cat).float()
    
    with torch.no_grad():
        # Move features to device and get 'q' (cluster probabilities)
        q = trained_model(features_cat.to(device))
        
        # Get the cluster with the highest probability
        preds_cat = torch.argmax(q, dim=1).cpu().numpy()

    # 5. Calculate Metrics
    # Convert tensors to numpy for sklearn-based metrics in utils
    metrics = evaluate_clustering(
        features_cat.cpu().numpy(), 
        preds_cat, 
        labels_cat.cpu().numpy(),
        sample_size=5000
    )
    
    print("\n--- Experiment Results ---")
    print(f"Silhouette Score: {metrics['silhouette']:.4f}")
    print(f"NMI: {metrics['nmi']:.4f}")
    print(f"ARI: {metrics['ari']:.4f}")
    
    # Save the model weights
    model_path = "./models/dec_model_final.pth"
    torch.save(trained_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Create a results dictionary
    results = {
        'predictions': preds_cat,
        'labels': labels_cat,
        'metrics': metrics
    }

    # Save using torch.save or pickle
    results_path = "./results/experiment_results.pt"
    torch.save(results, results_path)
    print(f"Experiment results saved to {results_path}")

if __name__ == "__main__":
    main()