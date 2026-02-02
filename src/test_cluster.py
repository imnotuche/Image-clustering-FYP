from data_manager import DataManager
from inference import run_inference
from utils import plot_clusters, plot_cluster_gallery 
import torch
from feature_extractor import FeatureExtractor
import numpy as np
from torchvision import datasets

def test_cluster():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the test set
    manager=DataManager(batch_size=128, device=device)

    test_loader=manager.get_loader(
        source=datasets.CIFAR10,
        path="./data/cifar10",
        train=False,
        shuffle=True
    )

    # Initialize Feature Extractor
    feature_extractor = FeatureExtractor()

    # 2. Execute Inference (Added the limit=1000 here)
    predictions, raw_images, probabilities = run_inference(
        test_loader, 
        feature_extractor, 
        device, 
        model_path="./models/dec_model_final.pth", 
        pca_model_path="./models/pca_model.pkl",
        limit=1000
    )

    # 3. Prove it worked!
    predictions = np.argmax(probabilities, axis=1)
    print(f"✅ Successfully clustered {len(predictions)} unseen images.")

    # VISUAL CHECK: 
    # IMPORTANT: Pass 'raw_images' (the tensors), NOT 'test_loader'
    for c_id in range(10):
        plot_cluster_gallery(raw_images, probabilities, cluster_id=c_id)

    # OPTIONAL: See the 2D map of these 1000 unseen points
    # We flatten the images just for the t-SNE visualization
    plot_clusters(raw_images.view(len(raw_images), -1).numpy(), predictions, title="t-SNE of 1000 Unseen Images")
    
if __name__ == "__main__":
    test_cluster()