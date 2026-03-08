import torch
import numpy as np
from data_manager import DataManager
from feature_extractor import FeatureExtractor
from inference import run_inference
from utils import plot_clusters, plot_cluster_gallery
from torchvision import datasets

def test_cluster():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test set
    stl10_manager = DataManager(path="./data/stl10", batch_size=128, device=device)
    test_loader = stl10_manager.get_loader(
        source=datasets.STL10,
        train=False,
        shuffle=True
    )
    
    feature_extractor = FeatureExtractor()
    model=stl10_manager.load__model("stl10_hdbscan_model", device=device)

    # Run inference on 1000 unseen images
    predictions, raw_images, strengths = run_inference(
        dataloader=test_loader,
        feature_extractor=feature_extractor,
        device=device,
        hdbscan_model=model,
        umap_model_path="./models/umap_model.pkl",
        limit=1000
    )

    print(f"Clustered {len(predictions)} unseen images.")
    print(f"Unique clusters found: {len(set(predictions)) - (1 if -1 in predictions else 0)}")
    print(f"Noise images: {np.sum(predictions == -1)}")

    # Show gallery for each cluster found (skip noise -1)
    unique_clusters = sorted(set(predictions) - {-1})
    for c_id in unique_clusters:
        # For gallery: use strength as confidence proxy
        # strengths is a 1D array of how strongly each point belongs to its cluster
        plot_cluster_gallery(raw_images, predictions, strengths, cluster_id=c_id)

    # 2D visualisation of the 1000 points
    # Flatten images just for t-SNE plot
    plot_clusters(
        raw_images.view(len(raw_images), -1).numpy(),
        predictions,
        title="t-SNE of 1000 Unseen Images (HDBSCAN)"
    )

if __name__ == "__main__":
    test_cluster()