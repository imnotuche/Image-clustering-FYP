import torch
import torch.nn.functional as F
from dino_hdbscan.data_manager import DataManager


class DinoFeatureExtractor:
    """
    Wraps DINO ViT-S/8 (frozen) for batch feature extraction.
    Outputs 384-dimensional L2-normalised embeddings per image.
    """

    def __init__(self, device):
        self.device = torch.device(device)

        print("Loading DINO ViT-S/8...")
        self.model = torch.hub.load(
            'facebookresearch/dino:main',
            'dino_vits8'
        )
        # Freeze all parameters — backbone never updates
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.to(self.device)
        self.model.eval()
        print("DINO loaded and frozen.")

    def extract_batch(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass on a single batch.
        Returns L2-normalised (batch, 384) tensor on CPU.
        """
        images = images.to(self.device)
        with torch.no_grad():
            features = self.model(images)          # (B, 384)
        features = F.normalize(features, p=2, dim=1)
        return features.cpu()

    def get_or_create_embeddings(self, dataloader, data_manager: DataManager, save_name: str):
        """
        If embeddings are already saved in DataManager registry, load and return them.
        Otherwise extract from the full dataloader, save, and return.

        Returns:
            features (torch.Tensor): shape (N, 384)
            labels   (torch.Tensor): shape (N,)
        """
        # Try loading from registry first
        try:
            features, labels = data_manager.load_embedding(save_name)
            print(f"Loaded existing embeddings '{save_name}' — shape {features.shape}")
            return features, labels
        except Exception:
            pass  # not saved yet, extract below

        print(f"No saved embeddings found for '{save_name}'. Extracting...")
        all_features = []
        all_labels = []

        for batch_idx, (images, labels_batch) in enumerate(dataloader):
            feats = self.extract_batch(images)
            all_features.append(feats)
            all_labels.append(labels_batch)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * dataloader.batch_size} images...")

        features = torch.cat(all_features, dim=0)   # (N, 384)
        labels   = torch.cat(all_labels,   dim=0)   # (N,)

        print(f"Extraction complete. Shape: {features.shape}")

        # Save via DataManager
        embeddings = {'features': features, 'labels': labels}
        data_manager.store_embedding(embeddings, name=save_name, overwrite=True)

        return features, labels
