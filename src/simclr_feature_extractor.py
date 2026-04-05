import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os

class SimCLRFeatureExtractor(nn.Module):
    """
    Feature extractor using a SimCLR-trained ResNet50 backbone.
    Drop-in replacement for the frozen ImageNet FeatureExtractor.
    
    The backbone weights come from SimCLR self-supervised training
    on STL-10, producing domain-specific features that are more
    semantically meaningful than frozen ImageNet features.
    
    Usage:
        extractor = SimCLRFeatureExtractor(checkpoint_path="./models/simclr_backbone_final.pt")
        features, labels = extractor.get_or_create_embeddings(dataloader, save_path)
    """

    def __init__(self, checkpoint_path=None):
        super().__init__()

        # Build ResNet50 backbone with classification head removed
        resnet = models.resnet50(weights=None)  # no pretrained weights, we load our own
        self.feature_layer = nn.Sequential(*list(resnet.children())[:-1])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if checkpoint_path is not None:
            self._load_simclr_weights(checkpoint_path)
        else:
            print("WARNING: No checkpoint path provided. Using random weights.")
            print("Pass checkpoint_path='./models/simclr_backbone_final.pt'")

        self.to(self.device)

    def _load_simclr_weights(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"SimCLR checkpoint not found at {checkpoint_path}. "
                f"Run the Colab training notebook first and download simclr_backbone_final.pt"
            )

        print(f"Loading SimCLR backbone weights from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.feature_layer.load_state_dict(state_dict)
        print("SimCLR weights loaded successfully.")

    def forward(self, x):
        with torch.no_grad():
            features = self.feature_layer(x)
            features = features.view(features.size(0), -1)   # (batch, 2048)
            features = F.normalize(features, p=2, dim=1)     # L2 normalise
        return features

    def get_or_create_embeddings(self, dataloader, save_path):
        """
        Check for local file:
        If found, load it.
        If not found, extract from SimCLR backbone and save it.
        """
        if os.path.exists(save_path):
            print(f"--- Found local embeddings at {save_path}. Loading... ---")
            data = torch.load(save_path, map_location=self.device, weights_only=False)
            return data['features'], data['labels']

        print(f"--- No local file found. Extracting features to {save_path}... ---")
        features_all = []
        labels_all = []

        self.eval()
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                feats = self.forward(images)
                features_all.append(feats.cpu())
                labels_all.append(labels if isinstance(labels, torch.Tensor) else torch.tensor(labels))

        features_cat = torch.cat(features_all)
        labels_cat = torch.cat(labels_all)

        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        embeddings = {'features': features_cat, 'labels': labels_cat}
        torch.save(embeddings, save_path)
        print(f"--- Extraction complete and saved to {save_path} ---")

        return features_cat, labels_cat
    

