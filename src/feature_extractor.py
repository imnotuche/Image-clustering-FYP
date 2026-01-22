import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import os

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained ResNet-50 
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.feature_layer = nn.Sequential(*list(resnet.children())[:-1])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        with torch.no_grad():
            features = self.feature_layer(x)
        return features.view(features.size(0), -1)

    def get_or_create_embeddings(self, dataloader, save_path):
        """
        Check for local file: 
        If found, load it.
        If not found, Extract from ResNet and save it.
        """
        if os.path.exists(save_path):
            print(f"--- Found local embeddings at {save_path}. Loading... ---")
            data = torch.load(save_path, map_location=self.device)
            return data['features'], data['labels']
        
        print(f"--- No local file found. Extracting features to {save_path}... ---")
        features_all = []
        labels_all = []
        
        self.eval() # Set to evaluation mode
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                feats = self.forward(images)
                features_all.append(feats.cpu())
                labels_all.append(labels)
        
        features_cat = torch.cat(features_all)
        labels_cat = torch.cat(labels_all)
        
        # Save the dictionary locally for next time
        torch.save({'features': features_cat, 'labels': labels_cat}, save_path)
        print(f"--- Extraction complete and saved to {save_path} ---")
        
        return features_cat, labels_cat