import torch.nn as nn
from feature_extractor import FeatureExtractor
from clustering_layer import ClusteringLayer

class DEC(nn.Module):
    def __init__(self, n_clusters=10):
        super().__init__()
        # Use the logic from your other files
        self.feature_extractor = FeatureExtractor()
        self.clustering_layer = ClusteringLayer(n_clusters=n_clusters)

    def forward(self, x):
        features = self.feature_extractor(x)
        q = self.clustering_layer(features)
        return q