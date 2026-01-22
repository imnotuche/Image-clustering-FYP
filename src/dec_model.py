import torch.nn as nn
from clustering_layer import ClusteringLayer

class DEC(nn.Module):
    def __init__(self, n_clusters=10, embedding_dim=50):
        super().__init__()
        self.clustering_layer = ClusteringLayer(n_clusters=n_clusters, embedding_dim=embedding_dim)

    def forward(self, x):
        q = self.clustering_layer(x)
        return q