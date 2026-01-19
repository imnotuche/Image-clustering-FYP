import torch
import torch.nn as nn

class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters=10, embedding_dim=2048, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        # The 'Centers' that the model will learn
        self.centers = nn.Parameter(torch.Tensor(n_clusters, embedding_dim))
        nn.init.xavier_normal_(self.centers)

    def forward(self, x):
        # Math to calculate probability of belonging to a cluster
        dist = torch.sum((x.unsqueeze(1) - self.centers)**2, 2)
        q = 1.0 / (1.0 + dist / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q
