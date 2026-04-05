import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    Trainable MLP projection head attached on top of the frozen DINO backbone.

    Architecture:
        DINO output (384d)
        -> Linear(384, 256) -> BatchNorm -> ReLU
        -> Linear(256, 128)
        -> L2 normalise

    Only the projection head weights are updated during training.
    The DINO backbone is never touched.

    At inference time the projection head transforms DINO embeddings into
    a refined 128-dimensional space optimised for clustering. The DINO
    backbone is run separately (in feature_extractor.py) so we never
    recompute DINO features unnecessarily.
    """

    def __init__(self, input_dim: int = 384, hidden_dim: int = 256,
                output_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) — L2-normalised DINO embeddings

        Returns:
            (B, output_dim) — L2-normalised projected embeddings
        """
        x = self.net(x)
        x = F.normalize(x, p=2, dim=1)
        return x
