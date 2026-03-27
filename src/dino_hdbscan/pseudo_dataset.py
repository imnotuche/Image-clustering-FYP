import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """
    Dataset of (dino_embedding, pseudo_label) pairs.

    We train on pre-extracted DINO embeddings rather than raw images.
    This avoids running the DINO forward pass on every training batch,
    which would be slow with a frozen backbone. Embeddings are extracted
    once by DinoFeatureExtractor and reused across all training epochs.

    Args:
        embeddings:    torch.Tensor of shape (N, 384)
        pseudo_labels: torch.Tensor of shape (N,) — all non-negative integers
    """

    def __init__(self, embeddings: torch.Tensor, pseudo_labels: torch.Tensor):
        assert len(embeddings) == len(pseudo_labels), (
            f"Embeddings ({len(embeddings)}) and labels ({len(pseudo_labels)}) "
            "must have the same length."
        )
        assert (pseudo_labels >= 0).all(), (
            "pseudo_labels must not contain -1. Run noise reassignment first."
        )

        self.embeddings    = embeddings.float()
        self.pseudo_labels = pseudo_labels.long()

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int):
        return self.embeddings[idx], self.pseudo_labels[idx]
