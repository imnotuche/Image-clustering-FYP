import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.neighbors import NearestNeighbors

from dino_hdbscan.embedding_model import ProjectionHead
from dino_hdbscan.pseudo_dataset import EmbeddingDataset


def mine_neighbours(embeddings: np.ndarray, k: int = 10) -> np.ndarray:
    """
    For each embedding, find its k nearest neighbours by Euclidean distance.
    Returns an (N, k) array of neighbour indices.
    """
    print(f"Mining {k} nearest neighbours for {len(embeddings)} embeddings...")
    nn_model = NearestNeighbors(n_neighbors=k + 1, metric='euclidean', n_jobs=1)
    nn_model.fit(embeddings)
    _, indices = nn_model.kneighbors(embeddings)
    return indices[:, 1:]   # (N, k) — drop self


def supervised_contrastive_loss(projections: torch.Tensor,
                                labels: torch.Tensor,
                                temperature: float = 0.1) -> torch.Tensor:
    """
    Supervised contrastive loss (Khosla et al., 2020).

    Uses pseudo-labels to define positives: all images in the batch sharing
    the same pseudo-label are pulled together, all others pushed apart.

    This is more numerically stable than the neighbour mask approach because:
    - Every sample in a typical batch has at least one positive (same cluster)
    - No risk of all-zero positive masks causing NaN in log
    - Well-tested formulation with known stable implementation

    Args:
        projections: (B, D) L2-normalised embeddings
        labels:      (B,)   pseudo-labels
        temperature: sharpness scaling

    Returns:
        scalar loss
    """
    B = projections.size(0)
    device = projections.device

    # Pairwise cosine similarity (L2-normalised, so dot product = cosine)
    sim = torch.matmul(projections, projections.T) / temperature   # (B, B)

    # For numerical stability: subtract row max before exp
    sim_max, _ = torch.max(sim, dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    # Positive mask: same label, different sample
    labels_col = labels.unsqueeze(1)   # (B, 1)
    labels_row = labels.unsqueeze(0)   # (1, B)
    positive_mask = (labels_col == labels_row).float()   # (B, B)

    # Remove self from both positive mask and denominator
    self_mask = torch.eye(B, dtype=torch.bool, device=device)
    positive_mask = positive_mask.masked_fill(self_mask, 0.0)

    # Denominator: sum over all non-self
    exp_sim = torch.exp(sim)
    exp_sim_no_self = exp_sim.masked_fill(self_mask, 0.0)
    denom = exp_sim_no_self.sum(dim=1, keepdim=True)   # (B, 1)

    # Log probability of each pair
    log_prob = sim - torch.log(denom + 1e-8)   # (B, B)

    # Only average over positive pairs; skip samples with no positives in batch
    n_positives = positive_mask.sum(dim=1)   # (B,)
    has_positives = n_positives > 0

    if not has_positives.any():
        return torch.tensor(0.0, requires_grad=True, device=device)

    loss_per_sample = -(positive_mask * log_prob).sum(dim=1)
    loss_per_sample = loss_per_sample[has_positives] / n_positives[has_positives]

    return loss_per_sample.mean()


def train_projection_head(
    embeddings:    torch.Tensor,
    pseudo_labels: torch.Tensor,
    neighbour_idx: np.ndarray,   # kept for API compatibility, not used in loss
    epochs:        int   = 50,
    batch_size:    int   = 256,
    lr:            float = 1e-4,
    temperature:   float = 0.1,
    device:        str   = 'cpu'
) -> ProjectionHead:
    """
    Trains the MLP projection head using supervised contrastive loss
    on HDBSCAN pseudo-labels.

    Uses pseudo-labels (not neighbour indices) to define positives.
    All images in a batch sharing the same pseudo-label are pulled together.
    This is numerically stable and directly optimises cluster separation.
    """
    device = torch.device(device)

    dataset   = EmbeddingDataset(embeddings, pseudo_labels)
    loader    = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, drop_last=True)   # drop_last avoids tiny batches

    model     = ProjectionHead(input_dim=384, hidden_dim=256, output_dim=128)
    model     = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs, eta_min=1e-6)

    best_loss  = float('inf')
    best_state = None

    print(f"\nTraining projection head — {epochs} epochs, "
        f"batch={batch_size}, lr={lr}, T={temperature}")
    print("-" * 55)

    for epoch in range(epochs):
        model.train()
        epoch_loss  = 0.0
        batch_count = 0

        for batch_emb, batch_labels in loader:
            batch_emb    = batch_emb.to(device)      # (B, 384)
            batch_labels = batch_labels.to(device)   # (B,)

            projections = model(batch_emb)           # (B, 128)

            loss = supervised_contrastive_loss(projections, batch_labels, temperature)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent any remaining instability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss  += loss.item()
            batch_count += 1

        scheduler.step()
        avg_loss = epoch_loss / max(batch_count, 1)
        lr_now   = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1:3d}/{epochs}]  Loss: {avg_loss:.4f}  LR: {lr_now:.2e}")

        if avg_loss < best_loss:
            best_loss  = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  -> New best (loss {best_loss:.4f})")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.cpu()
    model.eval()
    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    return model