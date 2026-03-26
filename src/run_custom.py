"""
run_custom.py — Cluster your own images

Point IMAGE_FOLDER at any folder of images and this script will:
1. Extract DINO features
2. Run through trained projection head
3. Cluster with HDBSCAN
4. Save gallery PNGs to ./results/galleries_custom/

Run:
    python run_custom.py
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from data_manager import DataManager
from feature_extractor import DinoFeatureExtractor
from dimension_reduction import DimensionReducer
from embedding_model import ProjectionHead
from inference import run_inference
from evaluate import evaluate_clustering, plot_clusters

# ─── Config ──────────────────────────────────────────────────────────────────

IMAGE_FOLDER     = './data/high-res-test/random-objects'   # <-- change this to your folder path
DEVICE           = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE       = 16

# Keep these small for small batches
MIN_CLUSTER_SIZE = 5
MIN_SAMPLES      = 2

UMAP_MODEL_PATH  = './models/stl10_inference_umap.pkl'
HEAD_WEIGHTS     = './models/stl10_projection_head.pth'

# ─── Sanity checks ───────────────────────────────────────────────────────────

for path in [UMAP_MODEL_PATH, HEAD_WEIGHTS]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing trained model: {path}\n"
            "Run run_pipeline.py first."
        )

if not os.path.isdir(IMAGE_FOLDER):
    raise NotADirectoryError(
        f"Image folder not found: {IMAGE_FOLDER}\n"
        "Set IMAGE_FOLDER to the path of your images."
    )

print(f"Device       : {DEVICE}")
print(f"Image folder : {IMAGE_FOLDER}")
print("=" * 55)

# ─── Load models ─────────────────────────────────────────────────────────────

print("\n[1/3] Loading trained models...")

dino    = DinoFeatureExtractor(device=DEVICE)
reducer = DimensionReducer(model_path=UMAP_MODEL_PATH)

projection_head = ProjectionHead(input_dim=384, hidden_dim=256, output_dim=128)
projection_head.load_state_dict(
    torch.load(HEAD_WEIGHTS, map_location='cpu', weights_only=True)
)
projection_head.eval()
print("Models loaded.")

# ─── Load images ─────────────────────────────────────────────────────────────

print("\n[2/3] Loading images from folder...")

manager = DataManager(path=IMAGE_FOLDER, batch_size=BATCH_SIZE, device=DEVICE)
loader  = manager.get_loader(source='local', shuffle=False)

n_images = len(loader.dataset)
print(f"Found {n_images} images.")

if n_images < MIN_CLUSTER_SIZE:
    print(f"\nWarning: only {n_images} images but MIN_CLUSTER_SIZE={MIN_CLUSTER_SIZE}.")
    print("Lower MIN_CLUSTER_SIZE further or add more images.")

# ─── Run inference ────────────────────────────────────────────────────────────

print("\n[3/3] Running inference...")

predictions, raw_images, strengths = run_inference(
    projection_head=projection_head,
    dimension_reducer=reducer,
    dataloader=loader,
    dino_extractor=dino,
    min_cluster_size=MIN_CLUSTER_SIZE,
    min_samples=MIN_SAMPLES,
    device=DEVICE
)

# ─── Results ─────────────────────────────────────────────────────────────────

n_clusters = len(set(predictions)) - (1 if -1 in predictions else 0)
n_noise    = int(np.sum(predictions == -1))

print("\n" + "=" * 55)
print("RESULTS")
print("=" * 55)
print(f"Images processed : {len(predictions)}")
print(f"Clusters found   : {n_clusters}")
print(f"Noise images     : {n_noise} ({n_noise / len(predictions) * 100:.1f}%)")
print("=" * 55)

if n_clusters == 0:
    print("\nNo clusters found. Try lowering MIN_CLUSTER_SIZE further.")
else:
    # ─── Metrics ─────────────────────────────────────────────────────────────
    # Re-extract features for metric computation
    # NMI and ARI need ground truth labels which custom folders don't have
    projection_head.eval()
    all_dino = []
    for imgs, _ in loader:
        all_dino.append(dino.extract_batch(imgs))
    dino_np = torch.cat(all_dino).numpy()

    with torch.no_grad():
        proj_np = projection_head(torch.tensor(dino_np)).numpy()
    reduced = reducer.transform(proj_np)

    metrics = evaluate_clustering(features=reduced, predicted_labels=predictions)

    print("\n" + "=" * 55)
    print("METRICS")
    print("=" * 55)
    print(f"Silhouette Score : {metrics.get('silhouette', float('nan')):.4f}")
    print("NMI / ARI        : not available (no ground truth labels)")
    print("=" * 55)

    # ─── t-SNE scatter ────────────────────────────────────────────────────────
    plot_clusters(reduced, predictions, title="Custom Images — Clusters")

    # ─── Galleries ───────────────────────────────────────────────────────────
    unique_clusters = sorted(set(predictions) - {-1})
    print(f"\nShowing galleries for {n_clusters} clusters...")

    for c_id in unique_clusters:
        mask    = predictions == c_id
        indices = np.where(mask)[0]

        sorted_order = np.argsort(strengths[indices])[::-1]
        top_indices  = indices[sorted_order[:10]]

        fig, axes = plt.subplots(1, len(top_indices), figsize=(15, 3))
        if len(top_indices) == 1:
            axes = [axes]

        for i, idx in enumerate(top_indices):
            img = raw_images[idx].permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            axes[i].imshow(img)
            axes[i].set_title(f"{strengths[idx]*100:.1f}%", fontsize=9)
            axes[i].axis('off')

        axes[0].set_ylabel(f"Cluster {c_id}", fontsize=11, fontweight='bold')
        fig.suptitle(
            f"Cluster {c_id} — {int(mask.sum())} images — "
            f"Top {len(top_indices)} shown",
            fontsize=13, y=1.02
        )
        plt.tight_layout()
        print(f"  Showing cluster {c_id} ({int(mask.sum())} images) — close window to continue...")
        plt.show()   # blocks until window is closed before showing next cluster

    print(f"\nDone.")