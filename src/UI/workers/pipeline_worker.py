"""
ui/workers/pipeline_worker.py

Runs the full inference pipeline in a background QThread so the UI
never freezes. Emits signals at each stage so the UI can show progress.

Usage:
    worker = PipelineWorker(image_folder, config)
    worker.progress.connect(status_bar.update)
    worker.finished.connect(main_window.on_results)
    worker.error.connect(main_window.on_error)
    worker.start()
"""

import torch
import numpy as np
from pathlib import Path
from PySide6.QtCore import QThread, Signal

from src.dino_hdbscan.data_manager import DataManager
from src.dino_hdbscan.feature_extractor import DinoFeatureExtractor
from src.dino_hdbscan.dimension_reduction import DimensionReducer
from src.dino_hdbscan.embedding_model import ProjectionHead
from src.dino_hdbscan.inference import run_inference
from src.dino_hdbscan.evaluate import evaluate_clustering
from src.dino_hdbscan.auto_min_cluster import auto_min_cluster_size


class PipelineConfig:
    """
    Holds all parameters needed to run inference.
    Pass this into PipelineWorker so the worker is self-contained.
    """

    def __init__(
        self,
        image_folder: str,
        umap_model_path: str,
        head_weights_path: str,
        stl10_data_path: str,
        batch_size: int = 16,
        min_samples: int = 2,
        device: str = None,
    ):
        self.image_folder = image_folder
        self.umap_model_path = umap_model_path
        self.head_weights_path = head_weights_path
        self.stl10_data_path = stl10_data_path
        self.batch_size = batch_size
        self.min_samples = min_samples
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")


class ClusterResult:
    """
    Everything the UI needs after a successful run.
    Passed through the finished signal.
    """

    def __init__(
        self,
        predictions: np.ndarray,       # (N,) cluster labels, -1 = noise
        raw_images: torch.Tensor,       # (N, C, H, W) normalized tensors
        strengths: np.ndarray,          # (N,) confidence scores 0-1
        image_paths: list,              # [str] original file paths, same order as images
        silhouette: float,
        n_clusters: int,
        n_noise: int,
        n_total: int,
    ):
        self.predictions = predictions
        self.raw_images = raw_images
        self.strengths = strengths
        self.image_paths = image_paths
        self.silhouette = silhouette
        self.n_clusters = n_clusters
        self.n_noise = n_noise
        self.n_total = n_total

    def images_for_cluster(self, cluster_id: int) -> list:
        """
        Returns list of dicts for all images in the given cluster, sorted
        by confidence descending.

        Each dict:
            {
                'index':    int,          original index in predictions array
                'path':     str,          file path
                'tensor':   Tensor,       (C, H, W) normalized tensor
                'strength': float,        confidence 0-1
            }
        """
        indices = np.where(self.predictions == cluster_id)[0]
        items = [
            {
                "index": int(idx),
                "path": self.image_paths[idx],
                "tensor": self.raw_images[idx],
                "strength": float(self.strengths[idx]),
            }
            for idx in indices
        ]
        return sorted(items, key=lambda x: x["strength"], reverse=True)

    def noise_images(self) -> list:
        """Returns images labelled as noise (-1), sorted by path."""
        indices = np.where(self.predictions == -1)[0]
        return [
            {
                "index": int(idx),
                "path": self.image_paths[idx],
                "tensor": self.raw_images[idx],
                "strength": 0.0,
            }
            for idx in indices
        ]

    def unique_cluster_ids(self) -> list:
        """Returns sorted list of valid cluster ids (no -1)."""
        return sorted(set(self.predictions) - {-1})


class PipelineWorker(QThread):
    """
    Background thread that runs the full inference pipeline.

    Signals:
        progress(str)          -- stage description for status bar
        stage_progress(int)    -- 0-100 percent for progress bar within a stage
        finished(ClusterResult)-- emitted on success
        error(str)             -- emitted on any exception
    """

    progress = Signal(str)
    stage_progress = Signal(int)
    finished = Signal(object)   # ClusterResult
    error = Signal(str)

    def __init__(self, config: PipelineConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            cfg = self.config

            # -- Validate inputs ----------------------------------------------
            self.progress.emit("Validating inputs...")
            if not Path(cfg.image_folder).is_dir():
                raise NotADirectoryError(f"Image folder not found: {cfg.image_folder}")
            if not Path(cfg.umap_model_path).exists():
                raise FileNotFoundError(f"UMAP model not found: {cfg.umap_model_path}")
            if not Path(cfg.head_weights_path).exists():
                raise FileNotFoundError(f"Projection head weights not found: {cfg.head_weights_path}")

            if self._cancelled:
                return

            # -- Load models --------------------------------------------------
            self.progress.emit("Loading DINO ViT-S/8...")
            dino = DinoFeatureExtractor(device=cfg.device)

            self.progress.emit("Loading projection head...")
            projection_head = ProjectionHead(input_dim=384, hidden_dim=256, output_dim=128)
            projection_head.load_state_dict(
                torch.load(cfg.head_weights_path, map_location="cpu", weights_only=True)
            )
            projection_head.eval()

            self.progress.emit("Loading UMAP model...")
            reducer = DimensionReducer(model_path=cfg.umap_model_path)

            if self._cancelled:
                return

            # -- Load images --------------------------------------------------
            self.progress.emit("Scanning image folder...")
            manager = DataManager(
                path=cfg.image_folder,
                batch_size=cfg.batch_size,
                device=cfg.device,
            )
            loader = manager.get_loader(source="local", shuffle=False)
            n_images = len(loader.dataset)

            if n_images < 100:
                raise ValueError(
                    f"Only {n_images} images found. Need at least 100 to cluster."
                )

            # Capture original file paths in dataloader order
            image_paths = [
                str(Path(cfg.image_folder) / f)
                for f in loader.dataset.image_files
            ]

            min_cluster_size = auto_min_cluster_size(n=n_images)

            if self._cancelled:
                return

            # -- Run inference ------------------------------------------------
            self.progress.emit(f"Extracting features for {n_images} images...")
            predictions, raw_images, strengths = run_inference(
                projection_head=projection_head,
                dimension_reducer=reducer,
                dataloader=loader,
                dino_extractor=dino,
                min_cluster_size=min_cluster_size,
                min_samples=cfg.min_samples,
                device=cfg.device,
            )

            if self._cancelled:
                return

            # -- Compute metrics ----------------------------------------------
            self.progress.emit("Computing silhouette score...")
            projection_head.eval()
            all_dino = []
            for imgs, _ in loader:
                all_dino.append(dino.extract_batch(imgs))
            dino_np = torch.cat(all_dino).numpy()

            with torch.no_grad():
                proj_np = projection_head(torch.tensor(dino_np)).numpy()
            reduced = reducer.transform(proj_np)

            metrics = evaluate_clustering(
                features=reduced, predicted_labels=predictions
            )

            # -- Package result -----------------------------------------------
            n_clusters = len(set(predictions)) - (1 if -1 in predictions else 0)
            n_noise = int(np.sum(predictions == -1))

            result = ClusterResult(
                predictions=predictions,
                raw_images=raw_images,
                strengths=strengths,
                image_paths=image_paths,
                silhouette=metrics.get("silhouette", float("nan")),
                n_clusters=n_clusters,
                n_noise=n_noise,
                n_total=n_images,
            )

            self.progress.emit(f"Done. Found {n_clusters} clusters.")
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))
