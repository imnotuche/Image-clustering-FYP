"""
ui/app.py

Main application window. Owns all components and workers.
Handles state transitions: idle -> running -> results.

Layout:
    ┌─────────────┬───────────────────────────────────────────┐
    │             │  StatsBar                                 │
    │  Sidebar    ├──────────────┬────────────────────────────┤
    │  (controls) │ ClusterList  │  ImageGrid                 │
    │             │ (nav)        │  (thumbnails)              │
    └─────────────┴──────────────┴────────────────────────────┘
"""

import os
import torch
from pathlib import Path
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QFileDialog, QMessageBox, QSplitter,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from UI.components.sidebar import Sidebar
from UI.components.stats_bar import StatsBar
from UI.components.image_grid import ImageGrid
from UI.components.cluster_list import ClusterList
from UI.workers.pipeline_worker import PipelineWorker, PipelineConfig
from UI.workers.export_worker import ExportWorker

from dino_hdbscan.data_manager import DataManager

# ---------------------------------------------------------------------------
# Paths to trained models — adjust if your folder structure differs
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent.parent   # project root

STL10_DATA_PATH  = str(_ROOT / "data" / "stl10")
UMAP_MODEL_PATH  = str(_ROOT / "models" / "stl10_inference_umap.pkl")

stl10_manager = DataManager(STL10_DATA_PATH)

# Head weights path is loaded via DataManager; app.py uses a helper below
def _resolve_head_weights() -> str:
    """
    Tries to locate stl10_projection_head.pth in the models directory.
    Returns the path string or empty string if not found.
    """
    models_dir = _ROOT / "models" / "stl10"
    candidates = list(models_dir.glob("stl10_projection_head*.pth")) if models_dir.exists() else []
    if candidates:
        return str(sorted(candidates)[-1])   # latest if multiple
    return ""


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("XCluster")
        self.setMinimumSize(1100, 700)
        self.resize(1280, 800)

        self._result = None           # ClusterResult, set after successful run
        self._pipeline_worker = None
        self._export_worker = None

        self._build_ui()
        self._apply_stylesheet()
        self._connect_signals()

    # -------------------------------------------------------------------------
    # Build UI
    # -------------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Left: sidebar
        self.sidebar = Sidebar()
        root_layout.addWidget(self.sidebar)

        # Right: results area
        results_area = QWidget()
        results_area.setObjectName("resultsArea")
        results_layout = QVBoxLayout(results_area)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(0)

        # Stats bar at top of results
        self.stats_bar = StatsBar()
        results_layout.addWidget(self.stats_bar)

        # Cluster list + image grid side by side
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        content_splitter.setObjectName("contentSplitter")
        content_splitter.setHandleWidth(1)

        self.cluster_list = ClusterList()
        self.image_grid = ImageGrid()

        content_splitter.addWidget(self.cluster_list)
        content_splitter.addWidget(self.image_grid)
        content_splitter.setStretchFactor(0, 0)
        content_splitter.setStretchFactor(1, 1)

        results_layout.addWidget(content_splitter)

        root_layout.addWidget(results_area)

    # -------------------------------------------------------------------------
    # Connect signals
    # -------------------------------------------------------------------------

    def _connect_signals(self):
        self.sidebar.run_requested.connect(self._on_run_requested)
        self.sidebar.cancel_requested.connect(self._on_cancel)
        self.sidebar.export_requested.connect(self._on_export_requested)
        self.cluster_list.cluster_selected.connect(self._on_cluster_selected)

    # -------------------------------------------------------------------------
    # Pipeline control
    # -------------------------------------------------------------------------

    def _on_run_requested(self, folder: str, batch_size: int, min_samples: int):
        head_weights = stl10_manager.load_model(name = "stl10_projection_head", path = True)
        if not head_weights:
            QMessageBox.critical(
                self, "Missing Model",
                f"Could not find model"
                "Run run_pipeline.py first."
            )
            return

        config = PipelineConfig(
            image_folder=folder,
            umap_model_path=UMAP_MODEL_PATH,
            head_weights_path=head_weights,
            stl10_data_path=STL10_DATA_PATH,
            batch_size=batch_size,
            min_samples=min_samples,
        )

        self.sidebar.set_running()
        self.stats_bar.clear()
        self.cluster_list.clear()
        self.image_grid.show_empty("Running pipeline...")
        self._result = None

        self._pipeline_worker = PipelineWorker(config)
        self._pipeline_worker.progress.connect(self._on_pipeline_progress)
        self._pipeline_worker.finished.connect(self._on_pipeline_finished)
        self._pipeline_worker.error.connect(self._on_pipeline_error)
        self._pipeline_worker.start()

    def _on_cancel(self):
        if self._pipeline_worker and self._pipeline_worker.isRunning():
            self._pipeline_worker.cancel()
            self._pipeline_worker.wait(3000)
        self.sidebar.set_idle()
        self.image_grid.show_empty("Cancelled.")

    def _on_pipeline_progress(self, message: str):
        self.sidebar.set_status(message)

    def _on_pipeline_finished(self, result):
        self._result = result
        self.sidebar.set_done()
        self.sidebar.set_status(
            f"Done — {result.n_clusters} clusters, {result.n_noise} noise"
        )

        self.stats_bar.update_stats(
            n_clusters=result.n_clusters,
            n_total=result.n_total,
            n_noise=result.n_noise,
            silhouette=result.silhouette,
        )

        self.cluster_list.populate(result)

        # Auto-display first cluster
        if result.unique_cluster_ids():
            first_id = result.unique_cluster_ids()[0]
            self._on_cluster_selected(first_id)

    def _on_pipeline_error(self, message: str):
        self.sidebar.set_idle()
        self.sidebar.set_status(f"Error: {message}", error=True)
        self.image_grid.show_empty(f"Pipeline error:\n{message}")
        QMessageBox.critical(self, "Pipeline Error", message)

    # -------------------------------------------------------------------------
    # Cluster navigation
    # -------------------------------------------------------------------------

    def _on_cluster_selected(self, cluster_id: int):
        if self._result is None:
            return
        if cluster_id == -1:
            items = self._result.noise_images()
        else:
            items = self._result.images_for_cluster(cluster_id)
        self.image_grid.display_cluster(items, cluster_id)

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------

    def _on_export_requested(self):
        if self._result is None:
            return

        export_dir = QFileDialog.getExistingDirectory(
            self, "Select Export Folder"
        )
        if not export_dir:
            return

        self.sidebar.set_status("Exporting...")
        self._export_worker = ExportWorker(self._result, export_dir)
        self._export_worker.progress.connect(self.sidebar.set_status)
        self._export_worker.finished.connect(self._on_export_finished)
        self._export_worker.error.connect(self._on_export_error)
        self._export_worker.start()

    def _on_export_finished(self, export_dir: str):
        self.sidebar.set_status(f"Exported to {Path(export_dir).name}/")
        QMessageBox.information(
            self, "Export Complete",
            f"Results exported to:\n{export_dir}"
        )

    def _on_export_error(self, message: str):
        self.sidebar.set_status(f"Export failed: {message}", error=True)
        QMessageBox.critical(self, "Export Error", message)

    # -------------------------------------------------------------------------
    # Stylesheet
    # -------------------------------------------------------------------------

    def _apply_stylesheet(self):
        self.setStyleSheet("""
            /* ── Base ─────────────────────────────────────────────────────── */
            QMainWindow, QWidget {
                background-color: #0f1117;
                color: #e2e8f0;
                font-family: "Segoe UI", sans-serif;
                font-size: 13px;
            }

            /* ── Sidebar ───────────────────────────────────────────────────── */
            Sidebar {
                background-color: #161b27;
                border-right: 1px solid #1e2536;
            }

            #sidebarTitle {
                font-size: 17px;
                font-weight: 700;
                color: #f0f4ff;
                letter-spacing: 0.5px;
            }

            #fieldLabel {
                color: #8892a4;
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 0.8px;
            }

            QLineEdit {
                background-color: #1e2536;
                border: 1px solid #2a3348;
                border-radius: 6px;
                padding: 6px 10px;
                color: #c8d3e6;
                selection-background-color: #3b5bdb;
            }
            QLineEdit:focus {
                border-color: #4c6ef5;
            }

            #browseButton {
                background-color: #1e2536;
                border: 1px solid #2a3348;
                border-radius: 6px;
                padding: 6px 10px;
                color: #8892a4;
            }
            #browseButton:hover {
                background-color: #2a3348;
                color: #c8d3e6;
            }

            #runButton {
                background-color: #3b5bdb;
                border: none;
                border-radius: 8px;
                color: #ffffff;
                font-weight: 600;
                font-size: 13px;
            }
            #runButton:hover { background-color: #4c6ef5; }
            #runButton:pressed { background-color: #2f4ac7; }
            #runButton:disabled { background-color: #1e2a4a; color: #4a5568; }

            #cancelButton {
                background-color: transparent;
                border: 1px solid #e53e3e;
                border-radius: 6px;
                color: #e53e3e;
                font-size: 12px;
            }
            #cancelButton:hover { background-color: #2d1515; }

            #exportButton {
                background-color: #1e2d24;
                border: 1px solid #2d6a4f;
                border-radius: 8px;
                color: #52b788;
                font-weight: 600;
            }
            #exportButton:hover { background-color: #2d6a4f; color: #d8f3dc; }
            #exportButton:disabled { border-color: #1e2536; color: #4a5568; background-color: transparent; }

            QProgressBar {
                background-color: #1e2536;
                border: none;
                border-radius: 2px;
            }
            QProgressBar::chunk {
                background-color: #4c6ef5;
                border-radius: 2px;
            }

            #statusLabel {
                color: #8892a4;
                font-size: 11px;
            }
            #statusLabel[error="true"] {
                color: #fc8181;
            }

            #divider {
                color: #1e2536;
                background-color: #1e2536;
                border: none;
                max-height: 1px;
            }

            QGroupBox#paramsGroup {
                border: 1px solid #1e2536;
                border-radius: 8px;
                margin-top: 10px;
                padding: 8px;
                color: #8892a4;
                font-size: 11px;
            }
            QGroupBox#paramsGroup::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
                color: #8892a4;
                text-transform: uppercase;
                letter-spacing: 0.8px;
            }

            QSpinBox {
                background-color: #1e2536;
                border: 1px solid #2a3348;
                border-radius: 4px;
                padding: 4px 6px;
                color: #c8d3e6;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: #2a3348;
                border: none;
                width: 16px;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: #3b4a68;
            }

            /* ── Stats bar ─────────────────────────────────────────────────── */
            #statsBar {
                background-color: #161b27;
                border-bottom: 1px solid #1e2536;
            }

            #statCard {
                padding: 0 16px;
                border-right: 1px solid #1e2536;
            }

            #statCardLabel {
                color: #8892a4;
                font-size: 10px;
                text-transform: uppercase;
                letter-spacing: 0.8px;
            }

            #statCardValue {
                color: #f0f4ff;
                font-size: 15px;
                font-weight: 700;
            }

            /* ── Cluster list ──────────────────────────────────────────────── */
            ClusterList {
                background-color: #12161f;
                border-right: 1px solid #1e2536;
            }

            #clusterListHeader {
                color: #8892a4;
                font-size: 10px;
                text-transform: uppercase;
                letter-spacing: 1px;
                font-weight: 600;
            }

            #clusterEntry {
                background-color: transparent;
                border: none;
                border-radius: 0;
                text-align: left;
                padding: 0;
            }
            #clusterEntry:hover {
                background-color: #1a2035;
            }
            #clusterEntry:checked {
                background-color: #1e2a45;
                border-left: 3px solid #4c6ef5;
            }

            #clusterEntryTitle {
                color: #c8d3e6;
                font-size: 13px;
                font-weight: 500;
            }
            #clusterEntrySub {
                color: #4a5568;
                font-size: 11px;
            }

            /* ── Image grid ────────────────────────────────────────────────── */
            #resultsArea {
                background-color: #0f1117;
            }

            #clusterTitle {
                color: #f0f4ff;
                font-size: 15px;
                font-weight: 600;
                background-color: #0f1117;
            }

            #gridContainer {
                background-color: #0f1117;
            }

            #thumbnailCard {
                background-color: #161b27;
                border: 1px solid #1e2536;
                border-radius: 8px;
            }
            #thumbnailCard:hover {
                border-color: #4c6ef5;
                background-color: #1a2035;
            }

            #confidenceBadge {
                color: #8892a4;
                font-size: 10px;
            }

            #emptyState {
                color: #2a3348;
                font-size: 15px;
            }

            /* ── Scrollbars ────────────────────────────────────────────────── */
            QScrollBar:vertical {
                background: #0f1117;
                width: 6px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #2a3348;
                border-radius: 3px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover { background: #3b4a68; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

            /* ── Splitter ──────────────────────────────────────────────────── */
            QSplitter#contentSplitter::handle {
                background-color: #1e2536;
            }
        """)
