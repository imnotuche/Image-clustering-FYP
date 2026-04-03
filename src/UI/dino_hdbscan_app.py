"""
UI/dino_hdbscan_app.py

Main application window.

Layout (horizontal root, all panels are rounded cards on #0D0D0D):
    ┌──────────────┬──────────────────────────────────────────────┐
    │              │  s1: StatsBar  (hidden until results)        │
    │  Sidebar     ├──────────────────────────────────────────────┤
    │  card        │  s2: ClusterList  (horizontal scroll chips)  │
    │              ├──────────────────────────────────────────────┤
    │              │  s3: ImageGrid  (fills remaining space)      │
    └──────────────┴──────────────────────────────────────────────┘
"""

from pathlib import Path
from importlib import resources
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QFileDialog, QMessageBox,
)

from UI.components.sidebar import Sidebar
from UI.components.stats_bar import StatsBar
from UI.components.image_grid import ImageGrid
from UI.components.cluster_list import ClusterList
from UI.workers.pipeline_worker import PipelineWorker, PipelineConfig
from UI.workers.export_worker import ExportWorker
from dino_hdbscan.data_manager import DataManager

_ROOT = Path(__file__).resolve().parent.parent.parent

STL10_DATA_PATH = str(_ROOT / "data" / "stl10")
UMAP_MODEL_PATH = str(_ROOT / "models" / "stl10_inference_umap.pkl")

stl10_manager = DataManager(STL10_DATA_PATH)


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("XCluster")
        self.setMinimumSize(1100, 700)
        self.resize(1320, 820)

        self._result          = None
        self._pipeline_worker = None
        self._export_worker   = None

        self._build_ui()
        self._apply_stylesheet()
        self._connect_signals()

    # -------------------------------------------------------------------------
    # Build UI
    # -------------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        central.setObjectName("rootCanvas")
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ── Section 1: sidebar card ───────────────────────────────────────────
        self.sidebar = Sidebar()
        root.addWidget(self.sidebar)

        # ── Section 2: right column (three stacked cards) ─────────────────────
        right_col = QVBoxLayout()
        right_col.setContentsMargins(0, 0, 0, 0)
        right_col.setSpacing(10)

        self.stats_bar    = StatsBar()
        self.cluster_list = ClusterList()
        self.image_grid   = ImageGrid()

        right_col.addWidget(self.stats_bar)        # s1 — fixed height, hidden until results
        right_col.addWidget(self.cluster_list)     # s2 — fixed height chip row
        right_col.addWidget(self.image_grid, stretch=1)  # s3 — fills rest

        right_wrapper = QWidget()
        right_wrapper.setObjectName("rightWrapper")
        right_wrapper.setLayout(right_col)
        root.addWidget(right_wrapper, stretch=1)

    # -------------------------------------------------------------------------
    # Signals
    # -------------------------------------------------------------------------

    def _connect_signals(self):
        self.sidebar.run_requested.connect(self._on_run_requested)
        self.sidebar.cancel_requested.connect(self._on_cancel)
        self.sidebar.export_requested.connect(self._on_export_requested)
        self.cluster_list.cluster_selected.connect(self._on_cluster_selected)

    # -------------------------------------------------------------------------
    # Pipeline
    # -------------------------------------------------------------------------

    def _on_run_requested(self, folder: str, batch_size: int, min_samples: int):
        head_weights = stl10_manager.load_model(name="stl10_projection_head", path=True)
        if not head_weights:
            QMessageBox.critical(
                self, "Missing Model",
                "Could not find projection head weights.\nRun run_pipeline.py first."
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
        self.image_grid.show_empty("Pipeline running — hang tight…")
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

        if result.unique_cluster_ids():
            self._on_cluster_selected(result.unique_cluster_ids()[0])

    def _on_pipeline_error(self, message: str):
        self.sidebar.set_idle()
        self.sidebar.set_status(f"Error: {message}", error=True)
        self.image_grid.show_empty("Pipeline error — check the sidebar for details.")
        QMessageBox.critical(self, "Pipeline Error", message)

    # -------------------------------------------------------------------------
    # Cluster navigation
    # -------------------------------------------------------------------------

    def _on_cluster_selected(self, cluster_id: int):
        if self._result is None:
            return
        items = (
            self._result.noise_images()
            if cluster_id == -1
            else self._result.images_for_cluster(cluster_id)
        )
        self.image_grid.display_cluster(items, cluster_id)

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------

    def _on_export_requested(self):
        if self._result is None:
            return
        export_dir = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if not export_dir:
            return

        self.sidebar.set_status("Exporting…")
        self._export_worker = ExportWorker(self._result, export_dir)
        self._export_worker.progress.connect(self.sidebar.set_status)
        self._export_worker.finished.connect(self._on_export_finished)
        self._export_worker.error.connect(self._on_export_error)
        self._export_worker.start()

    def _on_export_finished(self, export_dir: str):
        self.sidebar.set_status(f"Exported to {Path(export_dir).name}/")
        QMessageBox.information(self, "Export Complete",
                                f"Results exported to:\n{export_dir}")

    def _on_export_error(self, message: str):
        self.sidebar.set_status(f"Export failed: {message}", error=True)
        QMessageBox.critical(self, "Export Error", message)

    # -------------------------------------------------------------------------
    # Stylesheet
    # -------------------------------------------------------------------------

    def _apply_stylesheet(self):
        with resources.files("UI").joinpath("styles.qss").open("r",  encoding="utf-8") as f:
            self.setStyleSheet(f.read())