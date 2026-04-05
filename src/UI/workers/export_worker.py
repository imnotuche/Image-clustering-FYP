"""
ui/workers/export_worker.py

Exports clustering results to disk in a background QThread.

Output structure:
    export_dir/
        cluster_0/
            image1.jpg
            image2.jpg
        cluster_1/
            ...
        noise/
            ...

Usage:
    worker = ExportWorker(result, export_dir)
    worker.progress.connect(...)
    worker.finished.connect(...)
    worker.error.connect(...)
    worker.start()
"""

import shutil
from pathlib import Path
from PySide6.QtCore import QThread, Signal

from UI.workers.pipeline_worker import ClusterResult


class ExportWorker(QThread):
    """
    Copies images into cluster subfolders. Does not modify originals.

    Signals:
        progress(str)    -- current operation description
        percent(int)     -- 0-100 overall progress
        finished(str)    -- emitted with the export directory path on success
        error(str)       -- emitted on any exception
    """

    progress = Signal(str)
    percent = Signal(int)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, result: ClusterResult, export_dir: str, parent=None):
        super().__init__(parent)
        self.result = result
        self.export_dir = Path(export_dir)

    def run(self):
        try:
            result = self.result
            export_dir = self.export_dir

            # Build full image list: clusters + noise
            all_items = []

            for cluster_id in result.unique_cluster_ids():
                for item in result.images_for_cluster(cluster_id):
                    all_items.append((f"cluster_{cluster_id}", item["path"]))

            for item in result.noise_images():
                all_items.append(("noise", item["path"]))

            total = len(all_items)
            if total == 0:
                raise ValueError("No images to export.")

            # Create output folders
            for cluster_id in result.unique_cluster_ids():
                (export_dir / f"cluster_{cluster_id}").mkdir(parents=True, exist_ok=True)
            (export_dir / "noise").mkdir(parents=True, exist_ok=True)

            # Copy files
            for i, (folder_name, src_path) in enumerate(all_items):
                src = Path(src_path)
                dst = export_dir / folder_name / src.name

                # Handle filename collisions (rare but possible)
                if dst.exists():
                    stem = src.stem
                    suffix = src.suffix
                    count = 1
                    while dst.exists():
                        dst = export_dir / folder_name / f"{stem}_{count}{suffix}"
                        count += 1

                shutil.copy2(src, dst)

                pct = int((i + 1) / total * 100)
                self.percent.emit(pct)

                if (i + 1) % 20 == 0 or i + 1 == total:
                    self.progress.emit(f"Exported {i + 1} / {total} images...")

            self.progress.emit("Export complete.")
            self.finished.emit(str(export_dir))

        except Exception as e:
            self.error.emit(str(e))
