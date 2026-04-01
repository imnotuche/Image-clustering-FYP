"""
ui/components/image_grid.py

Scrollable image gallery. Displays one cluster at a time as a grid of
thumbnails with confidence badges.

Key design decisions for performance on i3/8GB:
    - Thumbnails are generated once and cached as QPixmap (small, fast to paint)
    - Only the currently visible cluster is rendered — switching clusters clears
      and rebuilds the grid (cheap, no pixmap recalculation)
    - Thumbnails are 160x160px. At that size you can fit ~4-5 columns and
      display hundreds of images without meaningful memory pressure.
    - Raw tensor -> QPixmap conversion happens on demand, not upfront for all images.

Usage:
    grid = ImageGrid()
    grid.display_cluster(result.images_for_cluster(0))
"""

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QScrollArea, QGridLayout, QLabel, QVBoxLayout,
    QSizePolicy, QFrame,
)
from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QPixmap, QImage
import torch


THUMB_SIZE = 160   # px, square


def tensor_to_pixmap(tensor: torch.Tensor) -> QPixmap:
    """
    Convert a (C, H, W) normalized image tensor to a QPixmap thumbnail.

    Handles the ImageNet denormalization that happens to all images
    passed through DataManager's transform pipeline.
    """
    img = tensor.permute(1, 2, 0).numpy()          # (H, W, C) float32
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # 0-1
    img = (img * 255).clip(0, 255).astype(np.uint8)            # 0-255 uint8

    img = np.ascontiguousarray(img) 
    h, w, c = img.shape
    qimg = QImage(img.data, w, h, w * c, QImage.Format.Format_RGB888)
    pixmap = QPixmap.fromImage(qimg)
    return pixmap.scaled(
        THUMB_SIZE, THUMB_SIZE,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


class ThumbnailCard(QFrame):
    """
    Single image card: thumbnail + confidence badge.
    Emits clicked signal with the image item dict.
    """

    clicked = Signal(dict)

    def __init__(self, item: dict, parent=None):
        super().__init__(parent)
        self.item = item
        self.setObjectName("thumbnailCard")
        self.setFixedSize(THUMB_SIZE + 8, THUMB_SIZE + 28)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._build(item)

    def _build(self, item: dict):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # Thumbnail
        pixmap = tensor_to_pixmap(item["tensor"])
        img_label = QLabel()
        img_label.setPixmap(pixmap)
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(img_label)

        # Confidence badge
        pct = item["strength"] * 100
        badge = QLabel(f"{pct:.1f}%")
        badge.setObjectName("confidenceBadge")
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(badge)

    def mousePressEvent(self, event):
        self.clicked.emit(self.item)
        super().mousePressEvent(event)


class ImageGrid(QWidget):
    """
    Scrollable grid of ThumbnailCards.

    Signals:
        image_clicked(dict)   -- emitted when a thumbnail is clicked
    """

    image_clicked = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("imageGrid")
        self._build_ui()
        self._current_cluster = None

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Cluster title bar
        self.cluster_label = QLabel("")
        self.cluster_label.setObjectName("clusterTitle")
        self.cluster_label.setContentsMargins(16, 12, 16, 8)
        root.addWidget(self.cluster_label)

        # Scroll area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._grid_container = QWidget()
        self._grid_container.setObjectName("gridContainer")
        self._grid = QGridLayout(self._grid_container)
        self._grid.setSpacing(8)
        self._grid.setContentsMargins(16, 8, 16, 16)

        self.scroll.setWidget(self._grid_container)
        root.addWidget(self.scroll)

        # Empty state
        self.empty_label = QLabel("Run clustering to see results here.")
        self.empty_label.setObjectName("emptyState")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self.empty_label)

        self.scroll.hide()

    def display_cluster(self, items: list, cluster_id):
        """
        Clears current grid and populates with images from items.

        Args:
            items:      list of dicts from ClusterResult.images_for_cluster()
            cluster_id: int or -1 for noise
        """
        self._clear_grid()
        self.empty_label.hide()
        self.scroll.show()

        if cluster_id == -1:
            self.cluster_label.setText(f"Noise  —  {len(items)} images")
        else:
            self.cluster_label.setText(
                f"Cluster {cluster_id}  —  {len(items)} images"
            )

        # Calculate columns based on current widget width
        available_w = max(self.scroll.viewport().width(), 600)
        cols = max(1, available_w // (THUMB_SIZE + 16))

        for i, item in enumerate(items):
            card = ThumbnailCard(item)
            card.clicked.connect(self.image_clicked.emit)
            row, col = divmod(i, cols)
            self._grid.addWidget(card, row, col)

        # Fill last row with spacers so cards align left
        last_row_count = len(items) % cols
        if last_row_count != 0:
            for j in range(last_row_count, cols):
                spacer = QWidget()
                spacer.setFixedSize(THUMB_SIZE + 8, THUMB_SIZE + 28)
                row = len(items) // cols
                self._grid.addWidget(spacer, row, j)

        self.scroll.verticalScrollBar().setValue(0)
        self._current_cluster = cluster_id

    def show_empty(self, message: str = "Run clustering to see results here."):
        self._clear_grid()
        self.scroll.hide()
        self.cluster_label.setText("")
        self.empty_label.setText(message)
        self.empty_label.show()

    def _clear_grid(self):
        """Remove all widgets from the grid layout."""
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
