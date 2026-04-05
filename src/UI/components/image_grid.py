"""
UI/components/image_grid.py

Scrollable image grid inside a rounded card.
Shows placeholder text before any cluster is selected.

Performance notes (i3 / 8GB):
    - Thumbnails generated once per display_cluster() call
    - np.ascontiguousarray() ensures QImage buffer compatibility
    - Only the active cluster is rendered at any time
"""

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QScrollArea, QGridLayout, QLabel,
    QVBoxLayout, QHBoxLayout, QFrame, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QImage
import torch


THUMB_SIZE = 160


def tensor_to_pixmap(tensor: torch.Tensor) -> QPixmap:
    """(C,H,W) normalised tensor → scaled QPixmap thumbnail."""
    img = tensor.permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).clip(0, 255).astype(np.uint8)
    img = np.ascontiguousarray(img)   # QImage requires C-contiguous buffer

    h, w, c = img.shape
    qimg    = QImage(img.data, w, h, w * c, QImage.Format.Format_RGB888)
    pixmap  = QPixmap.fromImage(qimg)
    return pixmap.scaled(
        THUMB_SIZE, THUMB_SIZE,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


class ThumbnailCard(QFrame):
    """Thumbnail + confidence badge. Emits clicked(dict)."""

    clicked = Signal(dict)

    def __init__(self, item: dict, parent=None):
        super().__init__(parent)
        self.item = item
        self.setObjectName("thumbnailCard")
        self.setFixedSize(THUMB_SIZE + 12, THUMB_SIZE + 32)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 5)
        layout.setSpacing(4)

        img_lbl = QLabel()
        img_lbl.setPixmap(tensor_to_pixmap(item["tensor"]))
        img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(img_lbl)

        badge = QLabel(f"{item['strength'] * 100:.1f}%")
        badge.setObjectName("confidenceBadge")
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(badge)

    def mousePressEvent(self, event):
        self.clicked.emit(self.item)
        super().mousePressEvent(event)


class ImageGrid(QWidget):
    """
    Scrollable grid of ThumbnailCards inside a rounded card.

    Signals:
        image_clicked(dict)
    """

    image_clicked = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("gridCard")
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header bar
        header_bar = QWidget()
        header_bar.setFixedHeight(48)
        header_layout = QHBoxLayout(header_bar)
        header_layout.setContentsMargins(20, 0, 20, 0)

        self.cluster_label = QLabel("")
        self.cluster_label.setObjectName("gridHeader")
        header_layout.addWidget(self.cluster_label)
        header_layout.addStretch()

        root.addWidget(header_bar)

        # Scroll area
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._grid_container = QWidget()
        self._grid_container.setObjectName("gridContainer")
        self._grid = QGridLayout(self._grid_container)
        self._grid.setSpacing(10)
        self._grid.setContentsMargins(20, 8, 20, 20)

        self._scroll.setWidget(self._grid_container)
        root.addWidget(self._scroll)

        # Placeholder (shown before results)
        self._empty = QLabel("Run clustering to see results here.")
        self._empty.setObjectName("emptyState")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._empty)

        self._scroll.hide()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def display_cluster(self, items: list, cluster_id):
        self._clear_grid()
        self._empty.hide()
        self._scroll.show()

        label = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
        self.cluster_label.setText(f"{label}  —  {len(items)} images")

        available_w = max(self._scroll.viewport().width(), 600)
        cols        = max(1, available_w // (THUMB_SIZE + 22))

        for i, item in enumerate(items):
            card = ThumbnailCard(item)
            card.clicked.connect(self.image_clicked.emit)
            self._grid.addWidget(card, *divmod(i, cols))

        # Invisible spacers to left-align last row
        remainder = len(items) % cols
        if remainder:
            last_row = len(items) // cols
            for j in range(remainder, cols):
                sp = QWidget()
                sp.setFixedSize(THUMB_SIZE + 12, THUMB_SIZE + 32)
                self._grid.addWidget(sp, last_row, j)

        self._scroll.verticalScrollBar().setValue(0)

    def show_empty(self, message: str = "Run clustering to see results here."):
        self._clear_grid()
        self._scroll.hide()
        self.cluster_label.setText("")
        self._empty.setText(message)
        self._empty.show()

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _clear_grid(self):
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()