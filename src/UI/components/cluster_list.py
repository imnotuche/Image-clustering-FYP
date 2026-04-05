"""
UI/components/cluster_list.py

Horizontal scrollable row of cluster chips inside a rounded card.
Each chip shows cluster id, image count, and avg confidence.
Clicking emits cluster_selected(int).

Shows a placeholder message when no results are loaded.
"""

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QScrollArea, QPushButton,
)
from PySide6.QtCore import Signal, Qt
import numpy as np


class ClusterChip(QPushButton):
    """
    Single horizontal chip:
        Cluster N
        X images · YY% conf
    """

    def __init__(self, cluster_id: int, n_images: int,
                 avg_strength: float, parent=None):
        super().__init__(parent)
        self.cluster_id = int(cluster_id)
        self.setObjectName("clusterChip")
        self.setCheckable(True)
        self.setFixedSize(130, 60)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(3)

        title_text = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
        title = QLabel(title_text)
        title.setObjectName("chipTitle")
        title.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        sub = QLabel(f"{n_images} imgs · {avg_strength * 100:.0f}%")
        sub.setObjectName("chipSub")
        sub.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        layout.addWidget(title)
        layout.addWidget(sub)


class ClusterList(QWidget):
    """
    Horizontal scrollable chip row.

    Signals:
        cluster_selected(int)
    """

    cluster_selected = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("clusterCard")
        self.setFixedHeight(90)
        self._chips = []
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 0, 14, 0)
        outer.setSpacing(0)
        outer.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        # Scrollable horizontal strip
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(self._scroll.Shape.NoFrame)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setFixedHeight(78)

        # Inner container holds chips OR placeholder
        self._inner = QWidget()
        self._inner_layout = QHBoxLayout(self._inner)
        self._inner_layout.setContentsMargins(0, 8, 0, 8)
        self._inner_layout.setSpacing(8)
        self._inner_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        # Placeholder shown before any results
        self._placeholder = QLabel("Run clustering to see clusters here.")
        self._placeholder.setObjectName("clusterPlaceholder")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._inner_layout.addWidget(self._placeholder)

        self._scroll.setWidget(self._inner)
        outer.addWidget(self._scroll)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def populate(self, result):
        """Rebuild chips from a ClusterResult."""
        self._clear_chips()

        for cid in result.unique_cluster_ids():
            items        = result.images_for_cluster(cid)
            avg_strength = float(np.mean([i["strength"] for i in items])) if items else 0.0
            chip         = ClusterChip(cid, len(items), avg_strength)
            chip.clicked.connect(lambda checked, c=cid: self._on_click(c))
            self._chips.append(chip)
            self._inner_layout.addWidget(chip)

        # Noise chip at the end
        noise = result.noise_images()
        if noise:
            noise_chip = ClusterChip(-1, len(noise), 0.0)
            noise_chip.clicked.connect(lambda: self._on_click(-1))
            self._chips.append(noise_chip)
            self._inner_layout.addWidget(noise_chip)

        self._inner_layout.addStretch()

        # Auto-select first
        if self._chips:
            self._chips[0].setChecked(True)

    def clear(self):
        self._clear_chips()
        self._placeholder.show()

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    def _on_click(self, cluster_id: int):
        cluster_id = int(cluster_id)
        for chip in self._chips:
            chip.setChecked(chip.cluster_id == cluster_id)
        self.cluster_selected.emit(cluster_id)

    def _clear_chips(self):
        self._placeholder.hide()
        for chip in self._chips:
            self._inner_layout.removeWidget(chip)
            chip.deleteLater()
        self._chips.clear()
        # Remove any trailing stretch items
        while self._inner_layout.count() > 1:   # keep placeholder
            item = self._inner_layout.takeAt(self._inner_layout.count() - 1)
            if item.widget():
                item.widget().deleteLater()