"""
ui/components/cluster_list.py

Vertical list of cluster entries in the results panel.
Clicking one emits cluster_selected with the cluster id.

Each entry shows:
    Cluster N
    X images  •  avg confidence%
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea,
    QFrame, QPushButton,
)
from PySide6.QtCore import Signal, Qt
import numpy as np


class ClusterEntry(QPushButton):
    """Single selectable cluster row."""

    def __init__(self, cluster_id, n_images: int, avg_strength: float, parent=None):
        super().__init__(parent)
        self.cluster_id = cluster_id
        self.setObjectName("clusterEntry")
        self.setCheckable(True)
        self.setFixedHeight(52)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(2)

        if cluster_id == -1:
            title_text = "Noise"
        else:
            title_text = f"Cluster {cluster_id}"

        title = QLabel(title_text)
        title.setObjectName("clusterEntryTitle")
        title.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        sub_text = f"{n_images} images  •  {avg_strength * 100:.0f}% avg conf"
        sub = QLabel(sub_text)
        sub.setObjectName("clusterEntrySub")
        sub.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        layout.addWidget(title)
        layout.addWidget(sub)


class ClusterList(QWidget):
    """
    Scrollable list of cluster entries.

    Signals:
        cluster_selected(int)  -- cluster_id, -1 for noise
    """

    cluster_selected = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("clusterList")
        self.setFixedWidth(200)
        self._entries = []
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QLabel("Clusters")
        header.setObjectName("clusterListHeader")
        header.setContentsMargins(12, 14, 12, 10)
        root.addWidget(header)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setObjectName("divider")
        root.addWidget(line)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._list_container = QWidget()
        self._list_layout = QVBoxLayout(self._list_container)
        self._list_layout.setContentsMargins(0, 4, 0, 4)
        self._list_layout.setSpacing(1)
        self._list_layout.addStretch()

        scroll.setWidget(self._list_container)
        root.addWidget(scroll)

    def populate(self, result):
        """
        Rebuild the list from a ClusterResult.

        Args:
            result: ClusterResult instance
        """
        self._clear()

        cluster_ids = result.unique_cluster_ids()

        for cid in cluster_ids:
            items = result.images_for_cluster(cid)
            avg_strength = float(np.mean([i["strength"] for i in items])) if items else 0.0
            entry = ClusterEntry(cid, len(items), avg_strength)
            entry.clicked.connect(lambda checked, c=cid: self._on_click(c))
            self._entries.append(entry)
            # Insert before the stretch
            self._list_layout.insertWidget(self._list_layout.count() - 1, entry)

        # Noise entry at the bottom
        noise_items = result.noise_images()
        if noise_items:
            noise_entry = ClusterEntry(-1, len(noise_items), 0.0)
            noise_entry.clicked.connect(lambda: self._on_click(-1))
            self._entries.append(noise_entry)
            self._list_layout.insertWidget(self._list_layout.count() - 1, noise_entry)

        # Auto-select first cluster
        if self._entries:
            self._entries[0].setChecked(True)

    def _on_click(self, cluster_id: int):
        # Deselect all, select clicked
        for entry in self._entries:
            entry.setChecked(entry.cluster_id == cluster_id)
        self.cluster_selected.emit(cluster_id)

    def _clear(self):
        self._entries.clear()
        while self._list_layout.count() > 1:  # keep the stretch
            item = self._list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def clear(self):
        self._clear()
