"""
UI/components/stats_bar.py

Horizontal metrics strip rendered as a rounded card.
Hidden by default — appears only after a successful run.
"""

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QSizePolicy,
)
from PySide6.QtCore import Qt


class _StatTile(QWidget):
    """One metric: small muted label stacked above a bold accent value."""

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 12, 24, 12)
        layout.setSpacing(2)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._lbl = QLabel(label.upper())
        self._lbl.setObjectName("statLabel")
        self._lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._val = QLabel("--")
        self._val.setObjectName("statValue")
        self._val.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self._lbl)
        layout.addWidget(self._val)

    def set_value(self, v: str):
        self._val.setText(v)


class StatsBar(QWidget):
    """
    Rounded card showing Clusters / Images / Noise / Silhouette.
    Hidden until update_stats() is called.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("statsCard")
        self.setFixedHeight(80)
        self._build()
        self.hide()

    def _build(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(0)

        self._clusters   = _StatTile("Clusters")
        self._images     = _StatTile("Images")
        self._noise      = _StatTile("Noise")
        self._silhouette = _StatTile("Silhouette")

        for tile in [self._clusters, self._images, self._noise, self._silhouette]:
            tile.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            layout.addWidget(tile)

        layout.addStretch()

    def update_stats(self, n_clusters: int, n_total: int,
                     n_noise: int, silhouette: float):
        noise_pct = n_noise / n_total * 100 if n_total > 0 else 0.0
        sil_str   = f"{silhouette:.3f}" if silhouette == silhouette else "N/A"

        self._clusters.set_value(str(n_clusters))
        self._images.set_value(str(n_total))
        self._noise.set_value(f"{n_noise}  ({noise_pct:.1f}%)")
        self._silhouette.set_value(sil_str)
        self.show()

    def clear(self):
        for tile in [self._clusters, self._images, self._noise, self._silhouette]:
            tile.set_value("--")
        self.hide()