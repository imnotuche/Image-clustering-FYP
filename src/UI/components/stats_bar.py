"""
ui/components/stats_bar.py

Horizontal bar at the top of the results area.
Shows key metrics after a successful run.

    [ Clusters: 7 ]  [ Images: 1000 ]  [ Noise: 43 (4.3%) ]  [ Silhouette: 0.412 ]
"""

from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QFrame
from PySide6.QtCore import Qt


class StatCard(QWidget):
    """Single metric tile: label above, value below."""

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(8)

        self._label = QLabel(label)
        self._label.setObjectName("statCardLabel")

        self._value = QLabel("--")
        self._value.setObjectName("statCardValue")

        layout.addWidget(self._label)
        layout.addWidget(self._value)
        self.setObjectName("statCard")

    def set_value(self, value: str):
        self._value.setText(value)


class StatsBar(QWidget):
    """
    Horizontal stats bar. Hidden until results are available.
    Call update_stats() after a successful pipeline run.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("statsBar")
        self.setFixedHeight(52)
        self._build_ui()
        self.hide()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)

        self._clusters_card = StatCard("Clusters")
        self._images_card = StatCard("Images")
        self._noise_card = StatCard("Noise")
        self._silhouette_card = StatCard("Silhouette")

        for card in [
            self._clusters_card,
            self._images_card,
            self._noise_card,
            self._silhouette_card,
        ]:
            layout.addWidget(card)

        layout.addStretch()

    def update_stats(
        self,
        n_clusters: int,
        n_total: int,
        n_noise: int,
        silhouette: float,
    ):
        noise_pct = n_noise / n_total * 100 if n_total > 0 else 0.0
        sil_str = f"{silhouette:.3f}" if silhouette == silhouette else "N/A"  # NaN check

        self._clusters_card.set_value(str(n_clusters))
        self._images_card.set_value(str(n_total))
        self._noise_card.set_value(f"{n_noise} ({noise_pct:.1f}%)")
        self._silhouette_card.set_value(sil_str)
        self.show()

    def clear(self):
        for card in [
            self._clusters_card,
            self._images_card,
            self._noise_card,
            self._silhouette_card,
        ]:
            card.set_value("--")
        self.hide()
