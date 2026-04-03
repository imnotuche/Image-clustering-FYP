"""
UI/components/sidebar.py

Left panel — rendered as a rounded card on the #0D0D0D canvas.
No QFrame dividers anywhere.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QSpinBox, QGroupBox, QProgressBar,
)
from PySide6.QtCore import Signal, Qt


class Sidebar(QWidget):
    """
    Signals:
        run_requested(folder: str, batch_size: int, min_samples: int)
        cancel_requested()
        export_requested()
    """

    run_requested    = Signal(str, int, int)
    cancel_requested = Signal()
    export_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # outer widget is transparent — the card is an inner container
        self._build_ui()
        self._connect()
        self.set_idle()

    # -------------------------------------------------------------------------
    # Build
    # -------------------------------------------------------------------------

    def _build_ui(self):
        # Outer layout — just holds the card with margins
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Card
        self._card = QWidget()
        self._card.setObjectName("sidebarCard")
        outer.addWidget(self._card)

        inner = QVBoxLayout(self._card)
        inner.setContentsMargins(20, 24, 20, 20)
        inner.setSpacing(16)

        # ── Title ─────────────────────────────────────────────────────────────
        title_row = QHBoxLayout()
        title_row.setSpacing(0)
        x_lbl = QLabel("X")
        x_lbl.setObjectName("appTitleAccent")
        rest_lbl = QLabel("Cluster")
        rest_lbl.setObjectName("appTitle")
        title_row.addWidget(x_lbl)
        title_row.addWidget(rest_lbl)
        title_row.addStretch()
        inner.addLayout(title_row)

        # ── Folder picker ──────────────────────────────────────────────────────
        inner.addWidget(self._folder_section())

        # ── Run button ─────────────────────────────────────────────────────────
        self.run_btn = QPushButton("Run Clustering")
        self.run_btn.setObjectName("runButton")
        self.run_btn.setFixedHeight(44)
        self.run_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        inner.addWidget(self.run_btn)

        # ── Cancel button ──────────────────────────────────────────────────────
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("cancelButton")
        self.cancel_btn.setFixedHeight(36)
        self.cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cancel_btn.hide()
        inner.addWidget(self.cancel_btn)

        # ── Progress bar ───────────────────────────────────────────────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFixedHeight(3)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()
        inner.addWidget(self.progress_bar)

        # ── Status label ───────────────────────────────────────────────────────
        self.status_label = QLabel("")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setWordWrap(True)
        inner.addWidget(self.status_label)

        # ── Advanced params ────────────────────────────────────────────────────
        inner.addWidget(self._params_section())

        inner.addStretch()

        # ── Export button ──────────────────────────────────────────────────────
        self.export_btn = QPushButton("Export Results")
        self.export_btn.setObjectName("exportButton")
        self.export_btn.setFixedHeight(44)
        self.export_btn.setEnabled(False)
        self.export_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        inner.addWidget(self.export_btn)

    def _folder_section(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)

        lbl = QLabel("Image Folder")
        lbl.setObjectName("fieldLabel")
        layout.addWidget(lbl)

        row = QHBoxLayout()
        row.setSpacing(6)

        self.folder_input = QLineEdit()
        self.folder_input.setPlaceholderText("Select a folder…")
        self.folder_input.setReadOnly(True)
        self.folder_input.setFixedHeight(36)
        row.addWidget(self.folder_input)

        self.browse_btn = QPushButton("Browse")
        self.browse_btn.setObjectName("browseButton")
        self.browse_btn.setFixedWidth(70)
        self.browse_btn.setFixedHeight(36)
        self.browse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        row.addWidget(self.browse_btn)

        layout.addLayout(row)
        return container

    def _params_section(self) -> QGroupBox:
        group = QGroupBox("Advanced Parameters")
        group.setObjectName("paramsGroup")
        layout = QVBoxLayout(group)
        layout.setSpacing(10)
        layout.setContentsMargins(8, 14, 8, 8)

        layout.addWidget(self._param_row("Batch Size", self._spinbox(
            "batch_spin", 16, 4, 128, 4,
            "Images per batch. Lower if memory is tight."
        )))
        layout.addWidget(self._param_row("Min Samples", self._spinbox(
            "min_samples_spin", 2, 1, 20, 1,
            "HDBSCAN min_samples. Lower = more clusters."
        )))
        return group

    def _param_row(self, text: str, widget: QWidget) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(text)
        lbl.setObjectName("fieldLabel")
        layout.addWidget(lbl)
        layout.addStretch()
        layout.addWidget(widget)
        return row

    def _spinbox(self, name, value, lo, hi, step, tip) -> QSpinBox:
        s = QSpinBox()
        s.setObjectName(name)
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setValue(value)
        s.setFixedWidth(74)
        s.setFixedHeight(32)
        s.setToolTip(tip)
        return s

    # -------------------------------------------------------------------------
    # Connect / slots
    # -------------------------------------------------------------------------

    def _connect(self):
        self.browse_btn.clicked.connect(self._browse)
        self.run_btn.clicked.connect(self._run)
        self.cancel_btn.clicked.connect(self.cancel_requested.emit)
        self.export_btn.clicked.connect(self.export_requested.emit)

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.folder_input.setText(folder)

    def _run(self):
        folder = self.folder_input.text().strip()
        if not folder:
            self.set_status("Please select an image folder first.", error=True)
            return
        batch   = self.findChild(QSpinBox, "batch_spin").value()
        samples = self.findChild(QSpinBox, "min_samples_spin").value()
        self.run_requested.emit(folder, batch, samples)

    # -------------------------------------------------------------------------
    # State management
    # -------------------------------------------------------------------------

    def set_idle(self):
        self.run_btn.setEnabled(True)
        self.cancel_btn.hide()
        self.progress_bar.hide()
        self.status_label.setText("")

    def set_running(self):
        self.run_btn.setEnabled(False)
        self.cancel_btn.show()
        self.progress_bar.show()
        self.export_btn.setEnabled(False)

    def set_done(self):
        self.run_btn.setEnabled(True)
        self.cancel_btn.hide()
        self.progress_bar.hide()
        self.export_btn.setEnabled(True)

    def set_status(self, message: str, error: bool = False):
        self.status_label.setText(message)
        self.status_label.setProperty("error", str(error).lower())
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)