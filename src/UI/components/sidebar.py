"""
ui/components/sidebar.py

Left panel of the main window.

Contains:
    - Image folder picker
    - Run / Cancel button
    - Advanced params (collapsible): batch size, min samples
    - Export button (enabled only after a successful run)
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QSpinBox, QGroupBox, QSizePolicy,
    QFrame, QProgressBar,
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont


class Sidebar(QWidget):
    """
    Signals:
        run_requested(str folder, int batch_size, int min_samples)
        cancel_requested()
        export_requested()
    """

    run_requested = Signal(str, int, int)
    cancel_requested = Signal()
    export_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(280)
        self._build_ui()
        self._connect()
        self.set_idle()

    # -------------------------------------------------------------------------
    # Build
    # -------------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 20, 16, 20)
        root.setSpacing(16)

        # Title
        title = QLabel("DINO Cluster")
        title.setObjectName("sidebarTitle")
        root.addWidget(title)

        divider = self._divider()
        root.addWidget(divider)

        # Folder picker
        root.addWidget(self._folder_section())

        # Run button
        self.run_btn = QPushButton("Run Clustering")
        self.run_btn.setObjectName("runButton")
        self.run_btn.setFixedHeight(42)
        root.addWidget(self.run_btn)

        # Cancel button
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("cancelButton")
        self.cancel_btn.setFixedHeight(36)
        self.cancel_btn.hide()
        root.addWidget(self.cancel_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)   # indeterminate
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()
        root.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        root.addWidget(self.status_label)

        root.addWidget(self._divider())

        # Advanced params
        root.addWidget(self._params_section())

        root.addStretch()

        root.addWidget(self._divider())

        # Export button
        self.export_btn = QPushButton("Export Results")
        self.export_btn.setObjectName("exportButton")
        self.export_btn.setFixedHeight(42)
        self.export_btn.setEnabled(False)
        root.addWidget(self.export_btn)

    def _folder_section(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        label = QLabel("Image Folder")
        label.setObjectName("fieldLabel")
        layout.addWidget(label)

        row = QHBoxLayout()
        row.setSpacing(6)

        self.folder_input = QLineEdit()
        self.folder_input.setPlaceholderText("Select a folder...")
        self.folder_input.setReadOnly(True)
        row.addWidget(self.folder_input)

        self.browse_btn = QPushButton("Browse")
        self.browse_btn.setObjectName("browseButton")
        self.browse_btn.setFixedWidth(64)
        row.addWidget(self.browse_btn)

        layout.addLayout(row)
        return container

    def _params_section(self) -> QGroupBox:
        group = QGroupBox("Advanced Parameters")
        group.setObjectName("paramsGroup")
        layout = QVBoxLayout(group)
        layout.setSpacing(10)

        # Batch size
        layout.addWidget(self._param_row("Batch Size", self._make_spinbox(
            name="batch_spin", value=16, min_=4, max_=128, step=4,
            tooltip="Images processed per batch. Lower if you run out of memory."
        )))

        # Min samples
        layout.addWidget(self._param_row("Min Samples", self._make_spinbox(
            name="min_samples_spin", value=2, min_=1, max_=20, step=1,
            tooltip="HDBSCAN min_samples. Lower = more clusters, more noise accepted."
        )))

        return group

    def _param_row(self, label_text: str, widget: QWidget) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label_text)
        lbl.setObjectName("fieldLabel")
        layout.addWidget(lbl)
        layout.addStretch()
        layout.addWidget(widget)
        return row

    def _make_spinbox(self, name, value, min_, max_, step, tooltip) -> QSpinBox:
        spin = QSpinBox()
        spin.setObjectName(name)
        spin.setRange(min_, max_)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setFixedWidth(72)
        spin.setToolTip(tooltip)
        return spin

    def _divider(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setObjectName("divider")
        return line

    # -------------------------------------------------------------------------
    # Connect
    # -------------------------------------------------------------------------

    def _connect(self):
        self.browse_btn.clicked.connect(self._browse_folder)
        self.run_btn.clicked.connect(self._on_run)
        self.cancel_btn.clicked.connect(self.cancel_requested.emit)
        self.export_btn.clicked.connect(self.export_requested.emit)

    # -------------------------------------------------------------------------
    # Slots
    # -------------------------------------------------------------------------

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.folder_input.setText(folder)

    def _on_run(self):
        folder = self.folder_input.text().strip()
        if not folder:
            self.set_status("Please select an image folder first.", error=True)
            return
        batch_size = self.findChild(QSpinBox, "batch_spin").value()
        min_samples = self.findChild(QSpinBox, "min_samples_spin").value()
        self.run_requested.emit(folder, batch_size, min_samples)

    # -------------------------------------------------------------------------
    # State management
    # -------------------------------------------------------------------------

    def set_idle(self):
        self.run_btn.show()
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
