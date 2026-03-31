"""
main.py — Entry point for the DINO Image Clustering UI

Run from the project root:
    python main.py

Requirements:
    pip install PySide6

Everything else (torch, umap-learn, hdbscan, etc.) is already in your
venv from the pipeline setup.
"""
import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from UI.dino_hdbscan_app import MainWindow


def main():
    # High DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("DINO Image Clustering")
    app.setOrganizationName("Adeleke University FYP")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
