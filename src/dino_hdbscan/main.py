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
    app.setOrganizationName("FYP")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
