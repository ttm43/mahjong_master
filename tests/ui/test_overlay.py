import sys

from PyQt5.QtWidgets import QApplication

from src.ui.overlay import OverlayWindow


def test_overlay_creation():
    app = QApplication.instance() or QApplication(sys.argv)

    window = OverlayWindow()
    window.update_detections([{"box": [10, 10, 20, 20], "label": "test"}])

    assert len(window.detections) == 1
    assert window.detections[0]["label"] == "test"

    window.close()
    app.processEvents()


def test_overlay_status_update():
    app = QApplication.instance() or QApplication(sys.argv)

    window = OverlayWindow()
    window.update_status("detector model missing")

    assert window.status_text == "detector model missing"

    window.close()
    app.processEvents()
