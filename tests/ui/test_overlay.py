import sys
import pytest
from PyQt5.QtWidgets import QApplication
from src.ui.overlay import OverlayWindow

def test_overlay_creation():
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()
        
    window = OverlayWindow()
    window.update_detections([{"box": [10, 10, 20, 20], "label": "test"}])
    assert len(window.detections) == 1
    assert window.detections[0]["label"] == "test"
