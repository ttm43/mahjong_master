import sys

from PyQt5.QtCore import QRect, Qt, pyqtSlot
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import QApplication, QMainWindow


class OverlayWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.WindowTransparentForInput
            | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)

        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(0, 0, screen.width(), screen.height())

        self.detections = []
        self.status_text = ""

    @pyqtSlot(list)
    def update_detections(self, detections):
        self.detections = detections
        self.update()

    @pyqtSlot(str)
    def update_status(self, status_text):
        self.status_text = status_text
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        box_pen = QPen(QColor(0, 255, 0))
        box_pen.setWidth(3)
        painter.setPen(box_pen)

        font = QFont("Arial", 16)
        font.setBold(True)
        painter.setFont(font)

        for det in self.detections:
            x1, y1, x2, y2 = det["box"]
            label = det["label"]

            rect = QRect(x1, y1, x2 - x1, y2 - y1)
            painter.drawRect(rect)

            painter.setPen(QPen(QColor(255, 0, 0)))
            painter.drawText(x1, y1 - 10, label)
            painter.setPen(box_pen)

        if self.status_text:
            painter.setPen(QPen(QColor(255, 255, 0)))
            painter.drawText(20, 30, self.status_text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OverlayWindow()
    window.update_detections([{"box": [100, 100, 200, 300], "label": "1m"}])
    window.update_status("Detector model missing")
    window.show()
    sys.exit(app.exec_())
