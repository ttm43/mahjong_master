import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt, QRect, pyqtSlot
from PyQt5.QtGui import QPainter, QPen, QColor, QFont

class OverlayWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Make the window frameless, stay on top, and transparent for mouse events
        self.setWindowFlags(
            Qt.FramelessWindowHint | 
            Qt.WindowStaysOnTopHint | 
            Qt.WindowTransparentForInput |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Default layout is fullscreen (assuming monitor 1)
        # Should be configured externally for specific roi
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(0, 0, screen.width(), screen.height())
        
        self.detections = []
        
    @pyqtSlot(list)
    def update_detections(self, detections):
        """
        detections: list of dicts {"box": [x1, y1, x2, y2], "label": str}
        """
        self.detections = detections
        self.update() # triggers paintEvent
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        pen = QPen(QColor(0, 255, 0)) # Green
        pen.setWidth(3)
        painter.setPen(pen)
        
        font = QFont("Arial", 16)
        font.setBold(True)
        painter.setFont(font)
        
        for det in self.detections:
            x1, y1, x2, y2 = det["box"]
            label = det["label"]
            
            # Draw bounding box
            rect = QRect(x1, y1, x2 - x1, y2 - y1)
            painter.drawRect(rect)
            
            # Draw label with a small background for visibility
            painter.setPen(QPen(QColor(255, 0, 0))) # Red text
            painter.drawText(x1, y1 - 10, label)
            painter.setPen(pen) # Restore green pen for next box

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = OverlayWindow()
    # Dummy data test
    window.update_detections([{"box": [100, 100, 200, 300], "label": "1m"}])
    window.show()
    sys.exit(app.exec_())
