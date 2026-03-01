import sys
import time
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread, pyqtSignal

from src.core.capture import ScreenCapturer
from src.vision.detector import TileDetector
from src.vision.classifier import TileClassifier
from src.core.pipeline import TileTracker
from src.ui.overlay import OverlayWindow

class VisionWorker(QThread):
    # Signal to send list of dicts: [{"box": [x1, y1, x2, y2], "label": str}]
    update_signal = pyqtSignal(list)
    
    def __init__(self, fps=10):
        super().__init__()
        self.fps = fps
        self.running = True
        
        # Initialize components
        self.capturer = ScreenCapturer(monitor_index=1)
        self.detector = TileDetector("models/dummy_yolo.pt") # Dummy path for now
        self.classifier = TileClassifier("models/dummy_mobilenet.pt")
        self.tracker = TileTracker(maxlen=5, iou_thresh=0.6)
        
    def run(self):
        # NOTE: For phase 1, we expect the weights to be missing but gracefully mocked/handled
        # In reality, without a real YOLO model, detector will crash if the weights don't exist.
        # For the sake of integration testing, we will mock the detector output if it crashes.
        
        frame_time = 1.0 / self.fps
        
        while self.running:
            start_t = time.time()
            
            try:
                # 1. Capture
                frame = self.capturer.grab_frame()
                roi = self.capturer.get_hand_roi(frame)
                
                # 2. Detect
                try:
                    raw_boxes = self.detector.detect(roi)
                except Exception as e:
                    # Fallback for testing without real weights: fake a box in the middle of ROI
                    h, w, _ = roi.shape
                    raw_boxes = [{"box": [w//2-20, h//2-30, w//2+20, h//2+30], "conf": 0.9}]
                    print(f"Detector fallback using dummy box. Reason: {e}")
                
                # Extract patches for CNN
                for det in raw_boxes:
                    x1, y1, x2, y2 = det["box"]
                    # clamp to roi bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(roi.shape[1], x2), min(roi.shape[0], y2)
                    
                    if y2 > y1 and x2 > x1:
                        det["patch"] = roi[y1:y2, x1:x2]
                    else:
                        det["patch"] = None
                
                # 3. Track and Classify
                tracked_results = self.tracker.update(raw_boxes, self.classifier.classify)
                
                # 4. Adjust coordinates from ROI to Full Screen
                roi_y_offset = self.capturer.roi_y_start
                final_detections = []
                for res in tracked_results:
                    rx1, ry1, rx2, ry2 = res["box"]
                    # Convert to absolute screen coordinates
                    abs_box = [rx1, ry1 + roi_y_offset, rx2, ry2 + roi_y_offset]
                    final_detections.append({
                        "box": abs_box,
                        "label": res["label"]
                    })
                    
                # 5. Emit to GUI
                self.update_signal.emit(final_detections)
                
            except Exception as e:
                print(f"Error in vision pipeline: {e}")
                
            # Sleep to maintain target FPS
            elapsed = time.time() - start_t
            sleep_t = max(0, frame_time - elapsed)
            time.sleep(sleep_t)
            
    def stop(self):
        self.running = False
        self.wait()

def main():
    app = QApplication(sys.argv)
    
    # 1. Create UI
    overlay = OverlayWindow()
    overlay.show()
    
    # 2. Create and start worker thread
    worker = VisionWorker(fps=10)
    # Connect worker signal to UI slot
    worker.update_signal.connect(overlay.update_detections)
    worker.start()
    
    # Run application
    exit_code = app.exec_()
    
    # Clean up
    worker.stop()
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
