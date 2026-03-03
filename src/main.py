import sys
import time

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication

from src.app_logging import configure_logging, get_logger
from src.config import load_app_config, resolve_path
from src.core.capture import ScreenCapturer
from src.core.pipeline import TileTracker
from src.ui.overlay import OverlayWindow
from src.vision.classifier import TileClassifier
from src.vision.detector import TileDetector


class VisionWorker(QThread):
    # Signal payload: [{"box": [x1, y1, x2, y2], "label": str}]
    update_signal = pyqtSignal(list)
    status_signal = pyqtSignal(str)

    def __init__(
        self,
        fps=10,
        monitor_index=1,
        hand_roi_height=200,
        detector_model_path="models/tile_detector.pt",
        classifier_model_path="models/tile_classifier.pt",
    ):
        super().__init__()
        self.logger = get_logger("worker")
        self.fps = max(1, int(fps))
        self.running = True

        self.detector_model_path = resolve_path(detector_model_path)
        self.classifier_model_path = resolve_path(classifier_model_path)

        self.capturer = ScreenCapturer(monitor_index=monitor_index, hand_roi_height=hand_roi_height)
        self.detector = TileDetector(self.detector_model_path, allow_missing_model=True)
        self.classifier = TileClassifier(self.classifier_model_path)
        self.tracker = TileTracker(maxlen=5, iou_thresh=0.6)

    def get_startup_status(self):
        messages = []
        if not self.detector.is_loaded:
            messages.append(f"Detector model missing: {self.detector_model_path}")
        if not self.classifier.weights_loaded:
            messages.append(f"Classifier weights missing: {self.classifier_model_path}")
        if not messages:
            return "Runtime ready"
        return " | ".join(messages)

    def run(self):
        frame_time = 1.0 / self.fps
        self.status_signal.emit(self.get_startup_status())

        while self.running:
            start_t = time.time()

            try:
                frame = self.capturer.grab_frame()
                roi = self.capturer.get_hand_roi(frame)

                try:
                    raw_boxes = self.detector.detect(roi)
                except Exception as exc:
                    self.logger.warning("Detector inference failed: %s", exc)
                    self.status_signal.emit(f"Detector inference failed: {exc}")
                    raw_boxes = []

                for det in raw_boxes:
                    x1, y1, x2, y2 = det["box"]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(roi.shape[1], x2), min(roi.shape[0], y2)
                    det["patch"] = roi[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else None

                tracked_results = self.tracker.update(raw_boxes, self.classifier.classify)

                roi_y_offset = self.capturer.roi_y_start
                monitor_x_offset = self.capturer.monitor_left
                monitor_y_offset = self.capturer.monitor_top

                final_detections = []
                for res in tracked_results:
                    rx1, ry1, rx2, ry2 = res["box"]
                    abs_box = [
                        rx1 + monitor_x_offset,
                        ry1 + roi_y_offset + monitor_y_offset,
                        rx2 + monitor_x_offset,
                        ry2 + roi_y_offset + monitor_y_offset,
                    ]
                    final_detections.append({"box": abs_box, "label": res["label"]})

                self.update_signal.emit(final_detections)
            except Exception as exc:
                self.logger.exception("Pipeline error")
                self.status_signal.emit(f"Pipeline error: {exc}")

            elapsed = time.time() - start_t
            time.sleep(max(0, frame_time - elapsed))

    def stop(self):
        self.running = False
        self.wait()


def main():
    cfg = load_app_config()
    logger = configure_logging(
        level=cfg.log_level,
        log_file=cfg.log_file,
        log_format=cfg.log_format,
        rotate_bytes=cfg.log_rotate_bytes,
        backup_count=cfg.log_backup_count,
    )

    app = QApplication.instance() or QApplication(sys.argv)

    overlay = OverlayWindow()
    overlay.show()

    worker = VisionWorker(
        fps=cfg.fps,
        monitor_index=cfg.monitor_index,
        hand_roi_height=cfg.hand_roi_height,
        detector_model_path=cfg.detector_model_path,
        classifier_model_path=cfg.classifier_model_path,
    )
    worker.update_signal.connect(overlay.update_detections)
    worker.status_signal.connect(overlay.update_status)
    overlay.update_status(worker.get_startup_status())
    worker.start()

    logger.info("Application started")
    try:
        exit_code = app.exec_()
    finally:
        worker.stop()
        logger.info("Application stopped")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
