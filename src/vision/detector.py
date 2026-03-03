from pathlib import Path
from ultralytics import YOLO
from src.app_logging import get_logger

class TileDetector:
    def __init__(self, model_path, allow_missing_model=False):
        self.logger = get_logger("detector")
        self.model_path = str(model_path)
        self.model = None
        self.is_loaded = False
        path = Path(self.model_path)

        if path.exists():
            self.model = YOLO(self.model_path)
            self.is_loaded = True
            return

        if allow_missing_model:
            self.logger.warning("Detector model not found at %s. Detection will return empty results.", self.model_path)
            return

        self.model = YOLO(self.model_path)
        self.is_loaded = True
        
    def detect(self, roi_img, conf_thresh=0.4):
        if self.model is None:
            return []

        results = self.model(roi_img, conf=conf_thresh, verbose=False)
        boxes_out = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                boxes_out.append({"box": [int(x1), int(y1), int(x2), int(y2)], "conf": conf})
        return boxes_out
