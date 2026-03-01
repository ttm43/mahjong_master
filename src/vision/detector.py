from ultralytics import YOLO

class TileDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
    def detect(self, roi_img, conf_thresh=0.4):
        results = self.model(roi_img, conf=conf_thresh, verbose=False)
        boxes_out = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                boxes_out.append({"box": [int(x1), int(y1), int(x2), int(y2)], "conf": conf})
        return boxes_out
