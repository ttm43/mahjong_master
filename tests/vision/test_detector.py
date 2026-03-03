import numpy as np
from unittest.mock import MagicMock, patch
from src.vision.detector import TileDetector

@patch("src.vision.detector.YOLO")
def test_detector_returns_boxes(mock_yolo_class):
    # Setup mock
    mock_model_instance = MagicMock()
    mock_yolo_class.return_value = mock_model_instance
    
    # Mock result list containing boxes
    mock_result = MagicMock()
    mock_box = MagicMock()
    import torch
    mock_box.xyxy = [torch.tensor([10.0, 20.0, 30.0, 40.0])]
    mock_box.conf = [torch.tensor(0.9)]
    mock_result.boxes = [mock_box]
    
    # YOLO object is callable: `self.model(roi_img, ...)`
    mock_model_instance.return_value = [mock_result]
    
    detector = TileDetector(model_path="models/dummy_yolo.pt")
    dummy_roi = np.zeros((200, 1920, 3), dtype=np.uint8)
    boxes = detector.detect(dummy_roi)
    
    assert isinstance(boxes, list)
    assert len(boxes) == 1
    assert boxes[0]["box"] == [10, 20, 30, 40]
    # conf is a float
    assert abs(boxes[0]["conf"] - 0.9) < 1e-6
