import numpy as np
from src.core.pipeline import TileTracker, calculate_iou

def test_iou_calculation():
    boxA = [0, 0, 10, 10]
    boxB = [5, 5, 15, 15]
    iou = calculate_iou(boxA, boxB)
    assert 0.14 < iou < 0.15

def test_tracker():
    tracker = TileTracker(maxlen=5, iou_thresh=0.6)
    
    mock_patch = np.zeros((10, 10, 3), dtype=np.uint8)
    # Frame 1: Detected 1m
    res1 = tracker.update([{"box": [10, 10, 20, 20], "conf": 0.9, "patch": mock_patch}], lambda p: "1m")
    assert len(res1) == 1
    assert res1[0]["label"] == "unknown"  # Not enough history
    
    # Frame 2: Detected 1m
    res2 = tracker.update([{"box": [10, 10, 21, 21], "conf": 0.8, "patch": mock_patch}], lambda p: "1m")
    assert len(res2) == 1
    assert res2[0]["label"] == "unknown"
    
    # Frame 3: Detected 1m (threshold reached)
    res3 = tracker.update([{"box": [10, 10, 20, 20], "conf": 0.9, "patch": mock_patch}], lambda p: "1m")
    assert len(res3) == 1
    assert res3[0]["label"] == "1m"
    
    # Frame 4: Detected 2m (wrong classification once, still 1m due to history 1m, 1m, 1m, 2m)
    res4 = tracker.update([{"box": [10, 10, 20, 20], "conf": 0.9, "patch": mock_patch}], lambda p: "2m")
    assert len(res4) == 1
    assert res4[0]["label"] == "1m"
