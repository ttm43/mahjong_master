import numpy as np
from src.core.capture import ScreenCapturer

def test_screen_capture():
    capturer = ScreenCapturer(monitor_index=1)
    frame = capturer.grab_frame()
    assert isinstance(frame, np.ndarray)
    assert frame.shape[2] == 3 # BGR or RGB

def test_roi_crop():
    capturer = ScreenCapturer(monitor_index=1)
    # 模拟一张 1920x1080
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    roi = capturer.get_hand_roi(dummy_frame)
    # 假设手牌区在底部 200 像素
    assert roi.shape[0] == 200 
