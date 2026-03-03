import numpy as np
from src.core.capture import ScreenCapturer


class _DummyMSS:
    def __init__(self):
        self.monitors = [
            {},
            {"left": 100, "top": 50, "width": 1920, "height": 1080},
        ]

    def grab(self, monitor):
        h, w = monitor["height"], monitor["width"]
        # Simulate BGRA screenshot result.
        frame = np.zeros((h, w, 4), dtype=np.uint8)
        frame[:, :, 3] = 255
        return frame


def test_screen_capture(monkeypatch):
    monkeypatch.setattr("src.core.capture.mss.mss", _DummyMSS)
    capturer = ScreenCapturer(monitor_index=1)

    frame = capturer.grab_frame()

    assert isinstance(frame, np.ndarray)
    assert frame.shape == (1080, 1920, 3)
    assert capturer.monitor_left == 100
    assert capturer.monitor_top == 50


def test_roi_crop(monkeypatch):
    monkeypatch.setattr("src.core.capture.mss.mss", _DummyMSS)
    capturer = ScreenCapturer(monitor_index=1, hand_roi_height=200)

    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    roi = capturer.get_hand_roi(dummy_frame)

    assert roi.shape[0] == 200
    assert capturer.roi_y_start == 880
