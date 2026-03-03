import mss
import numpy as np


class ScreenCapturer:
    def __init__(self, monitor_index=1, hand_roi_height=200):
        self.sct = mss.mss()
        if monitor_index < 1 or monitor_index >= len(self.sct.monitors):
            max_index = len(self.sct.monitors) - 1
            raise ValueError(f"Invalid monitor_index={monitor_index}, available range is 1..{max_index}")

        self.monitor = self.sct.monitors[monitor_index]
        self.monitor_left = self.monitor["left"]
        self.monitor_top = self.monitor["top"]
        self.hand_roi_height = max(1, int(hand_roi_height))
        self.roi_y_start = max(0, self.monitor["height"] - self.hand_roi_height)

    def grab_frame(self):
        sct_img = self.sct.grab(self.monitor)
        # Convert BGRA to BGR.
        return np.array(sct_img)[:, :, :3]

    def get_hand_roi(self, frame):
        return frame[-self.hand_roi_height :, :, :]
