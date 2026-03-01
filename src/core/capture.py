import mss
import numpy as np

class ScreenCapturer:
    def __init__(self, monitor_index=1):
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[monitor_index]
        self.roi_y_start = self.monitor["height"] - 200 # 依据 1080p 假定 880:1080
        
    def grab_frame(self):
        sct_img = self.sct.grab(self.monitor)
        # 将 BGRA 转为 BGR
        frame = np.array(sct_img)[:, :, :3]
        return frame
        
    def get_hand_roi(self, frame):
        return frame[-200:, :, :]
