# 雀魂视觉辅助系统 Phase 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现一个端到端的 Python 管线，使用 mss 实时截图，裁剪手牌区域，使用 YOLO 框出麻将牌，并使用 CNN 识别牌面，最后在全屏叠加显示识别结果。

**Architecture:** 
1. `capture`: 使用 `mss` 截屏并负责坐标系管理分配。
2. `detector`: 加载 `YOLOv8n` 模型在给定的手牌 ROI区域寻找只包含类 0 (tile) 的边框。
3. `classifier`: 将 YOLO 取出的图像数组通过 `MobileNetV3-small` 或现有小分类模型得出 34 类中的 1 类。
4. `pipeline`: 将上述三者结合，并加入 IoU 历史追踪和置信度过滤门限。
5. `overlay`: 使用 `PyQt5` 创建一个全屏透明、鼠标穿透的窗口，在对应的绝对坐标上绘制框和文字。

**Tech Stack:** Python 3, OpenCV, Ultralytics (YOLOv8), PyTorch, PyQt5, mss.

---

### Task 1: 屏幕抓取与 ROI 裁剪基础类

**Files:**
- Create: `src/core/capture.py`
- Test: `tests/core/test_capture.py`

**Step 1: Write the failing test**
```python
import numpy as np
from src.core.capture import ScreenCapturer

def test_screen_capture():
    capturer = ScreenCapturer(monitor_index=1)
    frame = capturer.grab_frame()
    assert isinstance(frame, np.ndarray)
    assert frame.shape[2] == 3 # RGB

def test_roi_crop():
    capturer = ScreenCapturer(monitor_index=1)
    # 模拟一张 1920x1080
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    roi = capturer.get_hand_roi(dummy_frame)
    # 假设手牌区在底部 200 像素
    assert roi.shape[0] == 200 
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/core/test_capture.py -v`
Expected: FAIL 

**Step 3: Write minimal implementation**
```python
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
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/core/test_capture.py -v`
Expected: PASS

**Step 5: Commit**
```bash
git add tests/core/test_capture.py src/core/capture.py
git commit -m "feat: implement fast mss screen capture and hand ROI cropping"
```

---

### Task 2:  YOLO Tile Detector 封装类

**Files:**
- Create: `src/vision/detector.py`
- Test: `tests/vision/test_detector.py`

**Step 1: Write the failing test**
```python
import numpy as np
from src.vision.detector import TileDetector

def test_detector_returns_boxes():
    detector = TileDetector(model_path="models/dummy_yolo.pt") # 假定有模型
    dummy_roi = np.zeros((200, 1920, 3), dtype=np.uint8)
    boxes = detector.detect(dummy_roi)
    assert isinstance(boxes, list)
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/vision/test_detector.py -v`
Expected: FAIL 

**Step 3: Write minimal implementation**
*(Note: 实际执行中需先有真的 YOLO 模型或者使用 Magic Mock，这里写出业务代码骨架)*
```python
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
```

**Step 4: Run test to verify it passes**
*(需处理模型不存在的 mock 或者提前下载 nano 模型。跳过展示)*

**Step 5: Commit**
```bash
git add src/vision/detector.py
git commit -m "feat: add YOLO detector wrapper for extracting tile bounding boxes"
```

---

### Task 3: CNN Patch 分类器封装类

**Files:**
- Create: `src/vision/classifier.py`

**Step 1-4 (Concept):** 
需要通过 PyTorch 加载 MobileNetV3。接收从 OpenCV `frame[y1:y2, x1:x2]` 裁切出来的小图，进行 transform (resize 96x96, normalize) 后放入模型，返回 0-33 的 index，并映射到 `['1m', '2m', ... 'P', 'F', 'C']` 字典。由于需要真实权重文件才能跑通单元测试，此 Task 在执行时需要先构建虚假模型权重进行测试隔离。

**Step 5: Commit**
```bash
git add src/vision/classifier.py
git commit -m "feat: add CNN tile classifier wrapper"
```

---

### Task 4: 追踪去抖管线 (Pipeline Tracker)

**Files:**
- Create: `src/core/pipeline.py`

**Goal:** 
编写逻辑合并 YOLO 框和 CNN 结果。核心代码包含一个计算 IoU 的函数，检查前后两帧的 BBox：如果 `IoU > 0.6`，认为是同一张牌。维护一个 `collections.deque(maxlen=5)`，存储最近 5 帧的 CNN 分类结果，如果其中有 3 帧以上分类为 `5m`，则确认为 `5m`，否则输出 `unknown` 或保持上一帧缓存。抛弃低置信度的框。此任务纯逻辑，不需要模型，可以直接写死 BBox 进行 TDD 测试。

---

### Task 5: PyQt 透明 Overlay 渲染

**Files:**
- Create: `src/ui/overlay.py`

**Goal:**
```python
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QPen, QColor

class OverlayWindow(QMainWindow):
    # 配置 Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.WindowTransparentForInput
    # 接收 pipeline 传来的带坐标和名字的字典，在 paintEvent 中画矩形框和 drawText
```
实现一个可以盖在游戏上的全屏透明画板。

---
