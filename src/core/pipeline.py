from collections import deque
from collections import Counter

def calculate_iou(boxA, boxB):
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou

class TileTracker:
    def __init__(self, iou_thresh=0.6, maxlen=5, conf_thresh=0.4):
        self.iou_thresh = iou_thresh
        self.maxlen = maxlen
        self.conf_thresh = conf_thresh
        # active_tracks: dict mapping track_id -> {'box': [], 'history': deque, 'last_label': str, 'frames_missed': int}
        self.active_tracks = {}
        self.next_track_id = 0

    def update(self, detected_boxes, classify_fn):
        """
        detected_boxes: list of dicts {"box": [x1, y1, x2, y2], "conf": float, "patch": numpy_array}
        classify_fn: function to classify a patch, returns string label
        """
        current_tracks = {}
        
        for det in detected_boxes:
            if det["conf"] < self.conf_thresh:
                continue
                
            box = det["box"]
            patch = det.get("patch")
            
            # Find best matching track
            best_iou = self.iou_thresh
            best_track_id = None
            
            for tid, track in self.active_tracks.items():
                iou = calculate_iou(box, track["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = tid
                    
            label = classify_fn(patch) if patch is not None else "unknown"
            
            if best_track_id is not None:
                # Update existing track
                track = self.active_tracks.pop(best_track_id)
                track["box"] = box
                track["history"].append(label)
                track["frames_missed"] = 0
                current_tracks[best_track_id] = track
            else:
                # Create new track
                history = deque(maxlen=self.maxlen)
                history.append(label)
                current_tracks[self.next_track_id] = {
                    "box": box,
                    "history": history,
                    "last_label": "unknown",
                    "frames_missed": 0
                }
                self.next_track_id += 1
                
        # Resolve labels for current tracks
        results = []
        for tid, track in current_tracks.items():
            counter = Counter(track["history"])
            most_common, count = counter.most_common(1)[0]
            if count >= 3:
                track["last_label"] = most_common
                
            results.append({
                "box": track["box"],
                "label": track["last_label"]
            })
            
        # Update active_tracks and prune missing ones if needed
        self.active_tracks = current_tracks
        return results
