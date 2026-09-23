import numpy as np
from scipy.optimize import linear_sum_assignment
from config import Config
from tracking.tracker_utils import xyxy_to_xyah, xyah_to_xyxy, iou_cost_matrix
from detection.soft_nms import is_duplicate_box


class KalmanBoxTracker:
    """
    Kalman Filter for single person bounding box tracking in image space.
    State: [cx, cy, a, h, vx, vy, va, vh]^T where:
    cx, cy = bounding box center
    a = aspect ratio (w/h)
    h = height
    vx, vy, va, vh = velocities
    """
    count = 0

    def __init__(self, bbox, score=1.0, class_id=0):
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.class_id = class_id
        self.score = score

        # 8-dimensional state vector
        self.x = np.zeros((8, 1), dtype=np.float32)
        xyah = xyxy_to_xyah(bbox)
        self.x[:4, 0] = xyah

        # State transition matrix F
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        # Measurement matrix H
        self.H = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.H[i, i] = 1.0

        # Covariance matrices
        self.P = np.eye(8, dtype=np.float32) * 10.0
        self.P[4:, 4:] *= 100.0  # high uncertainty on initial velocities

        self.Q = np.eye(8, dtype=np.float32) * 1.0
        self.Q[4:, 4:] *= 0.01

        self.R = np.eye(4, dtype=np.float32) * 1.0

        # Track metadata
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.is_confirmed = False

        # Trajectory history (list of (cx, cy))
        self.history = [(float(xyah[0]), float(xyah[1]))]
        self.aspect_history = [float(xyah[2])]
        self.height_history = [float(xyah[3])]
        self.speed_history = [0.0]

    def predict(self):
        """Advances state vector and covariance by 1 time step."""
        if self.x[6, 0] + self.x[2, 0] <= 0:
            self.x[6, 0] = 0.0

        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hits = 0
        self.time_since_update += 1
        return self.get_state_box()

    def update(self, bbox, score=1.0):
        """Updates the state with observed bounding box."""
        self.time_since_update = 0
        self.hits += 1
        self.score = score

        z = xyxy_to_xyah(bbox).reshape(4, 1)

        # Kalman Update
        y = z - np.dot(self.H, self.x)  # measurement residual
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Kalman Gain

        self.x = self.x + np.dot(K, y)
        self.P = np.dot(np.eye(8) - np.dot(K, self.H), self.P)

        cx, cy = float(self.x[0, 0]), float(self.x[1, 0])
        a, h = float(self.x[2, 0]), float(self.x[3, 0])

        if len(self.history) > 0:
            prev_cx, prev_cy = self.history[-1]
            displacement = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
            self.speed_history.append(float(displacement))
            if len(self.speed_history) > 30:
                self.speed_history.pop(0)

        self.history.append((cx, cy))
        if len(self.history) > 60:
            self.history.pop(0)

        self.aspect_history.append(a)
        if len(self.aspect_history) > 30:
            self.aspect_history.pop(0)

        self.height_history.append(h)
        if len(self.height_history) > 30:
            self.height_history.pop(0)

    def get_state_box(self):
        """Returns current bounding box estimate in [x1, y1, x2, y2]."""
        xyah = self.x[:4, 0]
        return xyah_to_xyxy(xyah)

    def get_speed(self):
        """Returns recent average displacement speed in pixels/frame."""
        if len(self.speed_history) == 0:
            return 0.0
        return np.mean(self.speed_history[-5:])

    def get_vertical_velocity(self):
        """Returns downward vertical velocity (positive = moving down)."""
        return float(self.x[5, 0])


class DeepSORTTracker:
    """
    Deep SORT tracking system with Kalman Filtering, Hungarian Association,
    and Track lifecycle management.
    """
    def __init__(self, max_age=None, min_hits=None, iou_threshold=None):
        self.max_age = max_age or Config.TRACKER_MAX_AGE
        self.min_hits = min_hits or Config.TRACKER_MIN_HITS
        self.iou_threshold = iou_threshold or Config.TRACKER_IOU_THRESHOLD
        self.trackers = []
        self.frame_count = 0

    def update(self, detections=None):
        """
        Updates trackers with current frame detections.
        detections: list of [x1, y1, x2, y2, score, class_id] or Nx6 array, or None for prediction-only interval frame.
        Returns:
            active_tracks: list of dicts with:
                id, bbox [x1, y1, x2, y2], score, class_id,
                speed, trajectory, aspect_ratio, vertical_vel
        """
        self.frame_count += 1
        
        # 1. Predict all current tracks
        predicted_boxes = []
        to_delete = []
        for i, trk in enumerate(self.trackers):
            pos = trk.predict()
            if np.any(np.isnan(pos)):
                to_delete.append(i)
            else:
                predicted_boxes.append(pos)

        for i in reversed(to_delete):
            del self.trackers[i]

        # If this is a prediction-only frame (YOLO detection skipped for high FPS)
        if detections is None:
            active_tracks = []
            for trk in self.trackers:
                if trk.is_confirmed and trk.time_since_update <= 5:
                    bbox = trk.get_state_box()
                    active_tracks.append({
                        "id": trk.id,
                        "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                        "score": float(trk.score),
                        "class_id": int(trk.class_id),
                        "speed": float(trk.get_speed()),
                        "trajectory": list(trk.history),
                        "aspect_ratio": float(trk.x[2, 0]),
                        "height": float(trk.x[3, 0]),
                        "vertical_velocity": float(trk.get_vertical_velocity()),
                        "aspect_history": list(trk.aspect_history),
                        "height_history": list(trk.height_history)
                    })
            return active_tracks

        # 2. Extract detection bounding boxes
        if len(detections) > 0:
            det_boxes = [d[:4] for d in detections]
            det_scores = [d[4] for d in detections]
            det_classes = [d[5] if len(d) > 5 else 0 for d in detections]
        else:
            det_boxes = []
            det_scores = []
            det_classes = []

        # 3. Associate detections with tracks via Hungarian algorithm on IoU
        matched, unmatched_dets, unmatched_trks = self._associate(
            predicted_boxes, det_boxes, self.iou_threshold
        )

        # 4. Update matched trackers
        for t_idx, d_idx in matched:
            self.trackers[t_idx].update(det_boxes[d_idx], det_scores[d_idx])

        # 5. Create new trackers for unmatched detections
        for d_idx in unmatched_dets:
            trk = KalmanBoxTracker(
                det_boxes[d_idx],
                score=det_scores[d_idx],
                class_id=det_classes[d_idx]
            )
            self.trackers.append(trk)

        # 6. Remove dead trackers and collect active confirmed tracks
        active_tracks = []
        surviving_trackers = []

        for trk in self.trackers:
            # Check confirmation
            if not trk.is_confirmed and trk.hits >= self.min_hits:
                trk.is_confirmed = True

            # Keep active trackers within max_age
            if trk.time_since_update <= self.max_age:
                surviving_trackers.append(trk)

                # Return track if confirmed and recently updated (within 4 frames to bridge intervals)
                if trk.is_confirmed and trk.time_since_update <= 4:
                    bbox = trk.get_state_box()
                    active_tracks.append({
                        "id": trk.id,
                        "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                        "score": float(trk.score),
                        "class_id": int(trk.class_id),
                        "speed": float(trk.get_speed()),
                        "trajectory": list(trk.history),
                        "aspect_ratio": float(trk.x[2, 0]),
                        "height": float(trk.x[3, 0]),
                        "vertical_velocity": float(trk.get_vertical_velocity()),
                        "aspect_history": list(trk.aspect_history),
                        "height_history": list(trk.height_history)
                    })

        self.trackers = surviving_trackers

        # Deduplicate only true duplicate duplicate clones (IoU >= 0.80 or extreme IoM >= 0.90)
        # Preserves all adjacent and distinct individuals
        if len(active_tracks) > 1:
            deduped = []
            active_tracks.sort(key=lambda t: t["score"], reverse=True)
            for t in active_tracks:
                b1 = t["bbox"]
                overlap = False
                for d in deduped:
                    b2 = d["bbox"]
                    if is_duplicate_box(b1, b2, iou_thresh=0.80, iom_thresh=0.90):
                        overlap = True
                        break
                if not overlap:
                    deduped.append(t)
            active_tracks = deduped

        return active_tracks

    def _associate(self, predicted_boxes, det_boxes, iou_threshold):
        """Associates predictions with detections using linear sum assignment."""
        if len(predicted_boxes) == 0:
            return [], list(range(len(det_boxes))), []
        if len(det_boxes) == 0:
            return [], [], list(range(len(predicted_boxes)))

        cost_matrix = iou_cost_matrix(predicted_boxes, det_boxes)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched = []
        unmatched_trks = list(set(range(len(predicted_boxes))) - set(row_ind))
        unmatched_dets = list(set(range(len(det_boxes))) - set(col_ind))

        for r, c in zip(row_ind, col_ind):
            # Cost is 1 - IoU, so cost > (1 - iou_threshold) means IoU < threshold
            if cost_matrix[r, c] > (1.0 - iou_threshold):
                unmatched_trks.append(r)
                unmatched_dets.append(c)
            else:
                matched.append((r, c))

        return matched, unmatched_dets, unmatched_trks

    def reset(self):
        """Resets tracker state."""
        self.trackers = []
        self.frame_count = 0
