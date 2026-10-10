import os
import joblib
import numpy as np
from config import Config


class FallDetector:
    """
    Intelligent temporal fall & abnormal posture detection engine.
    Fuses a trained Machine Learning Ensemble Classifier (HistGradientBoosting + RandomForest)
    with physical kinematic constraints (dynamic collapse, downward velocity, aspect transitions).
    """
    def __init__(self, aspect_ratio_thresh=None, velocity_thresh=None, model_path=None):
        self.aspect_ratio_thresh = aspect_ratio_thresh or Config.FALL_ASPECT_RATIO_THRESH
        self.velocity_thresh = velocity_thresh or Config.FALL_VELOCITY_THRESH
        self.fallen_tracks = {}  # track_id -> persistent fall count
        self.abnormal_tracks = {} # track_id -> persistent abnormal behavior count

        # Load trained ML Fall & Abnormal Behavior Classifier
        self.ml_classifier = None
        self._load_ml_model(model_path)

    def _load_ml_model(self, custom_path=None):
        search_paths = [
            custom_path,
            os.path.join(Config.BASE_DIR, "ml", "models", "fall_abnormal_classifier.joblib"),
            os.path.join(Config.BASE_DIR, "models", "weights", "fall_abnormal_classifier.joblib"),
            "ml/models/fall_abnormal_classifier.joblib",
            "models/weights/fall_abnormal_classifier.joblib"
        ]
        for p in search_paths:
            if p and os.path.exists(p):
                try:
                    self.ml_classifier = joblib.load(p)
                    print(f"[FallDetector] Loaded trained ML Fall/Abnormal Classifier: {p}")
                    break
                except Exception as e:
                    print(f"[FallDetector Warning] Failed loading ML weights from {p}: {e}")

    def update(self, active_tracks, frame_width=640, frame_height=480, optical_flow_motion=0.0, motion_entropy=0.0):
        """
        Evaluates active tracks for potential fall and abnormal posture events.
        Fuses ML probability with kinematic trajectory history.
        """
        if not active_tracks:
            return False, 0, [], 0.0

        current_active_ids = {t["id"] for t in active_tracks}
        # Clean up departed tracks
        self.fallen_tracks = {tid: count for tid, count in self.fallen_tracks.items() if tid in current_active_ids}
        self.abnormal_tracks = {tid: count for tid, count in self.abnormal_tracks.items() if tid in current_active_ids}

        fall_details = []

        for track in active_tracks:
            tid = track["id"]
            aspect_ratio = track.get("aspect_ratio", 0.4)
            aspect_history = track.get("aspect_history", [])
            height_history = track.get("height_history", [])
            v_vel = track.get("vertical_velocity", 0.0)
            h_vel = track.get("horizontal_velocity", 0.0)
            bbox = track["bbox"]
            x1, y1, x2, y2 = bbox

            bw = max(1.0, float(x2 - x1))
            bh = max(1.0, float(y2 - y1))
            inv_aspect = bh / bw
            bbox_h_ratio = bh / float(max(1, frame_height))
            bbox_w_ratio = bw / float(max(1, frame_width))
            norm_area = (bw * bh) / float(max(1, frame_width * frame_height))
            ground_prox = float(y2) / float(max(1, frame_height))

            # Edge Boundary Exclusion: If bounding box touches screen edge (e.g. cropped webcam view)
            margin = 8
            is_near_edge = (x1 <= margin or y1 <= margin or x2 >= frame_width - margin or y2 >= frame_height - margin)

            # Compute height drop and aspect delta
            height_drop_ratio = 0.0
            if len(height_history) >= 4:
                initial_height = float(np.mean(height_history[:3]))
                current_height = float(height_history[-1])
                height_drop_ratio = (initial_height - current_height) / max(1.0, initial_height)

            aspect_delta = 0.0
            if len(aspect_history) >= 4:
                aspect_delta = float(aspect_history[-1]) - float(np.mean(aspect_history[:3]))

            # ----------------- MACHINE LEARNING INFERENCE -----------------
            ml_fall_prob = 0.0
            ml_abnormal_prob = 0.0
            if self.ml_classifier is not None:
                try:
                    feat_vector = np.array([[
                        aspect_ratio,
                        inv_aspect,
                        bbox_h_ratio,
                        bbox_w_ratio,
                        norm_area,
                        v_vel,
                        h_vel,
                        height_drop_ratio,
                        aspect_delta,
                        ground_prox,
                        float(optical_flow_motion),
                        float(motion_entropy)
                    ]], dtype=np.float32)

                    probs = self.ml_classifier.predict_proba(feat_vector)[0]
                    # Classes: 0: NORMAL, 1: FALL_INCIDENT, 2: ABNORMAL_BEHAVIOR
                    ml_fall_prob = float(probs[1]) if len(probs) > 1 else 0.0
                    ml_abnormal_prob = float(probs[2]) if len(probs) > 2 else 0.0
                except Exception:
                    pass

            is_fall_candidate = False
            confidence = 0.0
            reason = ""

            # Check 1: Dynamic Transition - Rapid vertical collapse / drop with downward velocity
            if len(height_history) >= 4 and height_drop_ratio > 0.38 and v_vel > 5.0:
                is_fall_candidate = True
                confidence = min(0.98, height_drop_ratio + 0.40)
                reason = f"Sudden downward collapse (drop: {height_drop_ratio*100:.0f}%, v-vel: {v_vel:.1f})"
            elif len(aspect_history) >= 4 and aspect_delta >= 0.65 and v_vel > 3.0:
                is_fall_candidate = True
                confidence = min(0.98, 0.75 + aspect_delta * 0.3)
                reason = f"Rapid orientation transition from upright to lying (delta: +{aspect_delta:.2f})"

            # Check 2: Prone / lying flat on the floor (wide horizontal aspect ratio + lower scene)
            if not is_near_edge and aspect_ratio >= 1.40 and ground_prox >= 0.45:
                is_fall_candidate = True
                confidence = max(confidence, min(0.98, 0.78 + (aspect_ratio - 1.40) * 0.3))
                reason = reason or f"Prone horizontal posture on ground (AR: {aspect_ratio:.2f})"

            # Check 3: Machine Learning Model Confirmation
            if ml_fall_prob >= 0.50 and not is_near_edge:
                is_fall_candidate = True
                confidence = max(confidence, ml_fall_prob)
                reason = f"ML-Confirmed Fall Incident (Confidence: {ml_fall_prob*100:.1f}%)"

            if is_fall_candidate:
                self.fallen_tracks[tid] = self.fallen_tracks.get(tid, 0) + 1
                # Require 3 consecutive confirmed frames (or 2 if strong ML confidence > 0.85)
                thresh_frames = 2 if confidence >= 0.85 else 3
                if self.fallen_tracks[tid] >= thresh_frames:
                    fall_details.append({
                        "track_id": tid,
                        "bbox": bbox,
                        "confidence": round(confidence, 3),
                        "reason": reason or "Fallen person detected",
                        "ml_fall_prob": round(ml_fall_prob, 3),
                        "ml_abnormal_prob": round(ml_abnormal_prob, 3)
                    })
            else:
                if tid in self.fallen_tracks:
                    self.fallen_tracks[tid] = max(0, self.fallen_tracks[tid] - 1)

        num_falls = len(fall_details)
        fall_detected = num_falls > 0
        max_conf = max([float(f["confidence"]) for f in fall_details], default=0.0) if fall_detected else 0.0
        fall_score = min(1.0, float(num_falls) * 0.6 + (max_conf * 0.4 if fall_detected else 0.0))

        return fall_detected, num_falls, fall_details, fall_score

    def reset(self):
        self.fallen_tracks = {}
        self.abnormal_tracks = {}
