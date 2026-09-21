import numpy as np
from config import Config


class FallDetector:
    """
    Intelligent temporal fall detector.
    Accurately distinguishes genuine falls / ground collapses from normal seated poses,
    webcam portrait shots, and upright individuals.
    """
    def __init__(self, aspect_ratio_thresh=None, velocity_thresh=None):
        self.aspect_ratio_thresh = aspect_ratio_thresh or Config.FALL_ASPECT_RATIO_THRESH
        self.velocity_thresh = velocity_thresh or Config.FALL_VELOCITY_THRESH
        self.fallen_tracks = {}  # track_id -> persistent fall count

    def update(self, active_tracks, frame_width=640, frame_height=480):
        """
        Evaluates active tracks for potential fall events.
        active_tracks: list of dicts from DeepSORTTracker
        Returns:
            fall_detected: bool
            num_falls: int
            fall_details: list of dicts [{track_id, bbox, confidence, reason}]
            fall_score: float [0.0, 1.0]
        """
        if not active_tracks:
            return False, 0, [], 0.0

        current_active_ids = {t["id"] for t in active_tracks}
        # Clean up departed tracks
        self.fallen_tracks = {tid: count for tid, count in self.fallen_tracks.items() if tid in current_active_ids}

        fall_details = []

        for track in active_tracks:
            tid = track["id"]
            aspect_ratio = track.get("aspect_ratio", 0.4)
            aspect_history = track.get("aspect_history", [])
            height_history = track.get("height_history", [])
            v_vel = track.get("vertical_velocity", 0.0)
            bbox = track["bbox"]
            x1, y1, x2, y2 = bbox

            # Edge Boundary Exclusion: If bounding box touches screen edge (e.g. cropped webcam view)
            margin = 8
            is_near_edge = (x1 <= margin or y1 <= margin or x2 >= frame_width - margin or y2 >= frame_height - margin)

            is_fall_candidate = False
            confidence = 0.0
            reason = ""

            # Check 1: Dynamic Transition - Rapid vertical collapse / drop with downward velocity
            if len(height_history) >= 4:
                initial_height = float(np.mean(height_history[:3]))
                current_height = float(height_history[-1])
                height_drop_ratio = (initial_height - current_height) / max(1.0, initial_height)

                # Sudden downward vertical drop + velocity spike
                if height_drop_ratio > 0.38 and v_vel > 5.5:
                    is_fall_candidate = True
                    confidence = min(0.95, height_drop_ratio + 0.35)
                    reason = f"Sudden downward collapse (height drop: {height_drop_ratio*100:.0f}%, v-vel: {v_vel:.1f})"
                elif len(aspect_history) >= 4:
                    early_aspect = float(np.mean(aspect_history[:3]))
                    curr_aspect = float(aspect_history[-1])
                    # Person was standing/upright (aspect <= 0.55) and rapidly became horizontal (aspect >= 1.20) with downward movement
                    if early_aspect <= 0.55 and curr_aspect >= 1.20 and v_vel > 3.0:
                        is_fall_candidate = True
                        confidence = min(0.95, 0.70 + (curr_aspect - early_aspect) * 0.4)
                        reason = f"Rapid orientation transition from standing to lying ({early_aspect:.2f} -> {curr_aspect:.2f})"

            # Check 2: Prone / lying flat on the floor (wide horizontal aspect ratio + located in lower scene)
            # Must have high aspect ratio (w/h >= 1.45) AND not be cropped portrait shot (bottom y2 in lower half of screen)
            if not is_near_edge and aspect_ratio >= 1.45 and y2 >= frame_height * 0.45:
                is_fall_candidate = True
                confidence = max(confidence, min(0.95, 0.75 + (aspect_ratio - 1.45) * 0.3))
                reason = reason or f"Prone/horizontal posture on ground (aspect ratio: {aspect_ratio:.2f})"

            if is_fall_candidate:
                self.fallen_tracks[tid] = self.fallen_tracks.get(tid, 0) + 1
                # Require 3 consecutive confirmed frames to eliminate transient single-frame noise
                if self.fallen_tracks[tid] >= 3:
                    fall_details.append({
                        "track_id": tid,
                        "bbox": bbox,
                        "confidence": confidence,
                        "reason": reason or "Fallen person detected"
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
