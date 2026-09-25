import numpy as np


class CrowdSpeedAnalyzer:
    """
    Analyzes relative crowd movement speed by fusing Deep SORT tracking trajectories
    and dense optical flow velocity field.
    Outputs normalized speed score in [0.0, 1.0], avg speed, max speed, and acceleration.
    """
    def __init__(self, max_speed_pixels=25.0):
        self.max_speed_pixels = max_speed_pixels
        self.speed_history = []
        self.smoothed_speed = 0.0

    def compute_speed(self, active_tracks=None, flow_summary=None):
        """
        Calculates robust crowd speed metrics by fusing track displacement and optical flow field.
        active_tracks: list of dicts from DeepSORTTracker
        flow_summary: dict from OpticalFlowAnalyzer with mean_magnitude, motion_ratio, etc.
        Returns:
            normalized_speed: float [0.0, 1.0]
            avg_pixels_per_frame: float
            max_pixels_per_frame: float
            acceleration: float (speed rate of change)
        """
        # 1. Track-based speed computation
        track_speed_norm = 0.0
        avg_px = 0.0
        max_px = 0.0
        valid_track_speeds = []

        if active_tracks and len(active_tracks) > 0:
            for t in active_tracks:
                s = t.get("speed", 0.0)
                if s > 0:
                    valid_track_speeds.append(s)

            if valid_track_speeds:
                avg_px = float(np.mean(valid_track_speeds))
                max_px = float(np.max(valid_track_speeds))
                track_speed_norm = min(1.0, avg_px / self.max_speed_pixels)

        # 2. Optical-flow-based speed computation (captures dense crowd motion even under occlusion)
        flow_speed_norm = 0.0
        if isinstance(flow_summary, dict) and len(flow_summary) > 0:
            mean_mag = float(flow_summary.get("mean_magnitude", 0.0))
            motion_ratio = float(flow_summary.get("motion_ratio", 0.0))
            norm_intensity = float(flow_summary.get("normalized_intensity", 0.0))
            
            # Flow velocity magnitude scaling
            mag_norm = min(1.0, mean_mag / 5.5)
            flow_speed_norm = max(norm_intensity, (mag_norm * 0.70 + motion_ratio * 0.30))
            flow_speed_norm = min(1.0, max(0.0, flow_speed_norm))

        # 3. Dual-source Fusion
        if len(valid_track_speeds) > 0 and track_speed_norm > 0.05:
            # Both tracking and optical flow are active
            raw_speed = 0.50 * track_speed_norm + 0.50 * flow_speed_norm
        elif flow_speed_norm > 0:
            # Tracking initializing or occluded, use optical flow field
            raw_speed = flow_speed_norm
        else:
            raw_speed = track_speed_norm

        # Smooth velocity with exponential moving average (EMA)
        self.smoothed_speed = 0.45 * raw_speed + 0.55 * self.smoothed_speed
        normalized_speed = float(max(0.0, min(1.0, self.smoothed_speed)))

        # Update temporal history for acceleration
        self.speed_history.append(normalized_speed)
        if len(self.speed_history) > 30:
            self.speed_history.pop(0)

        if len(self.speed_history) >= 5:
            recent_avg = float(np.mean(self.speed_history[-3:]))
            early_avg = float(np.mean(self.speed_history[:3]))
            acceleration = (recent_avg - early_avg) / float(max(1, len(self.speed_history) - 3))
        else:
            acceleration = 0.0

        if avg_px == 0.0 and flow_speed_norm > 0:
            avg_px = flow_speed_norm * self.max_speed_pixels
            max_px = avg_px * 1.5

        return normalized_speed, avg_px, max_px, float(acceleration)

    def reset(self):
        self.speed_history = []
        self.smoothed_speed = 0.0

