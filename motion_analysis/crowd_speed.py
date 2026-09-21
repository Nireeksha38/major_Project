import numpy as np


class CrowdSpeedAnalyzer:
    """
    Analyzes relative crowd movement speed using Deep SORT tracking trajectories.
    Outputs normalized speed score in [0.0, 1.0].
    """
    def __init__(self, max_speed_pixels=25.0):
        self.max_speed_pixels = max_speed_pixels
        self.speed_history = []

    def compute_speed(self, active_tracks):
        """
        Calculates relative crowd speed metrics from active tracked persons.
        active_tracks: list of dicts from DeepSORTTracker
        Returns:
            normalized_speed: float [0.0, 1.0]
            avg_pixels_per_frame: float
            max_pixels_per_frame: float
            acceleration: float (speed rate of change)
        """
        if not active_tracks or len(active_tracks) == 0:
            self.speed_history.append(0.0)
            if len(self.speed_history) > 30:
                self.speed_history.pop(0)
            return 0.0, 0.0, 0.0, 0.0

        speeds = [t.get("speed", 0.0) for t in active_tracks]
        avg_speed = float(np.mean(speeds))
        max_speed = float(np.max(speeds))

        # Normalized relative speed capped at 1.0
        normalized_speed = min(1.0, avg_speed / self.max_speed_pixels)

        # Calculate acceleration (rate of change over last 5 frames)
        self.speed_history.append(normalized_speed)
        if len(self.speed_history) > 30:
            self.speed_history.pop(0)

        if len(self.speed_history) >= 5:
            acceleration = (self.speed_history[-1] - self.speed_history[-5]) / 5.0
        else:
            acceleration = 0.0

        return float(normalized_speed), float(avg_speed), float(max_speed), float(acceleration)
