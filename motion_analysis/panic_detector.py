import numpy as np
from config import Config


class PanicDetector:
    """
    Evaluates crowd panic, sudden rush, and stampede behaviors using a temporal window.
    Combines optical flow intensity, directional entropy, crowd speed, and acceleration.
    """
    def __init__(self, window_size=None):
        self.window_size = window_size or Config.TEMPORAL_WINDOW
        self.speed_buffer = []
        self.entropy_buffer = []
        self.motion_buffer = []
        self.density_buffer = []

    def update(self, crowd_count=None, relative_speed=0.0, motion_intensity=0.0, motion_entropy=0.0, density_score=0.0, people_count=None):
        """
        Updates temporal buffer with current frame metrics and evaluates panic/rush signals.
        Returns:
            panic_score: float [0.0, 1.0]
            is_rush: bool
            is_panic: bool
            event_type: str ("NORMAL", "SUDDEN_RUSH", "PANIC_MOVEMENT", "STAMPEDE_RISK")
            reason: str
        """
        count = people_count if people_count is not None else (crowd_count if crowd_count is not None else 0)

        self.speed_buffer.append(relative_speed)
        self.entropy_buffer.append(motion_entropy)
        self.motion_buffer.append(motion_intensity)
        self.density_buffer.append(density_score)

        if len(self.speed_buffer) > self.window_size:
            self.speed_buffer.pop(0)
            self.entropy_buffer.pop(0)
            self.motion_buffer.pop(0)
            self.density_buffer.pop(0)

        # Minimum frames to establish temporal pattern
        if len(self.speed_buffer) < 5 or count == 0:
            return 0.0, False, False, "NORMAL", "Normal crowd movement"

        # Recent averages over the temporal window
        avg_speed = float(np.mean(self.speed_buffer))
        avg_entropy = float(np.mean(self.entropy_buffer))
        avg_motion = float(np.mean(self.motion_buffer))
        avg_density = float(np.mean(self.density_buffer))

        # Sudden acceleration: speed surge from early in window to current
        if len(self.speed_buffer) >= 8:
            early_speed = float(np.mean(self.speed_buffer[:4]))
            recent_speed = float(np.mean(self.speed_buffer[-4:]))
            speed_surge = max(0.0, recent_speed - early_speed)
        else:
            speed_surge = 0.0

        # Speed-gated disorder: Multi-directional entropy only represents panic if crowd speed or motion is elevated.
        # Calm walking in opposing directions (crosswalks/halls) is normal and should not trigger panic.
        speed_factor = min(1.0, max(0.15, avg_speed / 0.40))
        effective_entropy = avg_entropy * speed_factor

        # Composite Panic Formula
        panic_score = (
            0.35 * effective_entropy +
            0.30 * avg_speed +
            0.20 * avg_motion +
            0.15 * min(1.0, speed_surge * 2.5)
        )
        panic_score = float(max(0.0, min(1.0, panic_score)))

        # Behavior Classification
        is_panic = False
        is_rush = False
        event_type = "NORMAL"
        reason = "Normal crowd movement"

        # Requires at least 3 tracked people and genuine high-speed chaotic motion for panic/stampede
        if count >= 3:
            if panic_score >= 0.75 and avg_entropy >= 0.68 and avg_speed >= 0.60 and avg_motion >= 0.55:
                is_panic = True
                event_type = "STAMPEDE_RISK"
                reason = "High disorder and chaotic crowd rush detected"
            elif avg_entropy >= 0.70 and avg_motion >= 0.60 and avg_speed >= 0.62:
                is_panic = True
                event_type = "PANIC_MOVEMENT"
                reason = "Turbulent, high-speed multi-directional panic movement detected"
            elif speed_surge >= 0.38 or (avg_speed >= 0.68 and avg_motion >= 0.55):
                is_rush = True
                event_type = "SUDDEN_RUSH"
                reason = "Sudden crowd rush / rapid forward movement surge detected"
            elif avg_density >= 0.85 and avg_speed >= 0.55:
                is_rush = True
                event_type = "HIGH_DENSITY_RUSH"
                reason = "High crowd density with elevated movement speed"

        return panic_score, is_rush, is_panic, event_type, reason

    def reset(self):
        self.speed_buffer = []
        self.entropy_buffer = []
        self.motion_buffer = []
        self.density_buffer = []
