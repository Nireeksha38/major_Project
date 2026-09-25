import numpy as np
from config import Config


class PanicDetector:
    """
    Advanced physical crowd panic, surge, and stampede detection engine.
    Combines optical flow turbulence (Motion x Entropy), kinetic energy flux,
    directional disorder, acceleration surges, and crowd density coupling.
    Includes temporal hysteresis and exponential smoothing to deliver continuous,
    reliable, and flicker-free predictions.
    """
    def __init__(self, window_size=None):
        self.window_size = window_size or getattr(Config, "TEMPORAL_WINDOW", 15)
        self.speed_buffer = []
        self.entropy_buffer = []
        self.motion_buffer = []
        self.density_buffer = []
        
        # Temporal smoothing and state hysteresis
        self.smoothed_panic_score = 0.0
        self.panic_hold_counter = 0
        self.rush_hold_counter = 0
        self.last_event_type = "NORMAL"
        self.last_reason = "Normal crowd flow"

    def update(self, crowd_count=None, relative_speed=0.0, motion_intensity=0.0, motion_entropy=0.0, density_score=0.0, people_count=None):
        """
        Updates temporal buffers and evaluates crowd behavior.
        Returns:
            panic_score: float [0.0, 1.0] (smoothed, continuous)
            is_rush: bool
            is_panic: bool
            event_type: str ("NORMAL", "ELEVATED_ACTIVITY", "SUDDEN_RUSH", "HIGH_DENSITY_RUSH", "PANIC_MOVEMENT", "STAMPEDE_RISK")
            reason: str
        """
        count = people_count if people_count is not None else (crowd_count if crowd_count is not None else 0)

        # Append metrics to rolling temporal window
        self.speed_buffer.append(float(relative_speed))
        self.entropy_buffer.append(float(motion_entropy))
        self.motion_buffer.append(float(motion_intensity))
        self.density_buffer.append(float(density_score))

        if len(self.speed_buffer) > self.window_size:
            self.speed_buffer.pop(0)
            self.entropy_buffer.pop(0)
            self.motion_buffer.pop(0)
            self.density_buffer.pop(0)

        # Baseline check: at least 3 frames required for temporal initialization
        if len(self.speed_buffer) < 3:
            return 0.0, False, False, "NORMAL", "Initializing surveillance monitoring"

        # Moving averages over the temporal window
        avg_speed = float(np.mean(self.speed_buffer))
        avg_entropy = float(np.mean(self.entropy_buffer))
        avg_motion = float(np.mean(self.motion_buffer))
        avg_density = float(np.mean(self.density_buffer))

        # 1. Physics: Crowd Turbulence Index T = Motion x Entropy
        # In crowd dynamics (Helbing Social Force Model), high kinetic motion + high directional disorder = turbulence
        turbulence = avg_motion * avg_entropy

        # 2. Sudden Surge / Jerk Acceleration
        if len(self.speed_buffer) >= 6:
            early_motion = float(np.mean(self.motion_buffer[:3]))
            recent_motion = float(np.mean(self.motion_buffer[-3:]))
            early_speed = float(np.mean(self.speed_buffer[:3]))
            recent_speed = float(np.mean(self.speed_buffer[-3:]))
            motion_surge = max(0.0, recent_motion - early_motion)
            speed_surge = max(0.0, recent_speed - early_speed)
            surge_intensity = max(motion_surge, speed_surge)
        else:
            surge_intensity = 0.0

        # 3. Density-Coupled Pressure
        density_coupling = avg_density * avg_motion

        # 4. Continuous Composite Panic Formula
        raw_panic = (
            0.35 * turbulence +
            0.25 * avg_motion +
            0.20 * avg_entropy +
            0.10 * avg_speed +
            0.10 * min(1.0, surge_intensity * 2.2)
        )
        if avg_density >= 0.40:
            raw_panic += 0.08 * density_coupling

        raw_panic = float(max(0.0, min(1.0, raw_panic)))

        # 5. Exponential Moving Average for Continuous Smoothness
        alpha = 0.40
        self.smoothed_panic_score = alpha * raw_panic + (1.0 - alpha) * self.smoothed_panic_score
        panic_score = float(max(0.0, min(1.0, self.smoothed_panic_score)))

        # 6. Anomaly Triggers
        instant_panic = False
        instant_rush = False
        instant_event = "NORMAL"
        instant_reason = "Normal crowd flow"

        # Stampede Risk: Severe turbulence or high chaos with high motion
        if (turbulence >= 0.35 and avg_motion >= 0.50) or (panic_score >= 0.48 and avg_entropy >= 0.58 and avg_motion >= 0.45) or (turbulence >= 0.45):
            instant_panic = True
            instant_event = "STAMPEDE_RISK"
            instant_reason = "High disorder, chaotic rush, and movement turbulence detected (Stampede Risk)"
        elif avg_entropy >= 0.60 and avg_motion >= 0.42 and panic_score >= 0.40:
            instant_panic = True
            instant_event = "PANIC_MOVEMENT"
            instant_reason = "Turbulent, multi-directional panic movement and rapid scattering detected"
        elif surge_intensity >= 0.25 or (avg_motion >= 0.55 and avg_entropy < 0.60) or (avg_speed >= 0.55 and avg_motion >= 0.40):
            instant_rush = True
            instant_event = "SUDDEN_RUSH"
            instant_reason = "Sudden crowd rush / forward surge wave detected"
        elif avg_density >= 0.65 and (avg_speed >= 0.40 or avg_motion >= 0.35):
            instant_rush = True
            instant_event = "HIGH_DENSITY_RUSH"
            instant_reason = "High crowd density coupled with elevated movement velocity"
        elif panic_score >= 0.30 or avg_motion >= 0.38 or avg_speed >= 0.45:
            instant_rush = True
            instant_event = "ELEVATED_ACTIVITY"
            instant_reason = "Elevated crowd movement detected"

        # 7. Temporal Hysteresis & Retention Counter (Prevents Flickering)
        if instant_panic:
            self.panic_hold_counter = 12  # Hold panic state for 12 frames
            self.rush_hold_counter = 12
            self.last_event_type = instant_event
            self.last_reason = instant_reason
        elif instant_rush:
            self.rush_hold_counter = 10   # Hold rush state for 10 frames
            if self.panic_hold_counter > 0:
                self.panic_hold_counter -= 1
            else:
                self.last_event_type = instant_event
                self.last_reason = instant_reason
        else:
            if self.panic_hold_counter > 0:
                self.panic_hold_counter -= 1
            if self.rush_hold_counter > 0:
                self.rush_hold_counter -= 1

        is_panic = self.panic_hold_counter > 0
        is_rush = self.rush_hold_counter > 0 and not is_panic

        if is_panic:
            event_type = self.last_event_type if self.last_event_type in ["STAMPEDE_RISK", "PANIC_MOVEMENT"] else "STAMPEDE_RISK"
            reason = self.last_reason
        elif is_rush:
            event_type = self.last_event_type if self.last_event_type not in ["NORMAL", "STAMPEDE_RISK", "PANIC_MOVEMENT"] else "SUDDEN_RUSH"
            reason = self.last_reason
        else:
            event_type = "NORMAL"
            reason = "Normal crowd flow"
            self.last_event_type = "NORMAL"
            self.last_reason = reason

        return panic_score, is_rush, is_panic, event_type, reason

    def reset(self):
        self.speed_buffer = []
        self.entropy_buffer = []
        self.motion_buffer = []
        self.density_buffer = []
        self.smoothed_panic_score = 0.0
        self.panic_hold_counter = 0
        self.rush_hold_counter = 0
        self.last_event_type = "NORMAL"
        self.last_reason = "Normal crowd flow"

