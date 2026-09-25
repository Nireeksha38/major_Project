from risk_engine.thresholds import RiskThresholds


class RiskClassifier:
    """
    Multi-factor risk evaluation engine that fuses density, movement speed,
    motion entropy, panic detection, and fall signals into a normalized risk score
    and classified threat level (GREEN / YELLOW / RED).
    """
    def __init__(self, thresholds=None):
        self.t = thresholds or RiskThresholds()

    def evaluate(self, people_count, density_score, relative_speed, motion_intensity, motion_entropy, panic_score, fall_detected, num_falls, fall_score, is_rush=False, is_panic=False):
        """
        Computes composite risk score and risk classification.
        All inputs normalized [0.0, 1.0] (except counts).
        """
        # Linear weighted baseline
        weighted_score = (
            self.t.WEIGHT_DENSITY * density_score +
            self.t.WEIGHT_SPEED * relative_speed +
            self.t.WEIGHT_MOTION * motion_intensity +
            self.t.WEIGHT_ENTROPY * motion_entropy +
            self.t.WEIGHT_PANIC * panic_score +
            self.t.WEIGHT_FALL * fall_score
        )

        risk_score = float(max(0.0, min(1.0, weighted_score)))

        # Priority overrides and non-linear risk escalation for emergency anomalies
        if is_panic:
            # Active Stampede / Panic Movement (Critical Level)
            panic_boost = 0.78 + 0.18 * panic_score + 0.04 * min(1.0, density_score)
            risk_score = max(risk_score, min(0.99, panic_boost))
        elif is_rush:
            # Active Crowd Rush / Speed Surge (Warning Level)
            rush_boost = 0.52 + 0.16 * max(relative_speed, motion_intensity)
            risk_score = max(risk_score, min(0.70, rush_boost))

        if fall_detected and num_falls > 0:
            # Fallen person in crowd presents immediate trampling hazard
            if people_count >= 6 or density_score >= 0.25:
                risk_score = max(risk_score, 0.82)  # Critical RED Trample Threat
            else:
                risk_score = max(risk_score, 0.55)  # Warning YELLOW Fall Event

        # Safety guarantee: If no fall, no rush, no panic, and not severe overcrowding, enforce GREEN
        if not fall_detected and not is_rush and not is_panic and density_score < 0.80 and motion_intensity < 0.35:
            risk_score = min(risk_score, self.t.GREEN_MAX - 0.05)

        # Classify Level
        if risk_score <= self.t.GREEN_MAX:
            risk_level = "GREEN"
            status_title = "NORMAL"
            summary_message = "Normal crowd flow. Orderly movement detected."
        elif risk_score <= self.t.YELLOW_MAX:
            risk_level = "YELLOW"
            status_title = "WARNING"
            if fall_detected:
                summary_message = f"Warning: {num_falls} fallen person(s) detected in monitored area."
            elif is_rush:
                summary_message = "Warning: sudden crowd rush / elevated velocity surge detected."
            elif density_score >= 0.65:
                summary_message = f"Warning: elevated crowd density ({people_count} people detected)."
            else:
                summary_message = "Warning: elevated crowd movement activity."
        else:
            risk_level = "RED"
            status_title = "CRITICAL"
            if fall_detected and (people_count >= 6 or density_score >= 0.25):
                summary_message = f"Critical: fallen person detected in crowd! High trample hazard!"
            elif is_panic:
                summary_message = "Critical: chaotic crowd panic and stampede risk detected!"
            else:
                summary_message = f"Critical: dangerous crowd congestion ({people_count} people)!"

        # Identify dominant risk factor
        factors = {
            "Density": density_score * self.t.WEIGHT_DENSITY,
            "Speed": relative_speed * self.t.WEIGHT_SPEED,
            "Motion": motion_intensity * self.t.WEIGHT_MOTION,
            "Entropy": motion_entropy * self.t.WEIGHT_ENTROPY,
            "Panic": panic_score * self.t.WEIGHT_PANIC,
            "Fall": fall_score * self.t.WEIGHT_FALL,
        }
        dominant_factor = max(factors, key=factors.get)

        return {
            "risk_level": risk_level,
            "risk_score": round(risk_score, 3),
            "status_title": status_title,
            "summary_message": summary_message,
            "dominant_factor": dominant_factor,
            "components": {
                "density": round(density_score, 3),
                "speed": round(relative_speed, 3),
                "motion": round(motion_intensity, 3),
                "entropy": round(motion_entropy, 3),
                "panic": round(panic_score, 3),
                "fall": round(fall_score, 3),
            }
        }

