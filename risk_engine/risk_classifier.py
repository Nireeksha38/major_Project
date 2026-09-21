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

        # Direct priority overrides for emergency anomalies
        if is_panic:
            # Active Stampede / Panic Movement
            risk_score = max(risk_score, 0.78)
        elif is_rush:
            # Active Crowd Rush / Speed Surge
            risk_score = max(risk_score, 0.55)

        if fall_detected and num_falls > 0:
            # Fallen person in crowd presents immediate trampling hazard
            if people_count >= 8 or density_score >= 0.35:
                risk_score = max(risk_score, 0.80)  # Critical RED
            else:
                risk_score = max(risk_score, 0.55)  # Warning YELLOW

        # Safety guarantee: If no fall, no rush, no panic, and not severe overcrowding, enforce GREEN
        if not fall_detected and not is_rush and not is_panic and density_score < 0.85:
            risk_score = min(risk_score, self.t.GREEN_MAX - 0.05)

        # Classify Level
        if risk_score <= self.t.GREEN_MAX:
            risk_level = "GREEN"
            status_title = "NORMAL"
            summary_message = "Normal crowd flow. No abnormal behavior, rush, or fall detected."
        elif risk_score <= self.t.YELLOW_MAX:
            risk_level = "YELLOW"
            status_title = "WARNING"
            if fall_detected:
                summary_message = "Warning: possible fallen person in low-density area."
            elif is_rush:
                summary_message = "Warning: sudden crowd rush / elevated movement detected."
            else:
                summary_message = "Warning: elevated crowd density."
        else:
            risk_level = "RED"
            status_title = "CRITICAL"
            if fall_detected and (people_count >= 8 or density_score >= 0.35):
                summary_message = "Critical: fallen person detected in crowded area! Trample risk!"
            elif is_panic:
                summary_message = "Critical: high-risk chaotic stampede movement detected!"
            else:
                summary_message = "Critical: severe overcrowding and uncontrolled compression!"

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
