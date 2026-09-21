from config import Config


class RiskThresholds:
    """Configurable weights and boundaries for stampede risk classification."""
    
    # Feature weights (must sum to 1.0)
    WEIGHT_DENSITY = Config.WEIGHT_DENSITY       # 0.20
    WEIGHT_SPEED = Config.WEIGHT_SPEED           # 0.20
    WEIGHT_MOTION = Config.WEIGHT_MOTION         # 0.20
    WEIGHT_ENTROPY = Config.WEIGHT_ENTROPY       # 0.15
    WEIGHT_PANIC = Config.WEIGHT_PANIC           # 0.10
    WEIGHT_FALL = Config.WEIGHT_FALL             # 0.15

    # Classification boundaries
    GREEN_MAX = Config.RISK_GREEN_THRESHOLD      # 0.46 -> GREEN (0.00 - 0.46)
    YELLOW_MAX = Config.RISK_YELLOW_THRESHOLD    # 0.70 -> YELLOW (0.47 - 0.70)
    RED_MIN = 0.71                               # 0.71 - 1.00 -> RED

    # Max expected crowd capacity for normalized density calculation
    DEFAULT_CAPACITY = 60.0
