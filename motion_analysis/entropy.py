import numpy as np
from config import Config


class MotionEntropyAnalyzer:
    """
    Computes directional Shannon entropy from optical flow to evaluate
    the randomness/disorder in crowd movement.
    0.0 = completely unidirectional, orderly motion
    1.0 = completely chaotic, multidirectional panic movement
    """
    def __init__(self, num_bins=None):
        self.num_bins = num_bins or Config.ENTROPY_BINS
        self.bin_edges = np.linspace(0.0, 360.0, self.num_bins + 1)
        self.max_entropy = np.log2(self.num_bins)

    def compute_entropy(self, angle_deg, magnitude, min_magnitude=0.8):
        """
        Computes normalized directional motion entropy.
        angle_deg: array of flow angles in degrees [0, 360)
        magnitude: array of flow magnitudes
        min_magnitude: threshold to discard static pixels / noise
        Returns:
            entropy: float normalized to [0.0, 1.0]
            histogram: array of directional bin weights
        """
        if angle_deg is None or magnitude is None or len(magnitude) == 0:
            return 0.0, np.zeros(self.num_bins)

        valid_mask = magnitude > min_magnitude
        valid_angles = angle_deg[valid_mask]
        valid_weights = magnitude[valid_mask]

        if len(valid_angles) < 10:
            return 0.0, np.zeros(self.num_bins)

        # Weighted histogram across angular bins (0-45, 45-90, etc.)
        hist, _ = np.histogram(valid_angles, bins=self.bin_edges, weights=valid_weights)
        total_weight = np.sum(hist)

        if total_weight <= 0:
            return 0.0, np.zeros(self.num_bins)

        # Probability distribution
        probabilities = hist / total_weight

        # Filter non-zero probabilities for log computation
        p_non_zero = probabilities[probabilities > 1e-7]
        shannon_entropy = -np.sum(p_non_zero * np.log2(p_non_zero))

        # Normalize to [0.0, 1.0]
        normalized_entropy = float(shannon_entropy / self.max_entropy) if self.max_entropy > 0 else 0.0
        normalized_entropy = max(0.0, min(1.0, normalized_entropy))

        return normalized_entropy, probabilities
