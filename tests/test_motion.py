import numpy as np
import pytest
from motion_analysis.entropy import MotionEntropyAnalyzer
from motion_analysis.crowd_speed import CrowdSpeedAnalyzer
from motion_analysis.panic_detector import PanicDetector


def test_motion_entropy_uniform_vs_directional():
    analyzer = MotionEntropyAnalyzer(num_bins=8)

    # 1. Unidirectional motion (all moving at 90 degrees) -> Very Low Entropy
    unidirectional_angles = np.array([90.0, 91.0, 89.5, 90.5, 90.0] * 20, dtype=np.float32)
    unidirectional_mag = np.ones_like(unidirectional_angles) * 3.0
    entropy_uni, _ = analyzer.compute_entropy(unidirectional_angles, unidirectional_mag)

    # 2. Chaotic omnidirectional motion (scattered across 0-360) -> High Entropy
    chaotic_angles = np.random.uniform(0.0, 360.0, 200).astype(np.float32)
    chaotic_mag = np.ones_like(chaotic_angles) * 3.0
    entropy_chaotic, _ = analyzer.compute_entropy(chaotic_angles, chaotic_mag)

    assert entropy_uni < 0.25
    assert entropy_chaotic > 0.70
    assert 0.0 <= entropy_uni <= 1.0
    assert 0.0 <= entropy_chaotic <= 1.0


def test_crowd_speed_computation():
    analyzer = CrowdSpeedAnalyzer(max_speed_pixels=20.0)

    tracks = [
        {"speed": 10.0},
        {"speed": 14.0}
    ]
    norm_speed, avg_px, max_px, accel = analyzer.compute_speed(tracks)

    assert avg_px == 12.0
    assert max_px == 14.0
    assert 0.0 <= norm_speed <= 1.0


def test_panic_detector_temporal():
    detector = PanicDetector(window_size=10)

    # Normal calm state
    for _ in range(8):
        score, is_rush, is_panic, event, _ = detector.update(
            people_count=20, relative_speed=0.2, motion_intensity=0.2,
            motion_entropy=0.2, density_score=0.3
        )
    assert is_panic is False
    assert is_rush is False

    # Surge into panic
    for _ in range(8):
        score, is_rush, is_panic, event, _ = detector.update(
            people_count=35, relative_speed=0.85, motion_intensity=0.8,
            motion_entropy=0.85, density_score=0.7
        )
    assert score > 0.65
