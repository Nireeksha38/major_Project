import os
import pytest
import numpy as np
from behavior.fall_detector import FallDetector


def test_fall_detector_initialization():
    detector = FallDetector()
    assert detector is not None
    assert detector.ml_classifier is not None


def test_fall_detector_normal_tracks():
    detector = FallDetector()
    # Upright track (aspect ratio 0.40, height drop 0, normal position)
    normal_track = [{
        "id": 1,
        "bbox": [100, 100, 160, 250],
        "aspect_ratio": 0.40,
        "aspect_history": [0.40, 0.41, 0.40, 0.40],
        "height_history": [150, 150, 149, 150],
        "vertical_velocity": 0.5,
        "horizontal_velocity": 1.2
    }]

    fall_detected, num_falls, details, score = detector.update(normal_track, frame_width=640, frame_height=480)
    assert not fall_detected
    assert num_falls == 0
    assert score == 0.0


def test_fall_detector_prone_detection():
    detector = FallDetector()
    # Prone track (wide aspect ratio 1.60, lying near bottom y2=420)
    prone_track = [{
        "id": 2,
        "bbox": [100, 320, 260, 420],
        "aspect_ratio": 1.60,
        "aspect_history": [0.40, 0.90, 1.40, 1.60],
        "height_history": [150, 120, 105, 100],
        "vertical_velocity": 6.5,
        "horizontal_velocity": 1.0
    }]

    # Update 3 consecutive frames to confirm
    for _ in range(3):
        fall_detected, num_falls, details, score = detector.update(prone_track, frame_width=640, frame_height=480)

    assert fall_detected
    assert num_falls == 1
    assert score > 0.5
    assert len(details) == 1
    assert "ml_fall_prob" in details[0]
