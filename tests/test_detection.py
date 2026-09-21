import numpy as np
import pytest
from detection.detector import YOLOv8Detector


def test_detector_initialization():
    detector = YOLOv8Detector()
    assert detector._model is not None


def test_detector_inference_on_synthetic_frame():
    detector = YOLOv8Detector()
    # Create synthetic RGB frame
    frame = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)

    boxes, scores, classes, detections = detector.detect_and_refine(frame, use_soft_nms=True)
    assert isinstance(boxes, np.ndarray)
    assert isinstance(scores, np.ndarray)
    assert isinstance(classes, np.ndarray)
    assert isinstance(detections, list)
