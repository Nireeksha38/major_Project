import numpy as np
import pytest
from detection.soft_nms import calculate_iou, gaussian_soft_nms, linear_soft_nms, soft_nms


def test_calculate_iou():
    box1 = [0, 0, 10, 10]
    box2 = [0, 0, 10, 10]
    # Exact overlap -> IoU = 1.0
    assert pytest.approx(calculate_iou(box1, box2), 0.01) == 1.0

    # Half overlap
    box3 = [5, 0, 15, 10]
    # intersection: 5*10=50, union: 100+100-50=150 -> 50/150 = 0.333
    assert pytest.approx(calculate_iou(box1, box3), 0.01) == 0.333

    # No overlap
    box4 = [20, 20, 30, 30]
    assert calculate_iou(box1, box4) == 0.0


def test_gaussian_soft_nms_preserves_overlaps():
    # Two overlapping person bounding boxes
    boxes = np.array([
        [10, 10, 50, 100],   # Person A
        [15, 10, 55, 100]    # Person B (Dense overlap with A)
    ], dtype=np.float32)
    scores = np.array([0.90, 0.85], dtype=np.float32)
    classes = np.array([0, 0], dtype=np.int32)

    kept_boxes, kept_scores, kept_classes = gaussian_soft_nms(
        boxes, scores, classes, sigma=0.5, score_threshold=0.20
    )

    # In Soft-NMS, both persons are retained, but the second person's confidence is smoothly decayed
    assert len(kept_boxes) == 2
    assert kept_scores[0] == 0.90
    assert 0.20 < kept_scores[1] < 0.85  # Decayed but preserved!


def test_linear_soft_nms():
    boxes = np.array([
        [0, 0, 20, 20],
        [5, 0, 25, 20]
    ], dtype=np.float32)
    scores = np.array([0.95, 0.80], dtype=np.float32)

    kept_boxes, kept_scores, _ = linear_soft_nms(
        boxes, scores, score_threshold=0.20, iou_threshold=0.3
    )
    assert len(kept_boxes) >= 1
    assert kept_scores[0] == 0.95


def test_soft_nms_empty():
    boxes, scores, classes = soft_nms([], [])
    assert len(boxes) == 0
