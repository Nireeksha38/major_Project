import numpy as np
import pytest
from tracking.deep_sort_tracker import DeepSORTTracker, KalmanBoxTracker
from tracking.tracker_utils import xyxy_to_xyah, xyah_to_xyxy


def test_bbox_conversions():
    orig_box = [10, 20, 50, 100]
    xyah = xyxy_to_xyah(orig_box)
    recov_box = xyah_to_xyxy(xyah)

    assert pytest.approx(orig_box[0], 0.1) == recov_box[0]
    assert pytest.approx(orig_box[1], 0.1) == recov_box[1]
    assert pytest.approx(orig_box[2], 0.1) == recov_box[2]
    assert pytest.approx(orig_box[3], 0.1) == recov_box[3]


def test_tracker_creation_and_association():
    tracker = DeepSORTTracker(min_hits=1, max_age=5)

    # Frame 1: 2 persons detected
    dets_f1 = [
        [10, 20, 50, 100, 0.9, 0],
        [150, 200, 190, 280, 0.85, 0]
    ]
    tracks_f1 = tracker.update(dets_f1)
    assert len(tracks_f1) == 2
    id1 = tracks_f1[0]["id"]
    id2 = tracks_f1[1]["id"]

    # Frame 2: Persons slightly moved
    dets_f2 = [
        [12, 22, 52, 102, 0.91, 0],
        [153, 202, 193, 282, 0.88, 0]
    ]
    tracks_f2 = tracker.update(dets_f2)
    assert len(tracks_f2) == 2
    # IDs must persist across frames
    assert tracks_f2[0]["id"] == id1
    assert tracks_f2[1]["id"] == id2
