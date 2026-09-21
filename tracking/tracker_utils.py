import numpy as np


def xyxy_to_xyah(bbox):
    """
    Convert bounding box [x1, y1, x2, y2] to [center_x, center_y, aspect_ratio (w/h), height]
    """
    x1, y1, x2, y2 = bbox
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    a = w / h
    return np.array([cx, cy, a, h], dtype=np.float32)


def xyah_to_xyxy(xyah):
    """
    Convert [center_x, center_y, aspect_ratio, height] to [x1, y1, x2, y2]
    """
    cx, cy, a, h = xyah
    w = a * h
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def iou_cost_matrix(tracks, detections):
    """
    Computes cost matrix based on 1 - IoU between tracks and detections.
    tracks: list of track bounding boxes [[x1,y1,x2,y2], ...]
    detections: list of detection bounding boxes [[x1,y1,x2,y2], ...]
    Returns: cost matrix of shape (num_tracks, num_detections)
    """
    num_tracks = len(tracks)
    num_dets = len(detections)

    if num_tracks == 0 or num_dets == 0:
        return np.empty((num_tracks, num_dets))

    cost_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)

    for t_idx, track_box in enumerate(tracks):
        for d_idx, det_box in enumerate(detections):
            # Calculate IoU
            x1 = max(track_box[0], det_box[0])
            y1 = max(track_box[1], det_box[1])
            x2 = min(track_box[2], det_box[2])
            y2 = min(track_box[3], det_box[3])

            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            inter = w * h

            area_t = max(0.0, (track_box[2] - track_box[0]) * (track_box[3] - track_box[1]))
            area_d = max(0.0, (det_box[2] - det_box[0]) * (det_box[3] - det_box[1]))
            union = area_t + area_d - inter

            iou = (inter / union) if union > 0 else 0.0
            cost_matrix[t_idx, d_idx] = 1.0 - iou

    return cost_matrix
