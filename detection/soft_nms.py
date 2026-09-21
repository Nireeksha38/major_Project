import numpy as np


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    Boxes format: [x1, y1, x2, y2]
    """
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])

    inter_width = max(0.0, x2_min - x1_max)
    inter_height = max(0.0, y2_min - y1_max)
    inter_area = inter_width * inter_height

    area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def calculate_iom(box1, box2):
    """
    Calculate Intersection over Minimum Area (IoM / Enclosure Ratio).
    Detects if one box is nested inside or heavily overlapping another box (e.g. head/torso vs whole body).
    """
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])

    inter_width = max(0.0, x2_min - x1_max)
    inter_height = max(0.0, y2_min - y1_max)
    inter_area = inter_width * inter_height

    area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    min_area = min(area1, area2)

    if min_area <= 0:
        return 0.0

    return inter_area / min_area


def is_duplicate_box(box1, box2, iou_thresh=0.38, iom_thresh=0.50):
    """
    Returns True if box1 and box2 represent the same physical person
    either via standard IoU overlap OR nested child-parent enclosure (IoM).
    """
    iou = calculate_iou(box1, box2)
    if iou >= iou_thresh:
        return True
    iom = calculate_iom(box1, box2)
    if iom >= iom_thresh:
        return True
    return False


def calculate_iou_vectorized(box, boxes):
    """
    Calculate IoU between a single box and an array of boxes.
    box: [x1, y1, x2, y2]
    boxes: Nx4 array of boxes
    """
    if len(boxes) == 0:
        return np.array([])

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_box = max(0.0, (box[2] - box[0]) * (box[3] - box[1]))
    area_boxes = np.maximum(0.0, (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    union_area = area_box + area_boxes - inter_area

    iou = np.zeros_like(inter_area)
    valid = union_area > 0
    iou[valid] = inter_area[valid] / union_area[valid]
    return iou


def gaussian_soft_nms(boxes, scores, classes=None, sigma=0.5, score_threshold=0.35, iou_threshold=0.45):
    """
    Gaussian Soft-NMS:
    Score decay formula: s_i = s_i * exp(-(iou(M, b_i)^2) / sigma)
    Preserves overlapping detections in dense crowds while penalizing duplicate candidate boxes.
    """
    if len(boxes) == 0:
        return np.empty((0, 4)), np.array([]), np.array([]) if classes is not None else np.array([])

    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    if classes is None:
        classes = np.zeros(len(boxes), dtype=np.int32)
    else:
        classes = np.array(classes, dtype=np.int32)

    kept_boxes = []
    kept_scores = []
    kept_classes = []

    N = len(boxes)
    pos_boxes = boxes.copy()
    pos_scores = scores.copy()
    pos_classes = classes.copy()

    for i in range(N):
        max_idx = np.argmax(pos_scores[i:]) + i
        max_i = int(max_idx)
        if i != max_i:
            pos_boxes[i], pos_boxes[max_i] = pos_boxes[max_i].copy(), pos_boxes[i].copy()
            pos_scores[i], pos_scores[max_i] = pos_scores[max_i], pos_scores[i]
            pos_classes[i], pos_classes[max_i] = pos_classes[max_i], pos_classes[i]

        max_box = pos_boxes[i]
        max_score = pos_scores[i]
        max_cls = pos_classes[i]

        if max_score < score_threshold:
            break

        kept_boxes.append(max_box)
        kept_scores.append(max_score)
        kept_classes.append(max_cls)

        if i + 1 < N:
            same_cls_mask = pos_classes[i + 1:] == max_cls
            rem_boxes = pos_boxes[i + 1:]
            ious = calculate_iou_vectorized(max_box, rem_boxes)

            # Gaussian decay: s = s * exp(-(iou^2)/sigma)
            decay = np.exp(-(ious ** 2) / sigma)
            
            pos_scores[i + 1:] = np.where(
                same_cls_mask,
                pos_scores[i + 1:] * decay,
                pos_scores[i + 1:]
            )

    if len(kept_boxes) == 0:
        return np.empty((0, 4)), np.array([]), np.array([])

    return np.array(kept_boxes), np.array(kept_scores), np.array(kept_classes)


def linear_soft_nms(boxes, scores, classes=None, iou_threshold=0.45, score_threshold=0.35):
    """
    Linear Soft-NMS:
    Score decay formula: s_i = s_i * (1 - iou(M, b_i)) for iou >= iou_threshold
    """
    if len(boxes) == 0:
        return np.empty((0, 4)), np.array([]), np.array([]) if classes is not None else np.array([])

    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    if classes is None:
        classes = np.zeros(len(boxes), dtype=np.int32)
    else:
        classes = np.array(classes, dtype=np.int32)

    kept_boxes = []
    kept_scores = []
    kept_classes = []

    N = len(boxes)
    pos_boxes = boxes.copy()
    pos_scores = scores.copy()
    pos_classes = classes.copy()

    for i in range(N):
        max_idx = np.argmax(pos_scores[i:]) + i
        max_i = int(max_idx)
        if i != max_i:
            pos_boxes[i], pos_boxes[max_i] = pos_boxes[max_i].copy(), pos_boxes[i].copy()
            pos_scores[i], pos_scores[max_i] = pos_scores[max_i], pos_scores[i]
            pos_classes[i], pos_classes[max_i] = pos_classes[max_i], pos_classes[i]

        max_box = pos_boxes[i]
        max_score = pos_scores[i]
        max_cls = pos_classes[i]

        if max_score < score_threshold:
            break

        kept_boxes.append(max_box)
        kept_scores.append(max_score)
        kept_classes.append(max_cls)

        if i + 1 < N:
            same_cls_mask = pos_classes[i + 1:] == max_cls
            rem_boxes = pos_boxes[i + 1:]
            ious = calculate_iou_vectorized(max_box, rem_boxes)

            # Linear decay: s = s * (1 - iou) for iou >= iou_threshold
            decay = np.where(ious >= iou_threshold, 1.0 - ious, 1.0)
            
            pos_scores[i + 1:] = np.where(
                same_cls_mask,
                pos_scores[i + 1:] * decay,
                pos_scores[i + 1:]
            )

    if len(kept_boxes) == 0:
        return np.empty((0, 4)), np.array([]), np.array([])

    return np.array(kept_boxes), np.array(kept_scores), np.array(kept_classes)


def soft_nms(boxes, scores, classes=None, method="gaussian", sigma=0.5, iou_threshold=0.45, score_threshold=0.35):
    """
    Main Soft-NMS interface.
    method: "gaussian" or "linear"
    Returns: (filtered_boxes, filtered_scores, filtered_classes)
    """
    if method.lower() == "linear":
        return linear_soft_nms(
            boxes, scores, classes,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold
        )
    else:
        return gaussian_soft_nms(
            boxes, scores, classes,
            sigma=sigma,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold
        )
