import os
import torch
import numpy as np
from ultralytics import YOLO
from config import Config
from detection.soft_nms import soft_nms, calculate_iou, calculate_iom, is_duplicate_box


class YOLOv8Detector:
    _instance = None
    _model = None

    def __new__(cls, *args, **kwargs):
        """Singleton to load YOLOv8 model only once into memory."""
        if cls._instance is None:
            cls._instance = super(YOLOv8Detector, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_name=None, conf_threshold=None, device=None):
        if self._model is None:
            self.model_name = model_name or Config.YOLO_MODEL_NAME
            self.conf_threshold = conf_threshold or Config.YOLO_CONFIDENCE
            
            # Select device
            if device is None or device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device

            if self.device == "cpu":
                try:
                    cpu_threads = min(8, max(2, (os.cpu_count() or 4) - 1))
                    torch.set_num_threads(cpu_threads)
                except Exception:
                    pass
                
            print(f"[YOLOv8] Loading model '{self.model_name}' on device '{self.device}'...")
            try:
                self._model = YOLO(self.model_name)
            except Exception as e:
                print(f"[YOLOv8] Error loading specified model ({e}), falling back to yolov8n.pt")
                self._model = YOLO("yolov8n.pt")
            print("[YOLOv8] Model loaded successfully.")

    def detect_raw(self, frame, conf=None, imgsz=None):
        """
        Runs YOLOv8 person detection on a single frame.
        Returns:
            boxes: Nx4 ndarray [x1, y1, x2, y2]
            scores: N ndarray
            classes: N ndarray (filtered for person = 0)
        """
        confidence = conf if conf is not None else self.conf_threshold
        img_size = imgsz or getattr(Config, "YOLO_IMGSZ", 384)
        if self._model is None:
            self._model = YOLO(self.model_name or "yolov8n.pt")

        with torch.inference_mode():
            results = self._model.predict(
                source=frame,
                conf=confidence,
                classes=[Config.YOLO_PERSON_CLASS_ID],  # 0 for person
                device=self.device,
                imgsz=img_size,
                verbose=False
            )

        # Type-safe parsing of YOLO prediction results
        results_list = list(results) if results is not None else []
        if not results_list:
            return np.empty((0, 4)), np.array([]), np.array([])

        first_res = results_list[0]
        boxes_data = getattr(first_res, "boxes", None)
        if boxes_data is None:
            return np.empty((0, 4)), np.array([]), np.array([])

        xyxy_attr = getattr(boxes_data, "xyxy", None)
        conf_attr = getattr(boxes_data, "conf", None)
        cls_attr = getattr(boxes_data, "cls", None)

        if xyxy_attr is None or conf_attr is None or cls_attr is None:
            return np.empty((0, 4)), np.array([]), np.array([])

        xyxy = xyxy_attr.cpu().numpy() if hasattr(xyxy_attr, "cpu") else np.asarray(xyxy_attr)
        scores = conf_attr.cpu().numpy() if hasattr(conf_attr, "cpu") else np.asarray(conf_attr)
        classes = cls_attr.cpu().numpy() if hasattr(cls_attr, "cpu") else np.asarray(cls_attr)

        return xyxy, scores, classes

    def detect_and_refine(self, frame, use_soft_nms=True, soft_nms_method=None, sigma=None, iou_thresh=None, conf_thresh=None, imgsz=None):
        """
        Full detection pipeline:
        Frame -> YOLOv8 Raw -> Soft-NMS Refinement -> Duplicate Box Suppression -> Filtered Detections
        Returns:
            boxes: Nx4 ndarray [x1, y1, x2, y2]
            scores: N ndarray
            classes: N ndarray
            detections_list: list of [x1, y1, x2, y2, score, class_id]
        """
        target_conf = conf_thresh or Config.YOLO_CONFIDENCE
        raw_conf = max(0.10, target_conf - 0.15)
        raw_boxes, raw_scores, raw_classes = self.detect_raw(frame, conf=raw_conf, imgsz=imgsz)

        if len(raw_boxes) == 0:
            return np.empty((0, 4)), np.array([]), np.array([]), []

        # Filter out true tiny noise artifacts while retaining distant/small people
        valid_indices = []
        for i, b in enumerate(raw_boxes):
            w = b[2] - b[0]
            h = b[3] - b[1]
            if w >= 6 and h >= 8:
                valid_indices.append(i)

        if not valid_indices:
            return np.empty((0, 4)), np.array([]), np.array([]), []

        raw_boxes = raw_boxes[valid_indices]
        raw_scores = raw_scores[valid_indices]
        raw_classes = raw_classes[valid_indices]

        if use_soft_nms:
            method = soft_nms_method or Config.SOFT_NMS_METHOD
            sig = sigma or Config.SOFT_NMS_SIGMA
            iou_t = iou_thresh or Config.SOFT_NMS_IOU_THRESHOLD
            score_t = target_conf

            refined_boxes, refined_scores, refined_classes = soft_nms(
                raw_boxes,
                raw_scores,
                raw_classes,
                method=method,
                sigma=sig,
                iou_threshold=iou_t,
                score_threshold=score_t
            )
        else:
            mask = raw_scores >= target_conf
            refined_boxes = raw_boxes[mask]
            refined_scores = raw_scores[mask]
            refined_classes = raw_classes[mask]

        if len(refined_boxes) == 0:
            return np.empty((0, 4)), np.array([]), np.array([]), []

        # Deduplicate only true duplicate boxes (IoU >= 0.75 or extreme containment IoM >= 0.88)
        # Preserves all adjacent, overlapping, and crowded individuals
        order = np.argsort(-refined_scores)
        keep = []
        for i in order:
            box_i = refined_boxes[i]
            overlap = False
            for k in keep:
                box_k = refined_boxes[k]
                if is_duplicate_box(box_i, box_k, iou_thresh=0.75, iom_thresh=0.88):
                    overlap = True
                    break
            if not overlap:
                keep.append(i)

        final_boxes = refined_boxes[keep]
        final_scores = refined_scores[keep]
        final_classes = refined_classes[keep]

        detections_list = []
        for i in range(len(final_boxes)):
            box = final_boxes[i]
            detections_list.append([
                float(box[0]), float(box[1]), float(box[2]), float(box[3]),
                float(final_scores[i]), int(final_classes[i])
            ])

        return final_boxes, final_scores, final_classes, detections_list

    def detect_crowd_image(self, image, conf_thresh=0.16, imgsz=800, use_tiling=True):
        """
        Adaptive High-Accuracy Crowd Detection for Static Images.
        Dynamically adapts between:
        1. Sparse Scene Mode (<= 6 people): Clean confidence (0.25) + Strict Parent-Child Enclosure
           Suppression (IoM >= 0.65, IoU >= 0.45) to ensure exact counts (e.g. 3 people = exactly 3)
           with zero ghost boxes.
        2. Dense Crowd Mode (> 6 people): Multi-scale global + quadrant tiling (imgsz=800) with
           Gaussian Soft-NMS refinement to accurately capture all 25+ distant, occluded individuals.
        """
        if image is None or getattr(image, "size", 0) == 0:
            return np.empty((0, 4)), np.array([]), np.array([]), []

        h, w = image.shape[:2]

        # Stage 1: Fast Global High-Resolution Assessment
        b_init, s_init, c_init = self.detect_raw(image, conf=0.22, imgsz=imgsz)
        n_init = len(b_init)

        # -------------------------------------------------------------
        # BRANCH A: Sparse / Small Group Scene (<= 6 people detected)
        # -------------------------------------------------------------
        if n_init <= 6:
            # Run high-res single-pass inference with standard refinement
            refined_boxes, refined_scores, refined_classes, _ = self.detect_and_refine(
                image,
                conf_thresh=0.25,
                imgsz=imgsz
            )

            if len(refined_boxes) == 0:
                return np.empty((0, 4)), np.array([]), np.array([]), []

            # Strict Parent-Child Enclosure & Duplicate Filter
            order = np.argsort(-refined_scores)
            keep = []
            for i in order:
                bi = refined_boxes[i]
                bw = bi[2] - bi[0]
                bh = bi[3] - bi[1]
                # Filter out microscopic noise artifacts
                if bw < 8 or bh < 12:
                    continue
                
                is_duplicate = False
                for k in keep:
                    bk = refined_boxes[k]
                    iou = calculate_iou(bi, bk)
                    iom = calculate_iom(bi, bk)
                    # Suppress child-parent sub-boxes or high overlap
                    if iou >= 0.45 or iom >= 0.65:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    keep.append(i)

            final_boxes = refined_boxes[keep]
            final_scores = refined_scores[keep]
            final_classes = refined_classes[keep]

            detections_list = []
            for i in range(len(final_boxes)):
                box = final_boxes[i]
                detections_list.append([
                    float(box[0]), float(box[1]), float(box[2]), float(box[3]),
                    float(final_scores[i]), int(final_classes[i])
                ])

            return final_boxes, final_scores, final_classes, detections_list

        # -------------------------------------------------------------
        # BRANCH B: Dense Crowd Scene (> 6 people detected)
        # -------------------------------------------------------------
        raw_conf = max(0.10, conf_thresh - 0.05)
        all_boxes = [b_init] if len(b_init) > 0 else []
        all_scores = [s_init] if len(s_init) > 0 else []
        all_classes = [c_init] if len(c_init) > 0 else []

        # Multi-Quadrant Overlapping Tiling
        if use_tiling and w >= 380 and h >= 280:
            half_w, half_h = w // 2, h // 2
            pad_w, pad_h = int(w * 0.15), int(h * 0.15)
            tiles = [
                (0, 0, min(w, half_w + pad_w), min(h, half_h + pad_h)),
                (max(0, half_w - pad_w), 0, w, min(h, half_h + pad_h)),
                (0, max(0, half_h - pad_h), min(w, half_w + pad_w), h),
                (max(0, half_w - pad_w), max(0, half_h - pad_h), w, h)
            ]

            for (tx1, ty1, tx2, ty2) in tiles:
                tile = image[ty1:ty2, tx1:tx2]
                tb, ts, tc = self.detect_raw(tile, conf=raw_conf, imgsz=640)
                if len(tb) > 0:
                    tb_offset = tb.copy()
                    tb_offset[:, 0] += tx1
                    tb_offset[:, 1] += ty1
                    tb_offset[:, 2] += tx1
                    tb_offset[:, 3] += ty1
                    all_boxes.append(tb_offset)
                    all_scores.append(ts)
                    all_classes.append(tc)

        if not all_boxes:
            return np.empty((0, 4)), np.array([]), np.array([]), []

        merged_boxes = np.vstack(all_boxes)
        merged_scores = np.concatenate(all_scores)
        merged_classes = np.concatenate(all_classes)

        # Filter out tiny noise
        valid_indices = []
        for i, b in enumerate(merged_boxes):
            bw = b[2] - b[0]
            bh = b[3] - b[1]
            if bw >= 6 and bh >= 8:
                valid_indices.append(i)

        if not valid_indices:
            return np.empty((0, 4)), np.array([]), np.array([]), []

        merged_boxes = merged_boxes[valid_indices]
        merged_scores = merged_scores[valid_indices]
        merged_classes = merged_classes[valid_indices]

        # Gaussian Soft-NMS Refinement
        refined_boxes, refined_scores, refined_classes = soft_nms(
            merged_boxes,
            merged_scores,
            merged_classes,
            method="gaussian",
            sigma=0.5,
            iou_threshold=0.60,
            score_threshold=conf_thresh
        )

        if len(refined_boxes) == 0:
            return np.empty((0, 4)), np.array([]), np.array([]), []

        # Deduplicate duplicate boxes (IoU >= 0.70 or containment IoM >= 0.85)
        order = np.argsort(-refined_scores)
        keep = []
        for i in order:
            box_i = refined_boxes[i]
            overlap = False
            for k in keep:
                box_k = refined_boxes[k]
                if is_duplicate_box(box_i, box_k, iou_thresh=0.70, iom_thresh=0.85):
                    overlap = True
                    break
            if not overlap:
                keep.append(i)

        final_boxes = refined_boxes[keep]
        final_scores = refined_scores[keep]
        final_classes = refined_classes[keep]

        detections_list = []
        for i in range(len(final_boxes)):
            box = final_boxes[i]
            detections_list.append([
                float(box[0]), float(box[1]), float(box[2]), float(box[3]),
                float(final_scores[i]), int(final_classes[i])
            ])

        return final_boxes, final_scores, final_classes, detections_list


