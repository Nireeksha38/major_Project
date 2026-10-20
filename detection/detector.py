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
        img_size = imgsz or getattr(Config, "YOLO_IMGSZ", 480)
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

    def detect_and_refine(self, frame, use_soft_nms=True, soft_nms_method=None, sigma=None, iou_thresh=None, conf_thresh=None):
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
        raw_conf = max(0.20, target_conf - 0.05)
        raw_boxes, raw_scores, raw_classes = self.detect_raw(frame, conf=raw_conf)

        if len(raw_boxes) == 0:
            return np.empty((0, 4)), np.array([]), np.array([]), []

        # Filter out tiny noise boxes
        valid_indices = []
        for i, b in enumerate(raw_boxes):
            w = b[2] - b[0]
            h = b[3] - b[1]
            if w >= 20 and h >= 25:
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

        # Deduplicate overlapping & nested boxes for exact single-person accuracy
        order = np.argsort(-refined_scores)
        keep = []
        for i in order:
            box_i = refined_boxes[i]
            overlap = False
            for k in keep:
                box_k = refined_boxes[k]
                if is_duplicate_box(box_i, box_k, iou_thresh=0.35, iom_thresh=0.50):
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
