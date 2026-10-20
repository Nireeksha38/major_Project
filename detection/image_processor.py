import os
import cv2
import time
from datetime import datetime, timezone
from config import Config
from detection.detector import YOLOv8Detector
from database.database import db
from models.detection_session import DetectionSession


class ImageProcessor:
    """
    Handles single image processing: YOLOv8 person detection + Soft-NMS filtering.
    Draws bounding boxes and confidence tags.
    Explicitly notifies that motion analysis requires video temporal input.
    """
    def __init__(self):
        self.detector = YOLOv8Detector()

    def process_image(self, image_path, user_id=None):
        """
        Processes single image file.
        Returns:
            result_dict: statistics and output image relative path
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read image file. Corrupted or unsupported format.")

        start_time = time.time()
        filename = os.path.basename(image_path)
        h, w = image.shape[:2]

        # Optimize resolution for fast CPU inference (limit max dimension to 960px)
        max_dim = 960
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = new_h, new_w

        # Run Detection + Soft-NMS + Duplicate suppression
        boxes, scores, classes, detections = self.detector.detect_and_refine(
            image,
            use_soft_nms=True,
            conf_thresh=Config.YOLO_CONFIDENCE
        )

        people_count = len(boxes)
        # Normalized static density (people per estimated visible area)
        density_score = min(1.0, people_count / 50.0)

        # Analyze individual poses for fallen/horizontal orientation
        prone_indices = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            aspect_ratio = bw / bh
            is_near_edge = (x1 <= 5 or y1 <= 5 or x2 >= w - 5 or y2 >= h - 5)
            if aspect_ratio >= 1.45 and not is_near_edge:
                prone_indices.append(i)

        fallen_count = len(prone_indices)

        if fallen_count > 0:
            condition_level = "RED"
            condition_title = f"CRITICAL - {fallen_count} FALLEN PERSON(S) DETECTED"
            condition_desc = "Emergency Alert: Fallen person hazard detected on the floor. High risk of crowd trampling!"
            default_box_color = (0, 0, 255)  # Bright Red
        elif density_score < 0.50:
            condition_level = "GREEN"
            condition_title = "SAFE - NORMAL DENSITY"
            condition_desc = "Normal crowd flow with clear personal spacing. No immediate stampede hazard."
            default_box_color = (0, 220, 0)  # Bright Green
        elif density_score < 0.80:
            condition_level = "YELLOW"
            condition_title = "WARNING - MODERATE DENSITY"
            condition_desc = "Crowd congestion building up. Recommend monitoring choke points and exit routes."
            default_box_color = (0, 215, 255)  # Bright Yellow/Gold
        else:
            condition_level = "RED"
            condition_title = "CRITICAL - SEVERE OVERCROWDING"
            condition_desc = "Dangerous crowd compression detected! Immediate crowd dispersal or diversion required."
            default_box_color = (0, 50, 255)  # Bright Red

        # Draw Bounding Boxes on Image
        annotated = image.copy()
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            score = scores[i]
            is_prone = (i in prone_indices)

            current_box_color = (0, 0, 255) if is_prone else default_box_color
            current_thickness = 3 if is_prone else 2

            # High-visibility bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), current_box_color, current_thickness)

            # Label background
            label = f"[! FALLEN PERSON] {int(score * 100)}%" if is_prone else f"Person {int(score * 100)}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), current_box_color, -1)
            text_color = (255, 255, 255) if is_prone else (0, 0, 0)
            cv2.putText(annotated, label, (x1 + 3, max(10, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)

        # Static analysis banner overlay at top
        banner_h = 60
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.88, annotated, 0.12, 0, annotated)

        cv2.putText(
            annotated,
            f"PEOPLE: {people_count} | FALLEN: {fallen_count} | DENSITY: {density_score:.2f} | STATUS: {condition_title[:35]}",
            (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            annotated,
            f"SOFT-NMS: ENABLED | {condition_desc[:70]}...",
            (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255) if fallen_count > 0 else default_box_color, 1, cv2.LINE_AA
        )

        # Save Processed Image
        output_filename = f"processed_{filename}"
        output_folder = Config.UPLOAD_FOLDER_IMAGES
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])

        processing_time = round(time.time() - start_time, 3)
        duration = processing_time

        # Calculate relative path for web serving
        rel_output_path = os.path.relpath(output_path, os.path.abspath(os.path.join(Config.BASE_DIR, "static"))).replace("\\", "/")

        result_dict = {
            "people_count": people_count,
            "fallen_count": fallen_count,
            "has_fallen": fallen_count > 0,
            "density_score": round(density_score, 2),
            "condition_level": condition_level,
            "condition_title": condition_title,
            "condition_desc": condition_desc,
            "processing_time": processing_time,
            "output_image": rel_output_path
        }
        
        now_dt = datetime.now(timezone.utc).replace(tzinfo=None)

        # Save Session to DB
        session_id = None
        if user_id is not None:
            try:
                session = DetectionSession(
                    user_id=user_id,
                    source_type="image",
                    source_name=filename,
                    start_time=now_dt,
                    end_time=now_dt,
                    maximum_people=people_count,
                    maximum_risk=condition_level,
                    max_risk_score=round(density_score * 0.5, 3),
                    duration=duration,
                    processed_video_path=os.path.join("uploads", "images", output_filename).replace("\\", "/"),
                    total_frames=1,
                    status="completed"
                )
                db.session.add(session)
                db.session.commit()
                session_id = session.id
            except Exception:
                try:
                    db.session.rollback()
                except Exception:
                    pass

        return {
            "session_id": session_id,
            "filename": filename,
            "processed_image_url": f"uploads/images/{output_filename}",
            "people_count": people_count,
            "density_score": round(density_score, 2),
            "condition_level": condition_level,
            "condition_title": condition_title,
            "condition_desc": condition_desc,
            "processing_time": round(duration, 3),
            "boxes": [[int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(scores[idx])] for idx, b in enumerate(boxes)],
            "motion_notice": "Temporal metrics (Optical Flow, Speed, Entropy, Falls) require video stream input."
        }
