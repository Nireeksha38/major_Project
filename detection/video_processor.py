import os
import time
import cv2
import numpy as np
from datetime import datetime, timezone
from config import Config
from database.database import db
from models.detection_session import DetectionSession
from models.crowd_data import CrowdData
from detection.detector import YOLOv8Detector
from tracking.deep_sort_tracker import DeepSORTTracker
from motion_analysis.optical_flow import OpticalFlowAnalyzer
from motion_analysis.entropy import MotionEntropyAnalyzer
from motion_analysis.crowd_speed import CrowdSpeedAnalyzer
from motion_analysis.panic_detector import PanicDetector
from behavior.fall_detector import FallDetector
from risk_engine.risk_classifier import RiskClassifier
from risk_engine.alert_logic import AlertManager


def _as_int(val, default: int = 0) -> int:
    if isinstance(val, (int, float)):
        return int(val)
    return default


def _as_float(val, default: float = 0.0) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    return default


def _as_str(val, default: str = "") -> str:
    if isinstance(val, str):
        return val
    return default


class VideoProcessor:
    """
    Main Computer Vision Video Processing Pipeline.
    Executes:
    Frame -> YOLOv8 -> Soft-NMS -> Deep SORT -> Optical Flow -> Entropy -> Speed -> Fall/Panic -> Risk -> Alerts -> HUD Rendering.
    """
    def __init__(self, frame_skip=None):
        self.frame_skip = frame_skip if frame_skip is not None else Config.FRAME_SKIP
        self.detector = YOLOv8Detector()
        self.tracker = DeepSORTTracker()
        self.flow_analyzer = OpticalFlowAnalyzer()
        self.entropy_analyzer = MotionEntropyAnalyzer()
        self.speed_analyzer = CrowdSpeedAnalyzer()
        self.panic_detector = PanicDetector()
        self.fall_detector = FallDetector()
        self.risk_classifier = RiskClassifier()
        self.alert_manager = AlertManager()
        self.people_count_history = []

    def reset(self):
        """Resets state for a new video stream."""
        self.tracker.reset()
        self.flow_analyzer.reset()
        self.panic_detector.reset()
        self.fall_detector.reset()
        self.alert_manager.reset()
        self.people_count_history = []

    def process_frame(self, frame, frame_idx=0, session_id=None):
        """
        Processes a single video frame through the full AI/ML pipeline.
        Returns:
            annotated_frame: Frame with HUD overlay
            telemetry: Dictionary containing real-time metrics
        """
        if frame is None:
            return None, {}

        orig_h, orig_w = frame.shape[:2]

        # 1. Detection + Soft-NMS Refinement
        refined_boxes, refined_scores, refined_classes, detections = self.detector.detect_and_refine(
            frame,
            use_soft_nms=True
        )

        # 2. Deep SORT Tracking
        active_tracks = self.tracker.update(detections)
        raw_count = len(active_tracks) if len(active_tracks) > 0 else len(refined_boxes)
        
        # Temporal smoothing: use rolling median over recent 5 frames to eliminate flickering
        self.people_count_history.append(raw_count)
        if len(self.people_count_history) > 5:
            self.people_count_history.pop(0)
        people_count = int(np.round(np.median(self.people_count_history)))

        # 3. Crowd Density (normalized to expected capacity)
        density_score = min(1.0, people_count / 40.0)

        # 4. Optical Flow & Motion Intensity
        flow, magnitude, angle_deg, motion_intensity, flow_summary = self.flow_analyzer.compute_flow(frame)

        # 5. Directional Motion Entropy
        motion_entropy, _ = self.entropy_analyzer.compute_entropy(angle_deg, magnitude)

        # 6. Crowd Speed
        relative_speed, avg_px_speed, max_px_speed, accel = self.speed_analyzer.compute_speed(active_tracks)

        # 7. Panic / Rush Detection (Temporal Window)
        panic_score, is_rush, is_panic, panic_event, panic_reason = self.panic_detector.update(
            people_count=people_count,
            relative_speed=relative_speed,
            motion_intensity=motion_intensity,
            motion_entropy=motion_entropy,
            density_score=density_score
        )

        # 8. Fall Detection
        fall_detected, num_falls, fall_details, fall_score = self.fall_detector.update(
            active_tracks,
            frame_width=orig_w,
            frame_height=orig_h
        )

        # 9. Risk Classification
        risk_eval = self.risk_classifier.evaluate(
            people_count=people_count,
            density_score=density_score,
            relative_speed=relative_speed,
            motion_intensity=motion_intensity,
            motion_entropy=motion_entropy,
            panic_score=panic_score,
            fall_detected=fall_detected,
            num_falls=num_falls,
            fall_score=fall_score,
            is_rush=is_rush,
            is_panic=is_panic
        )

        # 10. Alert Management & Cooldown Triggering
        panic_info = {"is_rush": is_rush, "is_panic": is_panic, "event": panic_event, "reason": panic_reason}
        fall_info = {"fall_detected": fall_detected, "num_falls": num_falls, "details": fall_details}
        new_alerts = self.alert_manager.check_and_trigger(session_id, risk_eval, panic_info, fall_info, frame_idx)

        # 11. Render Professional HUD Overlay
        annotated_frame = self.render_hud(
            frame,
            active_tracks,
            refined_boxes,
            flow,
            risk_eval,
            people_count,
            relative_speed,
            motion_entropy,
            fall_details,
            panic_info
        )

        telemetry = {
            "frame": frame_idx,
            "people_count": people_count,
            "density_score": round(density_score, 2),
            "relative_speed": round(relative_speed, 2),
            "motion_intensity": round(motion_intensity, 2),
            "motion_entropy": round(motion_entropy, 2),
            "panic_score": round(panic_score, 2),
            "risk_score": risk_eval["risk_score"],
            "risk_level": risk_eval["risk_level"],
            "status_title": risk_eval["status_title"],
            "summary_message": risk_eval["summary_message"],
            "fall_detected": fall_detected,
            "num_falls": num_falls,
            "is_rush": is_rush,
            "is_panic": is_panic,
            "dominant_factor": risk_eval["dominant_factor"],
            "new_alerts": new_alerts
        }

        return annotated_frame, telemetry

    def render_hud(self, frame, active_tracks, refined_boxes, flow, risk_eval, people_count, relative_speed, motion_entropy, fall_details, panic_info):
        """Draws bounding boxes, tracking trajectories, and dashboard HUD on the video frame."""
        vis = frame.copy()
        h, w = vis.shape[:2]

        risk_level = risk_eval["risk_level"]
        # Colors: Green (Safe), Yellow (Warning), Red (Critical)
        if risk_level == "GREEN":
            theme_color = (0, 220, 0)      # BGR Green
            badge_bg = (20, 100, 20)
        elif risk_level == "YELLOW":
            theme_color = (0, 215, 255)    # BGR Yellow / Amber
            badge_bg = (20, 90, 120)
        else:
            theme_color = (30, 30, 255)    # BGR Red
            badge_bg = (20, 20, 140)

        # 1. Draw Tracking Boxes & Trajectories
        fall_track_ids = {f["track_id"] for f in fall_details}

        if active_tracks:
            for track in active_tracks:
                tid = track["id"]
                x1, y1, x2, y2 = map(int, track["bbox"])
                is_fallen = tid in fall_track_ids
                track_speed = track.get("speed", 0.0)

                # Individual-level behavior color coding:
                # RED = Fallen / Prone Anomaly
                # YELLOW = Individual Rushing / Sprinting
                # GREEN = Normal Walking Person
                if is_fallen:
                    box_color = (0, 0, 255)  # Bright Red for fallen / abnormal
                    thickness = 3
                    lbl = f"ID:{tid} [! FALL DETECTED]"
                    text_color = (255, 255, 255)
                elif track_speed >= 18.0:
                    box_color = (0, 215, 255)  # Amber Yellow for rushing person
                    thickness = 2
                    lbl = f"ID:{tid} [RUSH]"
                    text_color = (0, 0, 0)
                else:
                    box_color = (0, 220, 0)  # Bright Green for normal person
                    thickness = 2
                    lbl = f"ID:{tid}"
                    text_color = (0, 0, 0)

                cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, thickness)

                # Draw track ID and state banner
                (lbl_w, lbl_h), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(vis, (x1, max(0, y1 - lbl_h - 8)), (x1 + lbl_w + 8, y1), box_color, -1)
                cv2.putText(vis, lbl, (x1 + 4, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)

                # Draw trajectory trail
                traj = track.get("trajectory", [])
                if len(traj) > 1:
                    pts = np.array(traj, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(vis, [pts], isClosed=False, color=(255, 200, 0), thickness=1)
        else:
            # Fallback to detection boxes if tracker initializing
            for box in refined_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 0), 2)

        # 2. Draw Top HUD Dashboard Banner
        banner_h = 55
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.85, vis, 0.15, 0, vis)

        # Risk Badge
        cv2.rectangle(vis, (10, 10), (135, banner_h - 10), badge_bg, -1)
        cv2.rectangle(vis, (10, 10), (135, banner_h - 10), theme_color, 2)
        cv2.putText(vis, f"RISK: {risk_level}", (18, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, theme_color, 2, cv2.LINE_AA)

        # Metrics display on top bar
        info_text = f"PEOPLE: {people_count}  |  SPEED: {relative_speed:.2f}  |  ENTROPY: {motion_entropy:.2f}  |  SCORE: {risk_eval['risk_score']:.2f}"
        cv2.putText(vis, info_text, (150, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Draw Bottom Alert / Event Status Line
        bot_banner_h = 32
        bot_overlay = vis.copy()
        cv2.rectangle(bot_overlay, (0, h - bot_banner_h), (w, h), (15, 23, 42), -1)
        cv2.addWeighted(bot_overlay, 0.85, vis, 0.15, 0, vis)

        status_msg = f"STATUS: {risk_eval['status_title']} - {risk_eval['summary_message']}"
        cv2.putText(vis, status_msg, (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, theme_color, 1, cv2.LINE_AA)

        return vis

    def quick_init_video(self, input_video_path, user_id=None):
        """
        Instant initialization of uploaded video (< 0.05 sec).
        Reads video metadata, registers session in DB, and returns immediately
        so the live AI surveillance stream begins playing without delay.
        """
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Input video not found: {input_video_path}")

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = round(total_frames / max(1.0, fps), 2) if total_frames > 0 else 0.0

        filename = os.path.basename(input_video_path)
        base_name = os.path.splitext(filename)[0]

        preview_folder = os.path.join(Config.UPLOAD_FOLDER_FRAMES, f"prev_{base_name}")
        os.makedirs(preview_folder, exist_ok=True)

        # Grab first frame thumbnail
        preview_frames = []
        peak_frame_rel = None
        ret, frame = cap.read()
        if ret and frame is not None:
            target_w = min(width, Config.TARGET_WIDTH)
            target_h = int(height * (target_w / width)) if width > 0 else Config.TARGET_HEIGHT
            small_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            prev_file = "f_00001.jpg"
            prev_path = os.path.join(preview_folder, prev_file)
            cv2.imwrite(prev_path, small_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            rel_prev = os.path.relpath(prev_path, os.path.abspath(os.path.join(Config.BASE_DIR, "static"))).replace("\\", "/")
            preview_frames.append(rel_prev)
            peak_frame_rel = rel_prev
        cap.release()

        # Initialize session in DB
        session_id = None
        start_dt = datetime.now(timezone.utc).replace(tzinfo=None)
        try:
            session = DetectionSession(
                user_id=user_id,
                source_type="video",
                source_name=filename,
                start_time=start_dt,
                maximum_people=0,
                maximum_risk="GREEN",
                max_risk_score=0.0,
                status="running",
                total_frames=total_frames,
                duration=duration_sec
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
            "source_name": filename,
            "total_frames": total_frames,
            "duration": duration_sec,
            "max_people": 0,
            "peak_risk_level": "GREEN",
            "max_risk_score": 0.0,
            "processed_video_path": None,
            "preview_frames": preview_frames,
            "peak_frame": peak_frame_rel,
            "fps_processed": round(fps, 1)
        }

    def process_video_file(self, input_video_path, output_video_path=None, user_id=None, progress_callback=None):
        """
        Fast video processor for uploaded surveillance video files.
        Extracts keyframe samples (~25-30 frames) for interactive scrubber and summary metrics.
        Returns:
            session_summary: dict of entire run statistics
        """
        self.reset()
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Input video not found: {input_video_path}")

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = round(total_frames / max(1.0, fps), 2) if total_frames > 0 else 0.0

        # Target dimensions for performance
        target_w = min(width, Config.TARGET_WIDTH)
        target_h = int(height * (target_w / width)) if width > 0 else Config.TARGET_HEIGHT

        filename = os.path.basename(input_video_path)
        base_name = os.path.splitext(filename)[0]

        # Frame preview directory for browser scrubber
        preview_folder = os.path.join(Config.UPLOAD_FOLDER_FRAMES, f"prev_{base_name}")
        os.makedirs(preview_folder, exist_ok=True)

        # Initialize Session in DB
        session_id = None
        start_dt = datetime.now(timezone.utc).replace(tzinfo=None)
        try:
            session = DetectionSession(
                user_id=user_id,
                source_type="video",
                source_name=filename,
                start_time=start_dt,
                maximum_people=0,
                maximum_risk="GREEN",
                max_risk_score=0.0,
                status="running"
            )
            db.session.add(session)
            db.session.commit()
            session_id = session.id
        except Exception:
            db.session.rollback()

        max_people = 0
        max_risk_score = 0.0
        peak_risk_level = "GREEN"
        peak_frame_img = None
        peak_frame_rel = None
        preview_frames = []
        batch_crowd_records = []

        start_time = time.time()

        # Generate sample frame indices evenly distributed (25 to 35 keyframes)
        if total_frames > 0:
            num_samples = min(total_frames, 30)
            sample_indices = sorted(list(set(int(idx) for idx in np.linspace(0, total_frames - 1, num_samples))))
        else:
            sample_indices = [0]

        try:
            for s_idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, s_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                frame_idx = s_idx + 1

                # Resize
                if frame.shape[1] != target_w:
                    frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

                annotated_frame, telemetry = self.process_frame(frame, frame_idx=frame_idx, session_id=session_id)

                cur_people = _as_int(telemetry.get("people_count", 0))
                cur_risk = _as_float(telemetry.get("risk_score", 0.0))
                cur_risk_level = _as_str(telemetry.get("risk_level", "GREEN"), "GREEN")

                if cur_people > max_people:
                    max_people = cur_people
                if cur_risk >= max_risk_score:
                    max_risk_score = cur_risk
                    peak_risk_level = cur_risk_level
                    if annotated_frame is not None:
                        peak_frame_img = annotated_frame.copy()

                if annotated_frame is not None:
                    prev_file = f"f_{frame_idx:05d}.jpg"
                    prev_path = os.path.join(preview_folder, prev_file)
                    cv2.imwrite(prev_path, annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    rel_prev = os.path.relpath(prev_path, os.path.abspath(os.path.join(Config.BASE_DIR, "static"))).replace("\\", "/")
                    preview_frames.append(rel_prev)

                if session_id is not None:
                    crowd_rec = CrowdData(
                        session_id=session_id,
                        crowd_count=cur_people,
                        density=_as_float(telemetry.get("density_score", 0.0)),
                        relative_speed=_as_float(telemetry.get("relative_speed", 0.0)),
                        motion_entropy=_as_float(telemetry.get("motion_entropy", 0.0)),
                        motion_intensity=_as_float(telemetry.get("motion_intensity", 0.0)),
                        panic_score=_as_float(telemetry.get("panic_score", 0.0)),
                        risk_score=cur_risk,
                        risk_level=cur_risk_level,
                        falls_detected=_as_int(telemetry.get("num_falls", 0)),
                        frame_number=frame_idx
                    )
                    batch_crowd_records.append(crowd_rec)

                if progress_callback:
                    progress_callback(frame_idx, total_frames, telemetry)

        finally:
            cap.release()

            if batch_crowd_records:
                try:
                    db.session.add_all(batch_crowd_records)
                    db.session.commit()
                except Exception:
                    try:
                        db.session.rollback()
                    except Exception:
                        pass

        duration = time.time() - start_time
        end_dt = datetime.now(timezone.utc).replace(tzinfo=None)

        # Save peak frame if available
        if peak_frame_img is not None:
            peak_path = os.path.join(preview_folder, "peak_risk_moment.jpg")
            cv2.imwrite(peak_path, peak_frame_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            peak_frame_rel = os.path.relpath(peak_path, os.path.abspath(os.path.join(Config.BASE_DIR, "static"))).replace("\\", "/")

        if session_id is not None:
            try:
                db_session = db.session.get(DetectionSession, session_id)
                if db_session:
                    db_session.end_time = end_dt
                    db_session.maximum_people = max_people
                    db_session.maximum_risk = peak_risk_level
                    db_session.max_risk_score = round(max_risk_score, 3)
                    db_session.duration = duration_sec or round(duration, 2)
                    db_session.total_frames = total_frames or len(preview_frames)
                    db_session.status = "completed"
                db.session.commit()
            except Exception:
                try:
                    db.session.rollback()
                except Exception:
                    pass

        return {
            "session_id": session_id,
            "source_name": filename,
            "total_frames": total_frames or len(preview_frames),
            "duration": duration_sec or round(duration, 2),
            "max_people": max_people,
            "peak_risk_level": peak_risk_level,
            "max_risk_score": round(max_risk_score, 3),
            "processed_video_path": None,
            "preview_frames": preview_frames,
            "peak_frame": peak_frame_rel,
            "fps_processed": round(len(sample_indices) / max(0.1, duration), 1)
        }
