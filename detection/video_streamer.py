import os
import cv2
import time
import numpy as np
from typing import Any
from config import Config
from detection.video_processor import VideoProcessor


class VideoFileStreamer:
    """
    Live real-time surveillance streamer for uploaded video files.
    Processes video frame-by-frame with YOLOv8 + Soft-NMS + Deep SORT + Optical Flow + Risk Engine.
    Streams frames as MJPEG multipart HTTP response for 100% web browser compatibility.
    """
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.processor = VideoProcessor()
        self.is_running = False
        self.last_telemetry: dict[str, Any] = {
            "people_count": 0,
            "risk_score": 0.0,
            "risk_level": "GREEN",
            "density_score": 0.0,
            "relative_speed": 0.0,
            "motion_entropy": 0.0,
            "motion_intensity": 0.0,
            "num_falls": 0,
            "status_title": "INITIALIZING",
            "summary_message": "Loading surveillance video stream..."
        }
        self.frame_count = 0

    def generate_frames(self, loop=True, session_id=None):
        """Yields JPEG multipart frames for Flask live streaming."""
        if not os.path.exists(self.video_path):
            err_frame = 20 * np.ones((480, 640, 3), dtype=np.uint8)
            cv2.putText(err_frame, "VIDEO FILE NOT FOUND", (160, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            _, jpeg = cv2.imencode(".jpg", err_frame)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
            return

        self.processor.reset()
        self.is_running = True

        while self.is_running:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                break

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            delay = max(0.01, 1.0 / fps)
            frame_idx = 0

            try:
                while self.is_running:
                    ret, frame = cap.read()
                    if not ret:
                        if loop and frame_idx > 0:
                            self.processor.reset()
                            break  # Will restart the outer while loop
                        else:
                            return

                    frame_idx += 1
                    self.frame_count = frame_idx

                    # Scale for optimal speed & high visual quality
                    h, w = frame.shape[:2]
                    target_w = min(w, Config.TARGET_WIDTH)
                    target_h = int(h * (target_w / w)) if w > 0 else Config.TARGET_HEIGHT
                    small_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

                    # Process through AI Pipeline
                    start_p = time.time()
                    annotated_frame, telemetry = self.processor.process_frame(
                        small_frame,
                        frame_idx=frame_idx,
                        session_id=session_id
                    )
                    if isinstance(telemetry, dict) and len(telemetry) > 0:
                        self.last_telemetry = telemetry

                    out_img = annotated_frame if isinstance(annotated_frame, np.ndarray) else small_frame

                    # Encode to JPEG
                    _, buffer = cv2.imencode(".jpg", out_img, [cv2.IMWRITE_JPEG_QUALITY, 82])
                    frame_bytes = buffer.tobytes()

                    try:
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
                    except GeneratorExit:
                        self.is_running = False
                        return

                    # Pace streaming to video's natural framerate
                    proc_time = time.time() - start_p
                    sleep_time = max(0.005, delay - proc_time)
                    time.sleep(sleep_time)

            except GeneratorExit:
                self.is_running = False
                return
            except Exception:
                pass
            finally:
                cap.release()

    def stop(self):
        """Stops the active video streamer."""
        self.is_running = False
