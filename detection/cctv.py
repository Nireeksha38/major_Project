import os
import cv2
import time
import numpy as np
from config import Config
from detection.video_processor import VideoProcessor


class CCTVStreamer:
    """
    RTSP / IP Camera / Demo Camera Stream Handler.
    Processes live stream frames through full detection and risk pipeline.
    """
    def __init__(self, stream_url=None, is_demo=False, demo_video_path=None):
        self.stream_url = stream_url
        self.is_demo = is_demo
        self.demo_video_path = demo_video_path
        self.processor = VideoProcessor()
        self.is_running = False
        self.status = "DISCONNECTED"
        self.last_telemetry = {}

    def start_stream(self, session_id=None):
        """Yields MJPEG multipart frame stream."""
        self.is_running = True
        self.status = "CONNECTED"
        self.processor.reset()
        frame_idx = 0

        # Choose capture source
        if self.is_demo:
            source = self.demo_video_path
            if not source or not os.path.exists(source):
                # Search default demo video in videos folder
                for category in ["normal", "rush", "panic", "abnormal"]:
                    cat_dir = os.path.join(Config.VIDEOS_DIR, category)
                    if os.path.exists(cat_dir):
                        vids = [f for f in os.listdir(cat_dir) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
                        if vids:
                            source = os.path.join(cat_dir, vids[0])
                            break
        else:
            source = self.stream_url

        if not source:
            self.status = "ERROR"
            err_frame = self._render_status_frame("NO STREAM SOURCE OR DEMO VIDEO CONFIGURED", (0, 0, 255))
            _, jpeg = cv2.imencode(".jpg", err_frame)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
            return

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self.status = "ERROR"
            err_frame = self._render_status_frame(f"CONNECTION FAILED: {self.stream_url or 'DEMO'}", (0, 0, 255))
            _, jpeg = cv2.imencode(".jpg", err_frame)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
            return

        try:
            while self.is_running:
                ret, frame = cap.read()

                # If demo video finishes, loop it seamlessly
                if not ret:
                    if self.is_demo:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                        if not ret:
                            break
                    else:
                        break

                frame_idx += 1

                # Resize for processing
                h, w = frame.shape[:2]
                target_w = min(w, Config.TARGET_WIDTH)
                target_h = int(h * (target_w / w)) if w > 0 else Config.TARGET_HEIGHT
                small_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

                # Process through AI pipeline
                annotated_frame, telemetry = self.processor.process_frame(
                    small_frame,
                    frame_idx=frame_idx,
                    session_id=session_id
                )
                self.last_telemetry = telemetry

                # If Demo Mode, add explicit watermarked label
                out_img = annotated_frame if isinstance(annotated_frame, np.ndarray) else small_frame
                if self.is_demo:
                    cv2.rectangle(out_img, (w - 180, 10), (w - 10, 35), (20, 20, 20), -1)
                    cv2.putText(out_img, "DEMO CAMERA", (w - 170, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 2)

                _, buffer = cv2.imencode(".jpg", out_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = buffer.tobytes()

                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
                time.sleep(0.03)

        except Exception as e:
            self.status = "ERROR"
            print(f"[CCTV] Stream exception: {e}")
        finally:
            self.is_running = False
            self.status = "DISCONNECTED"
            cap.release()

    def _render_status_frame(self, text, color):
        frame = np.zeros((480, 640, 3), dtype=np.uint8) + 25
        cv2.putText(frame, text, (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        return frame

    def stop(self):
        self.is_running = False
        self.status = "DISCONNECTED"
