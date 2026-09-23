import os
import cv2
import time
import threading
import numpy as np
from config import Config
from detection.video_processor import VideoProcessor


class ThreadedRTSPCapture:
    """
    Threaded RTSP / Network stream reader.
    Continuously discards stale frames from the network queue so the AI detection
    is always fed the latest real-time frame with 0 ms buffer lag.
    """
    def __init__(self, src):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;102400|max_delay;500000"
        self.cap = cv2.VideoCapture(src)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self.grabbed, self.frame = self.cap.read()
        self.is_running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.is_running and self.cap.isOpened():
            try:
                grabbed, frame = self.cap.read()
                if grabbed and frame is not None:
                    with self.lock:
                        self.grabbed = grabbed
                        self.frame = frame
                else:
                    time.sleep(0.01)
            except Exception:
                break

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.grabbed, self.frame.copy()
            return self.grabbed, None

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.is_running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=0.4)
        try:
            self.cap.release()
        except Exception:
            pass


class CCTVStreamer:
    """
    RTSP / IP Camera / Demo Camera Stream Handler.
    Processes live stream frames through full detection and risk pipeline with zero network buffer lag.
    """
    def __init__(self, stream_url=None, is_demo=False, demo_video_path=None):
        self.stream_url = stream_url
        self.is_demo = is_demo
        self.demo_video_path = demo_video_path
        self.processor = VideoProcessor()
        self.is_running = False
        self.status = "DISCONNECTED"
        self.last_telemetry = {}
        self._threaded_cap = None

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

        # For RTSP live stream, use ThreadedRTSPCapture to avoid network backlog
        # For Demo local video file, use cv2.VideoCapture with smooth loop
        if not self.is_demo:
            self._threaded_cap = ThreadedRTSPCapture(source)
            if not self._threaded_cap.isOpened():
                self.status = "ERROR"
                err_frame = self._render_status_frame(f"CONNECTION FAILED: {self.stream_url or 'CCTV'}", (0, 0, 255))
                _, jpeg = cv2.imencode(".jpg", err_frame)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
                return
            cap = self._threaded_cap
        else:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                self.status = "ERROR"
                err_frame = self._render_status_frame(f"DEMO VIDEO FAILED TO OPEN", (0, 0, 255))
                _, jpeg = cv2.imencode(".jpg", err_frame)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
                return

        try:
            while self.is_running:
                ret, frame = cap.read()

                # If demo video finishes, loop it seamlessly
                if not ret or frame is None:
                    if self.is_demo:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            time.sleep(0.02)
                            continue
                    else:
                        time.sleep(0.01)
                        continue

                frame_idx += 1

                # Resize for processing
                h, w = frame.shape[:2]
                target_w = min(w, Config.TARGET_WIDTH)
                target_h = int(h * (target_w / w)) if w > 0 else Config.TARGET_HEIGHT
                small_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

                # Process through AI pipeline (with smart interval support)
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

                try:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
                except GeneratorExit:
                    break

                time.sleep(0.02)

        except Exception as e:
            self.status = "ERROR"
            print(f"[CCTV] Stream exception: {e}")
        finally:
            self.is_running = False
            self.status = "DISCONNECTED"
            if cap is not None:
                cap.release()

    def _render_status_frame(self, text, color):
        frame = np.zeros((480, 640, 3), dtype=np.uint8) + 25
        cv2.putText(frame, text, (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        return frame

    def stop(self):
        self.is_running = False
        self.status = "DISCONNECTED"
        if self._threaded_cap is not None:
            self._threaded_cap.release()
