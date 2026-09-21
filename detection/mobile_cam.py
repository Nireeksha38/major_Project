import os
import time
import threading
import cv2
import numpy as np
from config import Config
from detection.video_processor import VideoProcessor


def _get_droidcam_candidates(raw_url):
    """
    Expands user input into candidate DroidCam video stream URLs.
    Handles:
      - 192.168.1.15
      - 192.168.1.15:4747
      - http://192.168.1.15:4747
      - http://192.168.1.15:4747/video
      - http://192.168.1.15:4747/mjpegfeed
    """
    raw_url = raw_url.strip()
    if not raw_url:
        return []

    # Strip existing protocol if any
    clean = raw_url
    if "://" in clean:
        clean = clean.split("://", 1)[1]

    # Strip subpaths
    for subpath in ["/video", "/mjpegfeed", "/video?640x480"]:
        if clean.endswith(subpath):
            clean = clean[:-len(subpath)]

    # If no port specified, default to 4747 for DroidCam
    if ":" not in clean.split("/")[0]:
        clean = f"{clean}:4747"

    ip_port = clean.rstrip("/")

    candidates = [
        f"http://{ip_port}/video",
        f"http://{ip_port}/mjpegfeed",
        f"http://{ip_port}/video?640x480",
        f"http://{ip_port}"
    ]

    # Ensure uniqueness while preserving priority order
    seen = set()
    return [c for c in candidates if not (c in seen or seen.add(c))]


def verify_droidcam_connection(stream_url, timeout=3.5):
    """
    Attempts to connect and fetch a test frame from DroidCam candidate URLs.
    Returns (success: bool, working_url: str, error_message: str).
    Guarded by a timeout thread to immediately detect unreachable/wrong IP addresses.
    """
    if not stream_url or not stream_url.strip():
        return False, "", "DroidCam address cannot be empty. Please enter your phone's IP (e.g. 192.168.1.15:4747)."

    candidates = _get_droidcam_candidates(stream_url)
    if not candidates:
        return False, "", "Invalid camera address format."

    result = {
        "success": False,
        "working_url": "",
        "error": f"Connection timed out. Could not reach DroidCam at '{stream_url}'."
    }

    def _test():
        for url in candidates:
            cap = None
            try:
                cap = cv2.VideoCapture(url)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None and getattr(frame, "size", 0) > 0:
                        result["success"] = True
                        result["working_url"] = url
                        result["error"] = ""
                        return
            except Exception as e:
                result["error"] = str(e)
            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass

    worker = threading.Thread(target=_test, daemon=True)
    worker.start()
    worker.join(timeout=timeout)

    if worker.is_alive():
        return False, "", f"Connection timed out. Host '{stream_url}' is unreachable. Ensure phone and PC are on the same Wi-Fi and DroidCam is open."

    if not result["success"]:
        return False, "", f"Unable to connect to DroidCam at '{stream_url}'. Please ensure DroidCam is open on your phone and the IP/Port is correct."

    return result["success"], result["working_url"], result["error"]


class MobileCamStreamer:
    """
    Dedicated DroidCam Mobile Phone Camera Streamer.
    Streams live video from DroidCam over Wi-Fi, running real-time
    YOLOv8 person detection, tracking, fall detection, and stampede risk analysis.
    """
    def __init__(self, stream_url=None):
        self.stream_url = stream_url.strip() if stream_url else ""
        self.processor = VideoProcessor()
        self.is_running = False
        self.status = "DISCONNECTED"
        self.last_telemetry = {}
        self.active_working_url = ""

    def _open_capture(self):
        """Tries DroidCam endpoints and returns the active VideoCapture."""
        candidates = _get_droidcam_candidates(self.stream_url)
        print(f"[DroidCam] Probing stream candidates for '{self.stream_url}': {candidates}")

        for url in candidates:
            try:
                cap = cv2.VideoCapture(url)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        self.active_working_url = url
                        print(f"[DroidCam] Successfully connected to active stream: {url}")
                        return cap
                    cap.release()
            except Exception as e:
                print(f"[DroidCam] Candidate '{url}' failed: {e}")
        return None

    def start_stream(self, session_id=None):
        """Yields MJPEG multipart frame stream for Flask Response."""
        self.is_running = True
        self.status = "CONNECTED"
        self.processor.reset()
        frame_idx = 0

        cap = self._open_capture()

        if cap is None or not cap.isOpened():
            self.status = "ERROR"
            err_frame = self._render_status_frame(
                [
                    "COULD NOT CONNECT TO DROIDCAM",
                    f"Target IP / Port: {self.stream_url}",
                    "Troubleshooting checklist:",
                    "1. Ensure Phone and Laptop are connected to the SAME Wi-Fi network.",
                    "2. Make sure the DroidCam app is OPEN on your phone screen.",
                    "3. Check the 'WiFi IP' and 'Port' in DroidCam (default port is 4747).",
                    "4. Try typing: 192.168.1.X:4747 or http://192.168.1.X:4747/video"
                ],
                (0, 0, 255)
            )
            _, jpeg = cv2.imencode(".jpg", err_frame)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
            return

        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.04)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break

                frame_idx += 1
                h, w = frame.shape[:2]
                target_w = min(w, Config.TARGET_WIDTH)
                target_h = int(h * (target_w / w)) if w > 0 else Config.TARGET_HEIGHT
                small_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

                # Process Frame through full AI surveillance pipeline
                annotated_frame, telemetry = self.processor.process_frame(
                    small_frame,
                    frame_idx=frame_idx,
                    session_id=session_id
                )
                self.last_telemetry = telemetry

                out_img = annotated_frame if isinstance(annotated_frame, np.ndarray) else small_frame

                # Add DroidCam watermark badge
                cv2.rectangle(out_img, (10, target_h - 45), (170, target_h - 15), (20, 20, 20), -1)
                cv2.putText(out_img, "📱 DROIDCAM LIVE", (18, target_h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1)

                _, buffer = cv2.imencode(".jpg", out_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = buffer.tobytes()

                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
                time.sleep(0.03)

        except Exception as e:
            self.status = "ERROR"
            print(f"[DroidCam] Stream exception: {e}")
        finally:
            self.is_running = False
            self.status = "DISCONNECTED"
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    def _render_status_frame(self, lines, color):
        frame = np.zeros((480, 640, 3), dtype=np.uint8) + 20
        # Header banner
        cv2.rectangle(frame, (0, 0), (640, 50), (25, 25, 160), -1)
        cv2.putText(frame, lines[0] if lines else "DROIDCAM OFFLINE", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
        y = 90
        for l in lines[1:]:
            cv2.putText(frame, l, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 220, 255), 1)
            y += 32
        return frame

    def stop(self):
        self.is_running = False
        self.status = "DISCONNECTED"
