import base64
import subprocess
import time
import cv2
import numpy as np
from config import Config
from detection.video_processor import VideoProcessor


def get_available_cameras():
    """
    Enumerate connected cameras on the system (both laptop built-in and external USB webcams).
    Uses DirectShow enumeration via pygrabber with Windows PnP fallback.
    """
    cameras = []
    
    # Method 1: Pygrabber DirectShow enumeration
    try:
        from pygrabber.dshow_graph import FilterGraph
        graph = FilterGraph()
        devices = graph.get_input_devices()
        for idx, name in enumerate(devices):
            cam_name_lower = name.lower()
            if any(k in cam_name_lower for k in ["hp", "integrated", "built-in", "internal", "truevision", "front", "facetime", "wide vision"]):
                cam_type = "Laptop Built-in Webcam"
                icon = "💻"
            else:
                cam_type = "External USB Webcam"
                icon = "📷"
            cameras.append({
                "index": idx,
                "name": name,
                "label": f"{icon} Camera {idx}: {name} ({cam_type})",
                "type": cam_type,
                "icon": icon
            })
    except Exception:
        pass

    # Method 2: PowerShell PnP fallback if pygrabber produced no devices
    if not cameras:
        try:
            ps_cmd = 'Get-PnpDevice -Class "Camera","Image" -Status OK -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FriendlyName'
            res = subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], capture_output=True, text=True, timeout=3)
            if res.returncode == 0 and res.stdout.strip():
                lines = [l.strip() for l in res.stdout.strip().splitlines() if l.strip()]
                for idx, name in enumerate(lines):
                    cam_name_lower = name.lower()
                    if any(k in cam_name_lower for k in ["hp", "integrated", "built-in", "internal", "truevision", "front", "facetime", "wide vision"]):
                        cam_type = "Laptop Built-in Webcam"
                        icon = "💻"
                    else:
                        cam_type = "External USB Webcam"
                        icon = "📷"
                    cameras.append({
                        "index": idx,
                        "name": name,
                        "label": f"{icon} Camera {idx}: {name} ({cam_type})",
                        "type": cam_type,
                        "icon": icon
                    })
        except Exception:
            pass

    # Method 3: Standard default device index fallback
    if not cameras:
        cameras = [
            {"index": 0, "name": "Camera 0 (Primary / Laptop Webcam)", "label": "💻 Camera 0: Primary / Built-in Webcam", "type": "Laptop Built-in Webcam", "icon": "💻"},
            {"index": 1, "name": "Camera 1 (External USB Webcam)", "label": "📷 Camera 1: External USB Webcam", "type": "External USB Webcam", "icon": "📷"},
            {"index": 2, "name": "Camera 2 (Secondary / Virtual Device)", "label": "📹 Camera 2: Secondary Video Device", "type": "Secondary", "icon": "📹"},
        ]

    return cameras


class WebcamStreamer:
    """
    Live Webcam surveillance stream handler supporting laptop built-in webcams and external USB webcams.
    Runs YOLOv8 + Soft-NMS + Deep SORT + Optical Flow + Risk Engine in real-time.
    Yields MJPEG multipart frames for HTML video streaming.
    """
    def __init__(self, camera_index=0):
        try:
            self.camera_index = int(camera_index)
        except (ValueError, TypeError):
            self.camera_index = 0
        self.processor = VideoProcessor()
        self.is_running = False
        self.last_telemetry = {}
        self.connection_status = "INITIALIZING"

    def _open_camera(self, index):
        """Attempts to open camera using multiple backends (DSHOW, MSMF, ANY)."""
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        for backend in backends:
            try:
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        return cap
                    cap.release()
            except Exception:
                pass
        return None

    def _create_error_frame(self, title, message_lines):
        """Generates an informative diagnostic frame when camera access is blocked or disconnected."""
        frame = np.full((480, 640, 3), 20, dtype=np.uint8)
        
        # Red Header Alert Banner
        cv2.rectangle(frame, (0, 0), (640, 56), (35, 35, 190), -1)
        cv2.putText(frame, title, (20, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)

        # Draw guideline box
        cv2.rectangle(frame, (20, 75), (620, 450), (45, 45, 55), 1)
        
        y = 110
        for line in message_lines:
            if line.startswith("•") or line.startswith("1.") or line.startswith("2.") or line.startswith("3.") or line.startswith("4."):
                cv2.putText(frame, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (120, 220, 255), 1)
            elif line.startswith("[!]") or line.startswith("💡"):
                cv2.putText(frame, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 220, 160), 1)
            else:
                cv2.putText(frame, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1)
            y += 32

        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")

    def generate_frames(self, session_id=None):
        """Generator function that yields JPEG encoded frames for Flask Response."""
        cap = self._open_camera(self.camera_index)
        
        if cap is None:
            # Try alternate camera index (if 0 failed, try 1; if 1 failed, try 0)
            alternate_index = 1 if self.camera_index == 0 else 0
            alt_cap = self._open_camera(alternate_index)
            if alt_cap is not None:
                cap = alt_cap
                self.camera_index = alternate_index

        if cap is None or not cap.isOpened():
            self.connection_status = "CAMERA_UNAVAILABLE"
            err_bytes = self._create_error_frame(
                f"CAMERA HARDWARE LOCKED / BLOCKED (Index {self.camera_index})",
                [
                    f"Selected Camera Device Index: {self.camera_index}",
                    "💡 RECOMMENDED: Switch to 'Browser WebCam' mode above for 1-click access!",
                    "If using Backend OpenCV mode:",
                    "1. Windows Privacy Settings: Privacy & Security -> Camera -> Desktop Apps: ON",
                    "2. Another app is currently using the camera (Zoom/Teams/Chrome/Camera app).",
                    "   Fix: Close background meeting apps or active browser video tabs.",
                    "3. Physical slider or Fn privacy camera switch is muted on laptop.",
                    "4. Try switching between Camera 0 and Camera 1 in the dropdown above."
                ]
            )
            yield err_bytes
            return

        self.connection_status = "STREAMING"
        self.processor.reset()
        self.is_running = True
        frame_idx = 0

        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                # Resize for smooth laptop performance
                h, w = frame.shape[:2]
                target_w = min(w, Config.TARGET_WIDTH)
                target_h = int(h * (target_w / w)) if w > 0 else Config.TARGET_HEIGHT
                small_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

                # Process Frame with AI Pipeline
                annotated_frame, telemetry = self.processor.process_frame(
                    small_frame,
                    frame_idx=frame_idx,
                    session_id=session_id
                )
                self.last_telemetry = telemetry

                out_img = annotated_frame if isinstance(annotated_frame, np.ndarray) else small_frame

                # Encode to JPEG
                _, buffer = cv2.imencode(".jpg", out_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = buffer.tobytes()

                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

                # Throttle slightly to keep ~20-25 FPS
                time.sleep(0.03)
        finally:
            self.is_running = False
            cap.release()

    def stop(self):
        self.is_running = False


class BrowserFrameProcessor:
    """
    Handles real-time AI processing for In-Browser WebCam (HTML5 / WebRTC capture).
    Receives base64 image frames from the browser, executes YOLOv8 + Soft-NMS +
    Deep SORT + Optical Flow + Risk Classifier, and returns annotated frame + live telemetry.
    """
    def __init__(self):
        self.processor = VideoProcessor()
        self.frame_idx = 0
        self.last_telemetry = {}

    def reset(self):
        """Resets tracker and temporal analyzers for a fresh stream."""
        self.processor.reset()
        self.frame_idx = 0
        self.last_telemetry = {}

    def process_base64_frame(self, base64_data, session_id=None):
        """
        Decodes a base64 frame, processes it through the surveillance AI pipeline,
        and returns the annotated base64 frame along with telemetry.
        """
        t_start = time.time()
        try:
            # Strip data URL prefix if present
            if "," in base64_data:
                base64_data = base64_data.split(",", 1)[1]

            img_bytes = base64.b64decode(base64_data)
            np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                return {
                    "success": False,
                    "error": "Failed to decode image frame"
                }

            self.frame_idx += 1
            h, w = frame.shape[:2]
            target_w = min(w, Config.TARGET_WIDTH)
            target_h = int(h * (target_w / w)) if w > 0 else Config.TARGET_HEIGHT
            small_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            # Process frame with AI Pipeline
            annotated_frame, telemetry = self.processor.process_frame(
                small_frame,
                frame_idx=self.frame_idx,
                session_id=session_id
            )
            self.last_telemetry = telemetry

            out_img = annotated_frame if isinstance(annotated_frame, np.ndarray) else small_frame

            # Encode annotated frame back to JPEG base64
            _, buffer = cv2.imencode(".jpg", out_img, [cv2.IMWRITE_JPEG_QUALITY, 78])
            annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

            elapsed_ms = round((time.time() - t_start) * 1000, 1)

            return {
                "success": True,
                "annotated_image": annotated_b64,
                "telemetry": telemetry,
                "frame_idx": self.frame_idx,
                "latency_ms": elapsed_ms
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
