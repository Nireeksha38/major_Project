import os
import shutil
import threading
import cv2
from flask import Blueprint, request, redirect, url_for, flash, render_template, Response, jsonify
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from config import Config
from detection.video_processor import VideoProcessor
from detection.image_processor import ImageProcessor
from detection.webcam import WebcamStreamer, get_available_cameras, BrowserFrameProcessor
from detection.cctv import CCTVStreamer
from detection.mobile_cam import MobileCamStreamer, verify_droidcam_connection
from detection.video_streamer import VideoFileStreamer
from models.detection_session import DetectionSession
from models.alert import Alert
from models.crowd_data import CrowdData
from models.camera import Camera
from database.database import db

detection = Blueprint("detection", __name__)

ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}
ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp"}

# Global active streamer singletons for CCTV, Webcam, DroidCam/Mobile, and Video Files
active_webcam_streamer = None
browser_webcam_processor = BrowserFrameProcessor()
active_cctv_streamer = None
active_mobile_streamer = None
active_video_streamers = {}


def allowed_video(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


def allowed_image(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def _get_demo_videos():
    demo_videos = {}
    for category in ["normal", "rush", "panic", "abnormal"]:
        cat_dir = os.path.join(Config.VIDEOS_DIR, category)
        if os.path.exists(cat_dir):
            demo_videos[category] = [f for f in os.listdir(cat_dir) if allowed_video(f)]
        else:
            demo_videos[category] = []
    return demo_videos


# ================= UPLOAD VIDEO =================
@detection.route("/upload-video", methods=["GET", "POST"])
@login_required
def upload_video():
    demo_videos = _get_demo_videos()
    if request.method == "GET":
        return render_template("upload_video.html", demo_videos=demo_videos)

    # POST Handling
    if "video" not in request.files:
        flash("No video selected.", "danger")
        return redirect(url_for("detection.upload_video"))

    video = request.files["video"]
    if not video or not video.filename:
        flash("Please choose a video file to upload.", "warning")
        return redirect(url_for("detection.upload_video"))

    if allowed_video(video.filename):
        filename = secure_filename(video.filename)
        save_path = os.path.join(Config.UPLOAD_FOLDER_ORIGINAL, filename)
        os.makedirs(Config.UPLOAD_FOLDER_ORIGINAL, exist_ok=True)
        video.save(save_path)

        # Fast Instant Initialization (< 0.05s)
        processor = VideoProcessor()
        try:
            summary = processor.quick_init_video(
                input_video_path=save_path,
                user_id=current_user.id
            )
            flash(
                f"Surveillance Video Ready ({summary['total_frames']} frames). Live AI Detection Stream Activated.",
                "success"
            )
            return render_template(
                "upload_video.html",
                summary=summary,
                processed_video=summary["processed_video_path"],
                live_video_file=filename,
                original_video=filename,
                demo_videos=demo_videos
            )
        except Exception as e:
            flash(f"Error loading video: {str(e)}", "danger")
            return redirect(url_for("detection.upload_video"))

    flash("Invalid file format. Supported formats: MP4, AVI, MOV, MKV.", "danger")
    return redirect(url_for("detection.upload_video"))


# ================= REAL-TIME VIDEO STREAM FEED =================
@detection.route("/video-feed/<filename>")
@login_required
def video_feed(filename):
    safe_name = secure_filename(filename)
    video_path = os.path.join(Config.UPLOAD_FOLDER_ORIGINAL, safe_name)
    if not os.path.exists(video_path):
        for cat in ["normal", "rush", "panic", "abnormal"]:
            p = os.path.join(Config.VIDEOS_DIR, cat, safe_name)
            if os.path.exists(p):
                video_path = p
                break

    old_streamer = active_video_streamers.get(safe_name)
    if old_streamer:
        try:
            old_streamer.stop()
        except Exception:
            pass

    streamer = VideoFileStreamer(video_path)
    active_video_streamers[safe_name] = streamer
    return Response(
        streamer.generate_frames(loop=True),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@detection.route("/api/video-telemetry/<filename>")
@login_required
def video_telemetry(filename):
    safe_name = secure_filename(filename)
    streamer = active_video_streamers.get(safe_name)
    if streamer:
        return jsonify(streamer.last_telemetry)
    return jsonify({
        "people_count": 0,
        "risk_score": 0.0,
        "risk_level": "GREEN",
        "density_score": 0.0,
        "relative_speed": 0.0,
        "motion_entropy": 0.0,
        "num_falls": 0,
        "status_title": "STREAMING",
        "summary_message": "AI surveillance live detection stream active"
    })


# ================= DEMO VIDEO TEST =================
@detection.route("/run-demo-video/<category>/<filename>", methods=["GET", "POST"])
@login_required
def run_demo_video(category, filename):
    safe_filename = secure_filename(filename)
    demo_path = os.path.join(Config.VIDEOS_DIR, category, safe_filename)

    if not os.path.exists(demo_path):
        flash(f"Demo video '{safe_filename}' not found in videos/{category}/.", "danger")
        return redirect(url_for("detection.upload_video"))

    # Copy to uploads original
    os.makedirs(Config.UPLOAD_FOLDER_ORIGINAL, exist_ok=True)
    target_path = os.path.join(Config.UPLOAD_FOLDER_ORIGINAL, safe_filename)
    shutil.copy2(demo_path, target_path)

    # Fast Instant Initialization (< 0.05s)
    processor = VideoProcessor()
    try:
        summary = processor.quick_init_video(
            input_video_path=target_path,
            user_id=current_user.id
        )
        flash(f"Surveillance Video ({category.upper()}) Ready. Live AI Detection Stream Activated.", "success")
        return render_template(
            "upload_video.html",
            summary=summary,
            processed_video=summary["processed_video_path"],
            live_video_file=safe_filename,
            original_video=safe_filename,
            demo_videos=_get_demo_videos()
        )
    except Exception as e:
        flash(f"Error loading demo video: {str(e)}", "danger")
        return redirect(url_for("detection.upload_video"))


# ================= UPLOAD IMAGE =================
@detection.route("/upload-image", methods=["GET", "POST"])
@login_required
def upload_image():
    samples_dir = os.path.join(Config.BASE_DIR, "static", "uploads", "samples")
    sample_images = []
    if os.path.exists(samples_dir):
        sample_images = [f for f in os.listdir(samples_dir) if allowed_image(f)]

    if request.method == "POST":
        if "image" not in request.files:
            flash("No image selected.", "danger")
            return redirect(url_for("detection.upload_image"))

        image_file = request.files["image"]
        if not image_file or not image_file.filename:
            flash("Please choose an image file.", "warning")
            return redirect(url_for("detection.upload_image"))

        if allowed_image(image_file.filename):
            fname = secure_filename(image_file.filename)
            upload_dir = Config.UPLOAD_FOLDER_IMAGES
            os.makedirs(upload_dir, exist_ok=True)
            save_path = os.path.join(upload_dir, fname)
            image_file.save(save_path)

            processor = ImageProcessor()
            try:
                result = processor.process_image(save_path, user_id=current_user.id)
                flash(f"Image analyzed: {result['people_count']} persons detected with Soft-NMS.", "success")
                return render_template("upload_image.html", result=result, sample_images=sample_images)
            except Exception as e:
                flash(f"Image analysis error: {str(e)}", "danger")
                return redirect(url_for("detection.upload_image"))

        flash("Invalid image format. Supported formats: JPG, JPEG, PNG, BMP.", "danger")

    return render_template("upload_image.html", result=None, sample_images=sample_images)


# ================= DEMO IMAGE TEST =================
@detection.route("/run-demo-image/<filename>", methods=["GET", "POST"])
@login_required
def run_demo_image(filename):
    safe_filename = secure_filename(filename)
    demo_path = os.path.join(Config.BASE_DIR, "static", "uploads", "samples", safe_filename)

    if not os.path.exists(demo_path):
        flash(f"Demo image '{safe_filename}' not found.", "danger")
        return redirect(url_for("detection.upload_image"))

    # Copy to uploads images
    upload_dir = Config.UPLOAD_FOLDER_IMAGES
    os.makedirs(upload_dir, exist_ok=True)
    target_path = os.path.join(upload_dir, safe_filename)
    shutil.copy2(demo_path, target_path)

    # Process through ImageProcessor
    processor = ImageProcessor()
    try:
        result = processor.process_image(target_path, user_id=current_user.id)
        flash(f"Demo Crowd Image analyzed: {result['people_count']} persons detected!", "success")
        
        samples_dir = os.path.join(Config.BASE_DIR, "static", "uploads", "samples")
        sample_images = [f for f in os.listdir(samples_dir) if allowed_image(f)] if os.path.exists(samples_dir) else []
        return render_template("upload_image.html", result=result, sample_images=sample_images)
    except Exception as e:
        flash(f"Error analyzing demo image: {str(e)}", "danger")
        return redirect(url_for("detection.upload_image"))


# ================= WEBCAM =================
@detection.route("/webcam")
@login_required
def webcam_page():
    cameras = get_available_cameras()
    selected_index = request.args.get("camera_index", default=0, type=int)
    return render_template(
        "webcam.html",
        cameras=cameras,
        selected_index=selected_index
    )


@detection.route("/webcam_feed")
@login_required
def webcam_feed():
    global active_webcam_streamer
    camera_index = request.args.get("camera_index", default=0, type=int)
    if active_webcam_streamer is not None:
        active_webcam_streamer.stop()
    active_webcam_streamer = WebcamStreamer(camera_index=camera_index)
    return Response(
        active_webcam_streamer.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@detection.route("/api/webcam/devices")
@login_required
def webcam_devices_api():
    cameras = get_available_cameras()
    return jsonify({
        "status": "success",
        "cameras": cameras,
        "count": len(cameras)
    })


@detection.route("/api/process_webcam_frame", methods=["POST"])
@login_required
def process_webcam_frame():
    global browser_webcam_processor
    data = request.get_json(silent=True) or {}
    frame_data = data.get("frame")
    if not frame_data:
        return jsonify({"success": False, "error": "No frame data received"}), 400

    result = browser_webcam_processor.process_base64_frame(frame_data)
    return jsonify(result)


@detection.route("/api/webcam/reset", methods=["POST"])
@login_required
def webcam_reset():
    global browser_webcam_processor, active_webcam_streamer
    browser_webcam_processor.reset()
    if active_webcam_streamer is not None:
        active_webcam_streamer.stop()
    return jsonify({"status": "success", "message": "Webcam pipeline reset"})


@detection.route("/api/webcam-telemetry")
@login_required
def webcam_telemetry():
    global active_webcam_streamer, browser_webcam_processor
    # Check browser webcam processor first, then backend active streamer
    if browser_webcam_processor and browser_webcam_processor.last_telemetry:
        return jsonify(browser_webcam_processor.last_telemetry)
    if active_webcam_streamer and active_webcam_streamer.last_telemetry:
        return jsonify(active_webcam_streamer.last_telemetry)
    return jsonify({
        "people_count": 0,
        "relative_speed": 0.0,
        "motion_entropy": 0.0,
        "risk_score": 0.0,
        "risk_level": "GREEN",
        "status_title": "NORMAL",
        "summary_message": "Webcam initializing..."
    })


# ================= CCTV / RTSP =================
@detection.route("/cctv")
@login_required
def cctv_page():
    cameras = Camera.query.all()
    return render_template(
        "cctv.html",
        cameras=cameras
    )


def _verify_cctv_connection(stream_url, timeout=3.5):
    """
    Attempts to connect and fetch a single test frame from the CCTV / RTSP stream.
    Returns (success: bool, error_message: str).
    Guarded by a timeout thread to prevent hangs on unreachable IP/RTSP hosts.
    """
    if not stream_url or not stream_url.strip():
        return False, "RTSP stream URL cannot be empty."

    result = {"success": False, "error": "Connection timed out. CCTV stream is unreachable."}

    def _test():
        cap = None
        try:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;3000000"
            cap = cv2.VideoCapture(stream_url.strip())
            if not cap.isOpened():
                result["error"] = "Unable to open RTSP stream. Please check camera IP, port, and credentials."
                return

            ret, frame = cap.read()
            if ret and frame is not None and getattr(frame, "size", 0) > 0:
                result["success"] = True
                result["error"] = ""
            else:
                result["error"] = "Stream opened but failed to retrieve any video frames."
        except Exception as e:
            result["error"] = f"Stream connection error: {str(e)}"
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
        return False, "Connection timed out (camera unreachable or network unresponsive)."

    return result["success"], result["error"]


@detection.route("/connect-cctv", methods=["POST"])
@login_required
def connect_cctv():
    global active_cctv_streamer
    camera_name = request.form.get("camera_name", "Live CCTV").strip()
    rtsp_url = request.form.get("rtsp_url", "").strip()
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()

    if username and password and "@" not in rtsp_url and "rtsp://" in rtsp_url:
        rtsp_url = rtsp_url.replace("rtsp://", f"{username}:{password}@")
        # In case replacement yielded rtsp://admin:pass@...
        if not rtsp_url.startswith("rtsp://"):
            rtsp_url = "rtsp://" + rtsp_url

    # Stop any previous active streamer
    if active_cctv_streamer:
        try:
            active_cctv_streamer.stop()
        except Exception:
            pass

    # Validate RTSP Stream Connection before marking connected
    is_connected, err_msg = _verify_cctv_connection(rtsp_url, timeout=3.5)

    existing = Camera.query.filter_by(name=camera_name).first()

    if not is_connected:
        if not existing:
            cam = Camera(name=camera_name, stream_url=rtsp_url, status="DISCONNECTED")
            db.session.add(cam)
        else:
            existing.stream_url = rtsp_url
            existing.status = "DISCONNECTED"
        db.session.commit()

        flash(f"CCTV Connection Failed: {err_msg}", "danger")
        return redirect(url_for("detection.cctv_page"))

    # Connection verified successfully
    active_cctv_streamer = CCTVStreamer(
        stream_url=rtsp_url
    )

    if not existing:
        cam = Camera(name=camera_name, stream_url=rtsp_url, status="CONNECTED")
        db.session.add(cam)
    else:
        existing.stream_url = rtsp_url
        existing.status = "CONNECTED"
    db.session.commit()

    flash(f"Successfully connected to CCTV camera: {camera_name}", "success")
    return redirect(url_for("detection.cctv_page", streaming="true"))


@detection.route("/disconnect-cctv", methods=["POST"])
@login_required
def disconnect_cctv():
    global active_cctv_streamer
    if active_cctv_streamer:
        active_cctv_streamer.stop()
    flash("Camera disconnected.", "info")
    return redirect(url_for("detection.cctv_page"))


@detection.route("/cctv_feed")
@login_required
def cctv_feed():
    global active_cctv_streamer
    if active_cctv_streamer is None:
        active_cctv_streamer = CCTVStreamer()

    return Response(
        active_cctv_streamer.start_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@detection.route("/api/cctv-telemetry")
@login_required
def cctv_telemetry():
    global active_cctv_streamer
    if active_cctv_streamer and active_cctv_streamer.last_telemetry:
        return jsonify(active_cctv_streamer.last_telemetry)
    return jsonify({
        "people_count": 0,
        "relative_speed": 0.0,
        "motion_entropy": 0.0,
        "risk_score": 0.0,
        "risk_level": "GREEN",
        "status_title": "NORMAL",
        "summary_message": "Awaiting stream signal..."
    })


# ================= MOBILE CAMERA SURVEILLANCE =================
import socket


def _get_lan_ip():
    """Returns local LAN IP for cross-device mobile camera connection."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        lan_ip = s.getsockname()[0]
        s.close()
        return lan_ip
    except Exception:
        return "127.0.0.1"


@detection.route("/mobile-cam")
@login_required
def mobile_cam_page():
    lan_ip = _get_lan_ip()
    port = request.host.split(":")[-1] if ":" in request.host else "5000"
    streaming = request.args.get("streaming", "false")
    active_url = active_mobile_streamer.stream_url if active_mobile_streamer else ""

    return render_template(
        "mobile_cam.html",
        lan_ip=lan_ip,
        port=port,
        streaming=streaming,
        active_url=active_url
    )


@detection.route("/connect-mobile-cam", methods=["POST"])
@login_required
def connect_mobile_cam():
    global active_mobile_streamer
    mobile_url = request.form.get("mobile_url", "").strip()

    if not mobile_url:
        flash("Please enter a valid DroidCam address (e.g. 192.168.1.15:4747).", "warning")
        return redirect(url_for("detection.mobile_cam_page", streaming="false"))

    if active_mobile_streamer:
        try:
            active_mobile_streamer.stop()
        except Exception:
            pass

    # Strictly verify the connection before marking as connected
    is_connected, working_url, err_msg = verify_droidcam_connection(mobile_url, timeout=3.5)

    if not is_connected:
        active_mobile_streamer = None
        flash(f"❌ DroidCam Connection Failed: {err_msg}", "danger")
        return redirect(url_for("detection.mobile_cam_page", streaming="false"))

    active_mobile_streamer = MobileCamStreamer(stream_url=working_url)
    flash(f"✅ Successfully connected to DroidCam ({working_url})!", "success")
    return redirect(url_for("detection.mobile_cam_page", streaming="true"))


@detection.route("/disconnect-mobile-cam", methods=["POST"])
@login_required
def disconnect_mobile_cam():
    global active_mobile_streamer
    if active_mobile_streamer:
        active_mobile_streamer.stop()
    flash("Mobile camera disconnected.", "info")
    return redirect(url_for("detection.mobile_cam_page", streaming="false"))


@detection.route("/mobile_cam_feed")
@login_required
def mobile_cam_feed():
    global active_mobile_streamer
    if active_mobile_streamer is None:
        active_mobile_streamer = MobileCamStreamer()

    return Response(
        active_mobile_streamer.start_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@detection.route("/api/mobile-telemetry")
@login_required
def mobile_telemetry():
    global active_mobile_streamer
    is_active = bool(active_mobile_streamer and active_mobile_streamer.is_running and active_mobile_streamer.status == "CONNECTED")
    if is_active and active_mobile_streamer.last_telemetry:
        data = dict(active_mobile_streamer.last_telemetry)
        data["connected"] = True
        data["stream_status"] = "CONNECTED"
        return jsonify(data)
    return jsonify({
        "connected": is_active,
        "stream_status": "CONNECTED" if is_active else "DISCONNECTED",
        "people_count": 0,
        "relative_speed": 0.0,
        "motion_entropy": 0.0,
        "risk_score": 0.0,
        "risk_level": "GREEN",
        "status_title": "CONNECTED" if is_active else "DISCONNECTED",
        "summary_message": "Live DroidCam stream active" if is_active else "Awaiting DroidCam / IP Camera signal..."
    })


@detection.route("/api/mobile/reset", methods=["POST"])
@login_required
def mobile_reset():
    global active_mobile_streamer
    if active_mobile_streamer:
        active_mobile_streamer.stop()
    return jsonify({"status": "success", "message": "Mobile camera pipeline reset"})