import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    BASE_DIR = BASE_DIR
    # Core Security
    SECRET_KEY = os.environ.get("SECRET_KEY", "CrowdStampede@2026")
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'database.db')}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Gmail SMTP Configuration
    MAIL_SERVER = os.environ.get("MAIL_SERVER", "smtp.gmail.com")
    MAIL_PORT = int(os.environ.get("MAIL_PORT", 587))
    MAIL_USE_TLS = os.environ.get("MAIL_USE_TLS", "True").lower() == "true"
    MAIL_USE_SSL = os.environ.get("MAIL_USE_SSL", "False").lower() == "true"
    MAIL_USERNAME = os.environ.get("MAIL_USERNAME", "nireekshapoojary38@gmail.com")
    MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD", "pqbupeudrpxjyisz")
    MAIL_DEFAULT_SENDER = os.environ.get("MAIL_DEFAULT_SENDER", ("Crowd Stampede AI System", MAIL_USERNAME))
    OTP_EXPIRY = int(os.environ.get("OTP_EXPIRY", 300))

    # Automatic Surveillance Emergency Alert Emails
    ENABLE_ALERT_EMAILS = os.environ.get("ENABLE_ALERT_EMAILS", "True").lower() == "true"
    ALERT_EMAIL_RECIPIENT = os.environ.get("ALERT_EMAIL_RECIPIENT", MAIL_USERNAME)
    ALERT_EMAIL_COOLDOWN = int(os.environ.get("ALERT_EMAIL_COOLDOWN", 60))

    # Directories
    UPLOAD_FOLDER_ORIGINAL = os.path.join(BASE_DIR, "static", "uploads", "original")
    UPLOAD_FOLDER_FRAMES = os.path.join(BASE_DIR, "static", "uploads", "frames")
    UPLOAD_FOLDER_PROCESSED = os.path.join(BASE_DIR, "static", "uploads", "processed")
    UPLOAD_FOLDER_IMAGES = os.path.join(BASE_DIR, "static", "uploads", "images")
    OUTPUT_LOGS = os.path.join(BASE_DIR, "outputs", "logs")
    OUTPUT_REPORTS = os.path.join(BASE_DIR, "reports")
    MODELS_DIR = os.path.join(BASE_DIR, "models", "weights")
    VIDEOS_DIR = os.path.join(BASE_DIR, "videos")

    # Detection & Model Settings
    _custom_best = os.path.join(BASE_DIR, "models", "weights", "best.pt")
    _ml_best = os.path.join(BASE_DIR, "ml", "models", "best.pt")
    _default_model = _custom_best if os.path.exists(_custom_best) else (_ml_best if os.path.exists(_ml_best) else "yolov8n.pt")
    YOLO_MODEL_NAME = os.environ.get("YOLO_MODEL_NAME", _default_model)
    YOLO_CONFIDENCE = float(os.environ.get("YOLO_CONFIDENCE", 0.35))
    YOLO_PERSON_CLASS_ID = 0
    YOLO_IMGSZ = int(os.environ.get("YOLO_IMGSZ", 480))
    DEVICE = os.environ.get("DEVICE", "auto")  # 'cuda', 'cpu', or 'auto'
    FRAME_SKIP = int(os.environ.get("FRAME_SKIP", 2))  # 1 = process every frame, 2 = every 2nd frame
    TARGET_WIDTH = 640
    TARGET_HEIGHT = 480

    # Soft-NMS Parameters
    SOFT_NMS_METHOD = os.environ.get("SOFT_NMS_METHOD", "gaussian")  # 'gaussian' or 'linear'
    SOFT_NMS_SIGMA = float(os.environ.get("SOFT_NMS_SIGMA", 0.5))
    SOFT_NMS_IOU_THRESHOLD = float(os.environ.get("SOFT_NMS_IOU_THRESHOLD", 0.45))
    SOFT_NMS_CONF_THRESHOLD = float(os.environ.get("SOFT_NMS_CONF_THRESHOLD", 0.35))

    # Deep SORT Tracking Parameters
    TRACKER_MAX_AGE = int(os.environ.get("TRACKER_MAX_AGE", 30))
    TRACKER_MIN_HITS = int(os.environ.get("TRACKER_MIN_HITS", 2))
    TRACKER_IOU_THRESHOLD = float(os.environ.get("TRACKER_IOU_THRESHOLD", 0.3))

    # Optical Flow & Motion Analysis Parameters
    OPTICAL_FLOW_DOWNSCALE = 0.35
    ENTROPY_BINS = 8
    TEMPORAL_WINDOW = 15  # frames for rush/panic temporal confirmation

    # Fall Detection Parameters
    FALL_ASPECT_RATIO_THRESH = float(os.environ.get("FALL_ASPECT_RATIO_THRESH", 1.45))
    FALL_VELOCITY_THRESH = float(os.environ.get("FALL_VELOCITY_THRESH", 18.0))

    # Risk Engine Weights & Thresholds
    WEIGHT_DENSITY = 0.20
    WEIGHT_SPEED = 0.20
    WEIGHT_MOTION = 0.20
    WEIGHT_ENTROPY = 0.15
    WEIGHT_PANIC = 0.10
    WEIGHT_FALL = 0.15

    RISK_GREEN_THRESHOLD = 0.46
    RISK_YELLOW_THRESHOLD = 0.70
    ALERT_COOLDOWN_SECONDS = 10

    # Demo Account
    DEMO_EMAIL = "demo@example.com"
    DEMO_USERNAME = "demo"
    DEMO_PASSWORD = "Demo@123"
    DEMO_NAME = "Demo Administrator"