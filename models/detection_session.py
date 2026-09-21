from database.database import db
from datetime import datetime


class DetectionSession(db.Model):
    __tablename__ = "detection_sessions"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    source_type = db.Column(db.String(50), nullable=False)  # 'video', 'image', 'webcam', 'cctv', 'demo'
    source_name = db.Column(db.String(255), nullable=False)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime, nullable=True)
    maximum_people = db.Column(db.Integer, default=0)
    maximum_risk = db.Column(db.String(20), default="GREEN")  # GREEN, YELLOW, RED
    max_risk_score = db.Column(db.Float, default=0.0)
    duration = db.Column(db.Float, default=0.0)  # in seconds
    processed_video_path = db.Column(db.String(255), nullable=True)
    total_frames = db.Column(db.Integer, default=0)
    status = db.Column(db.String(20), default="completed")  # running, completed, failed

    # Relationships
    events = db.relationship("DetectionEvent", backref="session", lazy=True, cascade="all, delete-orphan")
    alerts = db.relationship("Alert", backref="session", lazy=True, cascade="all, delete-orphan")
    crowd_records = db.relationship("CrowdData", backref="session", lazy=True, cascade="all, delete-orphan")

    def __init__(self, user_id=None, source_type="video", source_name="", start_time=None, end_time=None, maximum_people=0, maximum_risk="GREEN", max_risk_score=0.0, duration=0.0, processed_video_path=None, total_frames=0, status="completed", **kwargs):
        super(DetectionSession, self).__init__(**kwargs)
        self.user_id = user_id
        self.source_type = source_type
        self.source_name = source_name
        self.start_time = start_time or datetime.utcnow()
        self.end_time = end_time
        self.maximum_people = maximum_people
        self.maximum_risk = maximum_risk
        self.max_risk_score = max_risk_score
        self.duration = duration
        self.processed_video_path = processed_video_path
        self.total_frames = total_frames
        self.status = status

    def __repr__(self):
        return f"<DetectionSession {self.id} - {self.source_type} ({self.source_name})>"
