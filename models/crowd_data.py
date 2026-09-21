from database.database import db
from datetime import datetime


class CrowdData(db.Model):
    __tablename__ = "crowd_data"

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey("detection_sessions.id"), nullable=True)
    crowd_count = db.Column(db.Integer, nullable=False, default=0)
    density = db.Column(db.Float, nullable=False, default=0.0)
    relative_speed = db.Column(db.Float, default=0.0)
    motion_entropy = db.Column(db.Float, default=0.0)
    motion_intensity = db.Column(db.Float, default=0.0)
    panic_score = db.Column(db.Float, default=0.0)
    risk_score = db.Column(db.Float, default=0.0)
    risk_level = db.Column(db.String(20), default="GREEN")  # GREEN, YELLOW, RED
    falls_detected = db.Column(db.Integer, default=0)
    frame_number = db.Column(db.Integer, default=0)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __init__(self, session_id=None, crowd_count=0, density=0.0, relative_speed=0.0, motion_entropy=0.0, motion_intensity=0.0, panic_score=0.0, risk_score=0.0, risk_level="GREEN", falls_detected=0, frame_number=0, timestamp=None, **kwargs):
        super(CrowdData, self).__init__(**kwargs)
        self.session_id = session_id
        self.crowd_count = crowd_count
        self.density = density
        self.relative_speed = relative_speed
        self.motion_entropy = motion_entropy
        self.motion_intensity = motion_intensity
        self.panic_score = panic_score
        self.risk_score = risk_score
        self.risk_level = risk_level
        self.falls_detected = falls_detected
        self.frame_number = frame_number
        self.timestamp = timestamp or datetime.utcnow()

    def __repr__(self):
        return f"<CrowdData Count={self.crowd_count}, Risk={self.risk_level} at {self.timestamp}>"