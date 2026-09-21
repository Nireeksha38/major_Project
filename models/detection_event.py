from database.database import db
from datetime import datetime


class DetectionEvent(db.Model):
    __tablename__ = "detection_events"

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey("detection_sessions.id"), nullable=False)
    event_type = db.Column(db.String(100), nullable=False)  # 'sudden_rush', 'panic_movement', 'fall_detected', 'high_density', 'stampede_pattern'
    confidence = db.Column(db.Float, default=0.0)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    frame_number = db.Column(db.Integer, default=0)
    description = db.Column(db.Text, nullable=True)

    def __init__(self, session_id=None, event_type=None, confidence=0.0, frame_number=0, description=None, timestamp=None, **kwargs):
        super(DetectionEvent, self).__init__(**kwargs)
        self.session_id = session_id
        if event_type is not None:
            self.event_type = event_type
        self.confidence = confidence
        self.frame_number = frame_number
        self.description = description
        self.timestamp = timestamp or datetime.utcnow()

    def __repr__(self):
        return f"<DetectionEvent {self.event_type} at {self.timestamp}>"
