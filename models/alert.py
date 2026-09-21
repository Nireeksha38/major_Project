from database.database import db
from datetime import datetime


class Alert(db.Model):
    __tablename__ = "alerts"

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey("detection_sessions.id"), nullable=True)
    alert_type = db.Column(db.String(100), nullable=False)  # 'STAMPEDE_RISK', 'FALL_DETECTED', 'PANIC_MOVEMENT', 'SUDDEN_RUSH', 'HIGH_DENSITY'
    severity = db.Column(db.String(20), nullable=False, default="WARNING")  # 'INFO', 'WARNING', 'CRITICAL'
    risk_level = db.Column(db.String(20), default="YELLOW")  # 'GREEN', 'YELLOW', 'RED'
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    acknowledged = db.Column(db.Boolean, default=False)
    acknowledged_at = db.Column(db.DateTime, nullable=True)
    acknowledged_by = db.Column(db.String(100), nullable=True)

    def __init__(self, session_id=None, alert_type=None, severity="WARNING", risk_level="YELLOW", message=None, timestamp=None, acknowledged=False, acknowledged_at=None, acknowledged_by=None, **kwargs):
        super(Alert, self).__init__(**kwargs)
        self.session_id = session_id
        if alert_type is not None:
            self.alert_type = alert_type
        self.severity = severity
        self.risk_level = risk_level
        if message is not None:
            self.message = message
        self.timestamp = timestamp or datetime.utcnow()
        self.acknowledged = acknowledged
        self.acknowledged_at = acknowledged_at
        self.acknowledged_by = acknowledged_by

    def __repr__(self):
        return f"<Alert {self.severity} - {self.alert_type} at {self.timestamp}>"