from database.database import db
from datetime import datetime


class Report(db.Model):
    __tablename__ = "reports"

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey("detection_sessions.id"), nullable=True)
    report_name = db.Column(db.String(100), nullable=False)
    report_type = db.Column(db.String(50), default="PDF")  # PDF, CSV, JSON
    file_path = db.Column(db.String(255), nullable=True)
    summary = db.Column(db.Text, nullable=True)
    generated_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __init__(self, session_id=None, report_name="", report_type="PDF", file_path=None, summary=None, generated_at=None, **kwargs):
        super(Report, self).__init__(**kwargs)
        self.session_id = session_id
        self.report_name = report_name
        self.report_type = report_type
        self.file_path = file_path
        self.summary = summary
        self.generated_at = generated_at or datetime.utcnow()

    def __repr__(self):
        return f"<Report {self.report_name} ({self.report_type})>"