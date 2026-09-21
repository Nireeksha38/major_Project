from database.database import db


class EventLog(db.Model):
    __tablename__ = "event_logs"

    id = db.Column(db.Integer, primary_key=True)

    event = db.Column(db.String(255))

    description = db.Column(db.Text)

    created_at = db.Column(db.DateTime, server_default=db.func.now())