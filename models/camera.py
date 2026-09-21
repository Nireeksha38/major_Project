from database.database import db
from datetime import datetime


class Camera(db.Model):
    __tablename__ = "cameras"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    stream_url = db.Column(db.String(255), nullable=False)
    location = db.Column(db.String(150), default="Main Entrance")
    status = db.Column(db.String(20), default="DISCONNECTED")  # 'CONNECTED', 'DISCONNECTED', 'ERROR'
    fps = db.Column(db.Integer, default=25)
    last_connected = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __init__(self, name=None, stream_url=None, location="Main Entrance", status="DISCONNECTED", fps=25, last_connected=None, created_at=None, **kwargs):
        super(Camera, self).__init__(**kwargs)
        if name is not None:
            self.name = name
        if stream_url is not None:
            self.stream_url = stream_url
        self.location = location
        self.status = status
        self.fps = fps
        self.last_connected = last_connected
        self.created_at = created_at or datetime.utcnow()

    def __repr__(self):
        return f"<Camera {self.name} ({self.status})>"
