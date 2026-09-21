from database.database import db
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default="operator")  # admin, operator, viewer
    otp = db.Column(db.String(10), nullable=True)
    otp_expiry = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    sessions = db.relationship("DetectionSession", backref="user", lazy=True, cascade="all, delete-orphan")

    def __init__(self, full_name=None, username=None, email=None, password=None, role="operator", otp=None, otp_expiry=None, created_at=None, **kwargs):
        super(User, self).__init__(**kwargs)
        if full_name is not None:
            self.full_name = full_name
        if username is not None:
            self.username = username
        if email is not None:
            self.email = email
        if password is not None:
            self.set_password(password)
        self.role = role
        self.otp = otp
        self.otp_expiry = otp_expiry
        self.created_at = created_at or datetime.utcnow()

    def set_password(self, password):
        self.password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password, password)

    def __repr__(self):
        return f"<User {self.email}>"