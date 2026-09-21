from flask import Blueprint, render_template, request
from flask_login import login_required, current_user
from models.detection_session import DetectionSession
from models.alert import Alert
from models.crowd_data import CrowdData
from models.camera import Camera

dashboard = Blueprint("dashboard", __name__)


@dashboard.route("/dashboard")
@login_required
def dashboard_home():
    video = request.args.get("video")

    # Fetch recent session and alerts from database
    latest_session = DetectionSession.query.order_by(DetectionSession.id.desc()).first()
    recent_alerts = Alert.query.order_by(Alert.id.desc()).limit(5).all()
    active_alerts_count = Alert.query.filter_by(acknowledged=False).count()
    cameras = Camera.query.all()
    recent_sessions = DetectionSession.query.order_by(DetectionSession.id.desc()).limit(5).all()

    # Get latest metrics
    latest_crowd = CrowdData.query.order_by(CrowdData.id.desc()).first()

    current_risk = latest_crowd.risk_level if latest_crowd else (latest_session.maximum_risk if latest_session else "GREEN")
    current_people = latest_crowd.crowd_count if latest_crowd else (latest_session.maximum_people if latest_session else 0)
    current_speed = round(latest_crowd.relative_speed, 2) if latest_crowd else 0.0
    current_entropy = round(latest_crowd.motion_entropy, 2) if latest_crowd else 0.0

    return render_template(
        "dashboard.html",
        user=current_user,
        video=video,
        current_risk=current_risk,
        current_people=current_people,
        current_speed=current_speed,
        current_entropy=current_entropy,
        active_alerts_count=active_alerts_count,
        recent_alerts=recent_alerts,
        recent_sessions=recent_sessions,
        cameras=cameras,
        latest_session=latest_session
    )