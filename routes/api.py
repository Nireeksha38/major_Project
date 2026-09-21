from flask import Blueprint, jsonify, request
from flask_login import login_required
from models.alert import Alert
from models.detection_session import DetectionSession
from models.crowd_data import CrowdData
from database.database import db
from datetime import datetime, timezone

api = Blueprint("api", __name__)


@api.route("/api/system-status")
def system_status():
    import torch
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return jsonify({
        "status": "OPERATIONAL",
        "cuda_available": torch.cuda.is_available(),
        "device": "CUDA" if torch.cuda.is_available() else "CPU",
        "timestamp": now.isoformat()
    })


@api.route("/api/acknowledge-alert/<int:alert_id>", methods=["POST"])
@login_required
def acknowledge_alert(alert_id):
    alert = Alert.query.get_or_404(alert_id)
    alert.acknowledged = True
    alert.acknowledged_at = datetime.now(timezone.utc).replace(tzinfo=None)
    db.session.commit()
    return jsonify({"success": True, "alert_id": alert_id})


@api.route("/api/latest-stats")
@login_required
def latest_stats():
    latest_crowd = CrowdData.query.order_by(CrowdData.id.desc()).first()
    active_alerts = Alert.query.filter_by(acknowledged=False).count()

    if latest_crowd:
        return jsonify({
            "people_count": latest_crowd.crowd_count,
            "density": round(latest_crowd.density, 2),
            "relative_speed": round(latest_crowd.relative_speed, 2),
            "motion_entropy": round(latest_crowd.motion_entropy, 2),
            "risk_score": round(latest_crowd.risk_score, 2),
            "risk_level": latest_crowd.risk_level,
            "active_alerts": active_alerts
        })

    return jsonify({
        "people_count": 0,
        "density": 0.0,
        "relative_speed": 0.0,
        "motion_entropy": 0.0,
        "risk_score": 0.0,
        "risk_level": "GREEN",
        "active_alerts": 0
    })
