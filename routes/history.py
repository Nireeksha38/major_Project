from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
from datetime import datetime, timedelta, timezone
from models.detection_session import DetectionSession
from models.detection_event import DetectionEvent
from models.alert import Alert
from database.database import db

history = Blueprint("history", __name__)


@history.route("/history")
@login_required
def history_home():
    filter_period = request.args.get("filter", "all")
    query = DetectionSession.query

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    if filter_period == "today":
        start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        query = query.filter(DetectionSession.start_time >= start_date)
    elif filter_period == "week":
        start_date = now - timedelta(days=7)
        query = query.filter(DetectionSession.start_time >= start_date)
    elif filter_period == "month":
        start_date = now - timedelta(days=30)
        query = query.filter(DetectionSession.start_time >= start_date)

    sessions = query.order_by(DetectionSession.id.desc()).all()

    return render_template(
        "history.html",
        sessions=sessions,
        filter_period=filter_period
    )


@history.route("/history/session/<int:session_id>")
@login_required
def session_details(session_id):
    session = DetectionSession.query.get_or_404(session_id)
    events = DetectionEvent.query.filter_by(session_id=session_id).order_by(DetectionEvent.timestamp.desc()).all()
    alerts = Alert.query.filter_by(session_id=session_id).order_by(Alert.timestamp.desc()).all()

    return jsonify({
        "id": session.id,
        "source_type": session.source_type,
        "source_name": session.source_name,
        "start_time": session.start_time.strftime("%Y-%m-%d %H:%M:%S") if session.start_time else "",
        "end_time": session.end_time.strftime("%Y-%m-%d %H:%M:%S") if session.end_time else "",
        "max_people": session.maximum_people,
        "max_risk": session.maximum_risk,
        "max_risk_score": session.max_risk_score,
        "duration": session.duration,
        "processed_video_path": session.processed_video_path,
        "events": [{
            "type": e.event_type,
            "confidence": e.confidence,
            "description": e.description,
            "timestamp": e.timestamp.strftime("%H:%M:%S")
        } for e in events],
        "alerts": [{
            "type": a.alert_type,
            "severity": a.severity,
            "message": a.message,
            "timestamp": a.timestamp.strftime("%H:%M:%S")
        } for a in alerts]
    })


@history.route("/history/delete/<int:session_id>", methods=["POST"])
@login_required
def delete_session(session_id):
    session = DetectionSession.query.get_or_404(session_id)
    db.session.delete(session)
    db.session.commit()
    flash(f"Session #{session_id} deleted successfully.", "success")
    return redirect(url_for("history.history_home"))
