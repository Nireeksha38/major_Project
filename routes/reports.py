import os
import csv
from datetime import datetime, timezone
from flask import Blueprint, render_template, request, redirect, url_for, flash, send_file
from flask_login import login_required
from config import Config
from models.detection_session import DetectionSession
from models.detection_event import DetectionEvent
from models.alert import Alert
from models.report import Report
from database.database import db

reports = Blueprint("reports", __name__)


@reports.route("/reports")
@login_required
def reports_home():
    all_reports = Report.query.order_by(Report.id.desc()).all()
    sessions = DetectionSession.query.order_by(DetectionSession.id.desc()).limit(15).all()
    return render_template("reports.html", reports=all_reports, sessions=sessions)


@reports.route("/generate-report/<int:session_id>", methods=["POST"])
@login_required
def generate_session_report(session_id):
    session = DetectionSession.query.get_or_404(session_id)
    report_type = request.form.get("report_type", "CSV").upper()

    os.makedirs(Config.OUTPUT_REPORTS, exist_ok=True)
    now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_name = f"Incident_Report_Session_{session.id}_{now_str}"

    events = DetectionEvent.query.filter_by(session_id=session.id).all()
    alerts = Alert.query.filter_by(session_id=session.id).all()

    if report_type == "CSV":
        file_name = f"{report_name}.csv"
        file_path = os.path.join(Config.OUTPUT_REPORTS, file_name)

        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["CROWD STAMPEDE PREDICTION SYSTEM - INCIDENT REPORT"])
            writer.writerow(["Session ID", session.id])
            writer.writerow(["Source", f"{session.source_type} ({session.source_name})"])
            writer.writerow(["Start Time", session.start_time])
            writer.writerow(["Peak People Count", session.maximum_people])
            writer.writerow(["Maximum Risk Level", session.maximum_risk])
            writer.writerow(["Maximum Risk Score", session.max_risk_score])
            writer.writerow(["Duration (seconds)", session.duration])
            writer.writerow([])
            writer.writerow(["--- DETECTED EVENTS ---"])
            writer.writerow(["Event Type", "Confidence", "Timestamp", "Description"])
            for e in events:
                writer.writerow([e.event_type, e.confidence, e.timestamp, e.description])
            writer.writerow([])
            writer.writerow(["--- ALERTS TRIGGERED ---"])
            writer.writerow(["Alert Type", "Severity", "Risk Level", "Timestamp", "Message"])
            for a in alerts:
                writer.writerow([a.alert_type, a.severity, a.risk_level, a.timestamp, a.message])

        # Save Report in DB
        db_rep = Report(
            session_id=session.id,
            report_name=file_name,
            report_type="CSV",
            file_path=file_path,
            summary=f"Incident CSV report for session {session.id} with peak risk {session.maximum_risk}"
        )
        db.session.add(db_rep)
        db.session.commit()

        flash(f"Report '{file_name}' generated successfully!", "success")
        return redirect(url_for("reports.reports_home"))

    flash("Report generated.", "info")
    return redirect(url_for("reports.reports_home"))


@reports.route("/download-report/<int:report_id>")
@login_required
def download_report(report_id):
    report = Report.query.get_or_404(report_id)
    if report.file_path and os.path.exists(report.file_path):
        return send_file(report.file_path, as_attachment=True)
    flash("Report file not found on disk.", "danger")
    return redirect(url_for("reports.reports_home"))
