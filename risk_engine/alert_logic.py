import time
from datetime import datetime, timezone
from flask import has_app_context
from config import Config
from database.database import db
from models.alert import Alert
from models.detection_event import DetectionEvent
from utils.email_service import send_alert_email_async


class AlertManager:
    """
    Manages generation, throttling (cooldown), automated emergency email dispatch,
    and database persistence of surveillance alerts.
    """
    def __init__(self, cooldown_seconds=None):
        self.cooldown_seconds = cooldown_seconds or Config.ALERT_COOLDOWN_SECONDS
        self.email_cooldown_seconds = getattr(Config, "ALERT_EMAIL_COOLDOWN", 60)
        self.last_alert_times = {}  # alert_type -> timestamp
        self.last_email_times = {}  # alert_type -> timestamp
        self.active_alerts = []

    def check_and_trigger(self, session_id, risk_eval, panic_info, fall_info, frame_number=0):
        """
        Evaluates current risk state and generates dominant alert if thresholds are exceeded
        and cooldown has elapsed.
        Returns:
            new_alerts: list of newly triggered alert dicts
        """
        now = time.time()
        new_alerts = []
        risk_level = risk_eval.get("risk_level", "GREEN")
        risk_score = risk_eval.get("risk_score", 0.0)
        people_count = risk_eval.get("people_count", "N/A")

        # Alert Condition 1: Critical Fall Alert (Highest immediacy)
        if fall_info.get("fall_detected", False):
            alert_type = "FALL_DETECTED"
            if (now - self.last_alert_times.get(alert_type, 0)) > self.cooldown_seconds:
                severity = "CRITICAL" if risk_level == "RED" else "WARNING"
                count = fall_info.get("num_falls", 1)
                msg = f"Possible fallen person detected! ({count} person(s) identified on ground)"
                alert_obj = self._create_alert(session_id, alert_type, severity, risk_level, msg, frame_number, people_count)
                new_alerts.append(alert_obj)
                self.last_alert_times[alert_type] = now
                self._maybe_send_email_alert(alert_obj, now)

        # Alert Condition 2: Stampede / Chaotic Panic Movement
        elif panic_info.get("is_panic", False) or panic_info.get("panic_detected", False):
            alert_type = "PANIC_MOVEMENT"
            if (now - self.last_alert_times.get(alert_type, 0)) > self.cooldown_seconds:
                msg = f"Critical: Chaotic stampede movement detected! (Score: {panic_info.get('panic_score', 0.0):.2f})"
                alert_obj = self._create_alert(session_id, alert_type, "CRITICAL", "RED", msg, frame_number, people_count)
                new_alerts.append(alert_obj)
                self.last_alert_times[alert_type] = now
                self._maybe_send_email_alert(alert_obj, now)

        # Alert Condition 3: Sudden Rush Surge
        elif panic_info.get("is_rush", False):
            alert_type = "SUDDEN_RUSH"
            if (now - self.last_alert_times.get(alert_type, 0)) > self.cooldown_seconds:
                msg = "Warning: Sudden abnormal crowd rush surge detected!"
                alert_obj = self._create_alert(session_id, alert_type, "WARNING", "YELLOW", msg, frame_number, people_count)
                new_alerts.append(alert_obj)
                self.last_alert_times[alert_type] = now
                self._maybe_send_email_alert(alert_obj, now)

        # Alert Condition 4: General High Stampede Risk (RED)
        elif risk_level == "RED":
            alert_type = "STAMPEDE_RISK"
            if (now - self.last_alert_times.get(alert_type, 0)) > self.cooldown_seconds:
                msg = f"Critical risk score ({risk_score:.2f}) reached! High danger of crowd crush."
                alert_obj = self._create_alert(session_id, alert_type, "CRITICAL", "RED", msg, frame_number, people_count)
                new_alerts.append(alert_obj)
                self.last_alert_times[alert_type] = now
                self._maybe_send_email_alert(alert_obj, now)

        # Alert Condition 5: Moderate High Density Risk (YELLOW)
        elif risk_level == "YELLOW":
            alert_type = "HIGH_DENSITY"
            if (now - self.last_alert_times.get(alert_type, 0)) > self.cooldown_seconds:
                msg = f"Alert: Moderate crowd congestion detected (Risk score: {risk_score:.2f})."
                alert_obj = self._create_alert(session_id, alert_type, "WARNING", "YELLOW", msg, frame_number, people_count)
                new_alerts.append(alert_obj)
                self.last_alert_times[alert_type] = now
                # Optionally send email if critical/yellow alert emails enabled
                self._maybe_send_email_alert(alert_obj, now)

        return new_alerts

    def _maybe_send_email_alert(self, alert_dict: dict, now_ts: float):
        """Dispatches an asynchronous emergency email alert if email cooldown has elapsed."""
        alert_type = alert_dict.get("alert_type", "GENERAL")
        last_sent = self.last_email_times.get(alert_type, 0)

        if (now_ts - last_sent) >= self.email_cooldown_seconds:
            try:
                send_alert_email_async(alert_dict)
                self.last_email_times[alert_type] = now_ts
                print(f"[AlertManager] Emergency Alert Email dispatched for {alert_type}")
            except Exception as e:
                print(f"[AlertManager Warning] Could not dispatch alert email: {e}")

    def _create_alert(self, session_id, alert_type, severity, risk_level, message, frame_number, people_count=None):
        """Creates alert object and logs to database if session and app context exist."""
        now_dt = datetime.now(timezone.utc).replace(tzinfo=None)
        alert_dict = {
            "session_id": session_id or "Active",
            "alert_type": alert_type,
            "severity": severity,
            "risk_level": risk_level,
            "message": message,
            "frame_number": frame_number,
            "people_count": people_count or "N/A",
            "timestamp": now_dt.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Persist to database if inside Flask application context
        if has_app_context() and session_id is not None:
            try:
                db_alert = Alert(
                    session_id=session_id,
                    alert_type=alert_type,
                    severity=severity,
                    risk_level=risk_level,
                    message=message,
                    timestamp=now_dt
                )
                db.session.add(db_alert)

                # Also log a DetectionEvent
                event = DetectionEvent(
                    session_id=session_id,
                    event_type=alert_type.lower(),
                    confidence=0.85,
                    frame_number=frame_number,
                    description=message,
                    timestamp=now_dt
                )
                db.session.add(event)
                db.session.commit()
            except Exception:
                db.session.rollback()

        self.active_alerts.append(alert_dict)
        if len(self.active_alerts) > 20:
            self.active_alerts.pop(0)

        return alert_dict

    def reset(self):
        self.last_alert_times = {}
        self.last_email_times = {}
        self.active_alerts = []
