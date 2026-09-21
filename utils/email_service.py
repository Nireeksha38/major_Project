import os
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from datetime import datetime
from config import Config


def _send_smtp_direct(subject: str, recipient_email: str, html_body: str, text_body: str = ""):
    """
    Direct SMTP sender that works cleanly across both web requests and background video processing threads.
    Handles UTF-8 character encodings cleanly across all platforms.
    """
    server = None
    try:
        mail_server = Config.MAIL_SERVER
        mail_port = Config.MAIL_PORT
        mail_username = Config.MAIL_USERNAME
        mail_password = Config.MAIL_PASSWORD
        sender = Config.MAIL_DEFAULT_SENDER

        if not mail_username or not mail_password or "your-app-password" in mail_password:
            print(f"[Email Service] Email sending skipped: MAIL_USERNAME or MAIL_PASSWORD not configured.")
            return False

        if isinstance(sender, tuple):
            sender_display_name, sender_addr = sender
            sender_str = f"{Header(sender_display_name, 'utf-8').encode()} <{sender_addr}>"
        else:
            sender_str = str(sender)

        msg = MIMEMultipart("alternative")
        msg["Subject"] = Header(subject, "utf-8").encode()
        msg["From"] = sender_str
        msg["To"] = recipient_email

        if text_body:
            part1 = MIMEText(text_body, "plain", "utf-8")
            msg.attach(part1)

        if html_body:
            part2 = MIMEText(html_body, "html", "utf-8")
            msg.attach(part2)

        if Config.MAIL_USE_SSL:
            server = smtplib.SMTP_SSL(mail_server, mail_port, timeout=12)
        else:
            server = smtplib.SMTP(mail_server, mail_port, timeout=12)
            if Config.MAIL_USE_TLS:
                server.starttls()

        server.login(mail_username, mail_password)
        # Send via send_message to handle UTF-8 cleanly
        server.send_message(msg)
        server.quit()
        print(f"[Email Service] Successfully sent email '{subject}' to {recipient_email}")
        return True
    except smtplib.SMTPAuthenticationError as auth_err:
        print(f"[Email Service ERROR] SMTP Authentication failed: Bad credentials or expired App Password. ({auth_err})")
        return False
    except Exception as e:
        print(f"[Email Service ERROR] Failed to send email to {recipient_email}: {e}")
        return False
    finally:
        if server:
            try:
                server.close()
            except Exception:
                pass


def send_otp_email(receiver_email: str, otp: str):
    """
    Sends 6-digit OTP verification email for Password Reset.
    """
    subject = "Crowd Stampede AI - Password Reset OTP Code"
    
    text_body = f"""Hello,

Your One-Time Password (OTP) for resetting your Crowd Stampede Prediction System account password is:

{otp}

This OTP is valid for 5 minutes. If you did not request this, please disregard this email.

Security Operations Team
AI Crowd Stampede Prediction System
"""

    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f0f4f8; margin: 0; padding: 24px; color: #1e293b; }}
            .container {{ max-width: 520px; margin: 0 auto; background: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border: 1px solid #e2e8f0; }}
            .header {{ background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%); color: #ffffff; padding: 28px 24px; text-align: center; }}
            .header h1 {{ margin: 0; font-size: 20px; font-weight: 700; letter-spacing: -0.3px; }}
            .content {{ padding: 32px 28px; text-align: center; }}
            .otp-box {{ display: inline-block; font-size: 32px; font-weight: 800; letter-spacing: 8px; color: #0284c7; background: #f0f9ff; border: 2px dashed #38bdf8; border-radius: 10px; padding: 14px 28px; margin: 20px 0; }}
            .note {{ font-size: 13px; color: #64748b; line-height: 1.5; margin-top: 16px; }}
            .footer {{ background: #f8fafc; padding: 16px; text-align: center; font-size: 12px; color: #94a3b8; border-top: 1px solid #e2e8f0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>AI Crowd Surveillance System</h1>
                <p style="margin: 6px 0 0; font-size: 13px; opacity: 0.9;">Password Reset Request</p>
            </div>
            <div class="content">
                <p style="font-size: 15px; margin: 0 0 12px; color: #334155;">You requested a password reset code. Use the OTP below to proceed:</p>
                <div class="otp-box">{otp}</div>
                <p class="note">This verification code expires in <strong>5 minutes</strong>.<br>For security reasons, do not share this code with anyone.</p>
            </div>
            <div class="footer">
                Crowd Stampede & Surveillance Early Warning System &bull; Automated Security Alert
            </div>
        </div>
    </body>
    </html>
    """

    return _send_smtp_direct(subject, receiver_email, html_body, text_body)


def send_alert_email(alert_data: dict, recipient_email: str = None):
    """
    Sends emergency surveillance risk notification email (Stampede, Panic, Fall, High Density).
    """
    if not Config.ENABLE_ALERT_EMAILS:
        return False

    recipient = recipient_email or Config.ALERT_EMAIL_RECIPIENT or Config.MAIL_USERNAME
    if not recipient:
        return False

    alert_type = alert_data.get("alert_type", "STAMPEDE_RISK").replace("_", " ").title()
    severity = alert_data.get("severity", "CRITICAL")
    risk_level = alert_data.get("risk_level", "RED")
    message = alert_data.get("message", "Immediate attention required in crowd surveillance area.")
    timestamp = alert_data.get("timestamp", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))
    people_count = alert_data.get("people_count", "N/A")
    session_id = alert_data.get("session_id", "Live Stream")

    badge_color = "#dc2626" if risk_level == "RED" or severity == "CRITICAL" else "#d97706"
    subject = f"[EMERGENCY ALERT - {severity}] {alert_type} Detected - Crowd Safety System"

    text_body = f"""*** SURVEILLANCE EMERGENCY ALERT ***
Severity: {severity} ({risk_level} RISK)
Incident Type: {alert_type}
Timestamp: {timestamp}
Surveillance Stream: Session #{session_id}
Observed Crowd Count: {people_count}

Details:
{message}

RECOMMENDED ACTION:
1. Dispatch on-site safety response personnel immediately.
2. Verify CCTV / Surveillance live stream on dashboard.
3. Initiate crowd dispersion and exit clearance protocols if density persists.

AI Crowd Stampede Early Warning System
"""

    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f8fafc; margin: 0; padding: 20px; color: #0f172a; }}
            .card {{ max-width: 580px; margin: 0 auto; background: #ffffff; border-radius: 12px; overflow: hidden; border: 1px solid #e2e8f0; box-shadow: 0 10px 25px rgba(0,0,0,0.1); }}
            .header {{ background: {badge_color}; color: #ffffff; padding: 24px; text-align: center; }}
            .header h2 {{ margin: 0 0 6px; font-size: 22px; font-weight: 800; letter-spacing: -0.5px; text-transform: uppercase; }}
            .header p {{ margin: 0; font-size: 14px; opacity: 0.95; font-weight: 500; }}
            .body {{ padding: 24px 28px; }}
            .alert-pill {{ display: inline-block; background: #fee2e2; color: #b91c1c; font-weight: 700; font-size: 13px; padding: 4px 12px; border-radius: 9999px; border: 1px solid #fca5a5; margin-bottom: 16px; }}
            .info-table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
            .info-table td {{ padding: 10px 12px; font-size: 14px; border-bottom: 1px solid #f1f5f9; }}
            .info-table td.label {{ color: #64748b; font-weight: 600; width: 38%; }}
            .info-table td.value {{ color: #0f172a; font-weight: 700; }}
            .msg-box {{ background: #fef2f2; border-left: 4px solid #ef4444; padding: 14px 18px; border-radius: 0 8px 8px 0; margin: 18px 0; color: #991b1b; font-size: 14px; line-height: 1.5; }}
            .actions {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin-top: 20px; font-size: 13px; color: #334155; }}
            .actions h4 {{ margin: 0 0 8px; color: #0f172a; font-size: 14px; }}
            .actions ul {{ margin: 0; padding-left: 20px; }}
            .actions li {{ margin-bottom: 4px; }}
            .footer {{ text-align: center; padding: 16px; font-size: 12px; color: #94a3b8; background: #f8fafc; border-top: 1px solid #e2e8f0; }}
        </style>
    </head>
    <body>
        <div class="card">
            <div class="header">
                <h2>SURVEILLANCE EMERGENCY ALERT</h2>
                <p>AI Crowd Stampede Prediction & Early Warning System</p>
            </div>
            <div class="body">
                <span class="alert-pill">SEVERITY: {severity} &bull; RISK: {risk_level}</span>
                
                <div class="msg-box">
                    <strong>Incident:</strong> {message}
                </div>

                <table class="info-table">
                    <tr>
                        <td class="label">Alert Type</td>
                        <td class="value">{alert_type}</td>
                    </tr>
                    <tr>
                        <td class="label">Detection Timestamp</td>
                        <td class="value">{timestamp}</td>
                    </tr>
                    <tr>
                        <td class="label">Surveillance Stream</td>
                        <td class="value">Session #{session_id}</td>
                    </tr>
                    <tr>
                        <td class="label">Observed Crowd Count</td>
                        <td class="value">{people_count}</td>
                    </tr>
                </table>

                <div class="actions">
                    <h4>Recommended Operational Action:</h4>
                    <ul>
                        <li>Dispatch ground safety personnel & crowd marshals immediately.</li>
                        <li>Open emergency egress gates to relieve localized crowd bottleneck.</li>
                        <li>Verify live CCTV / Camera feed on the system monitoring dashboard.</li>
                    </ul>
                </div>
            </div>
            <div class="footer">
                Automated notification generated by AI Surveillance Risk Engine &bull; Stampede Defense System
            </div>
        </div>
    </body>
    </html>
    """

    return _send_smtp_direct(subject, recipient, html_body, text_body)


def send_alert_email_async(alert_data: dict, recipient_email: str = None):
    """
    Fires send_alert_email in a background daemon thread so computer vision & streaming never block.
    """
    thread = threading.Thread(
        target=send_alert_email,
        args=(alert_data, recipient_email),
        daemon=True
    )
    thread.start()
    return thread