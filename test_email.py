"""
Email Diagnostic & Verification Script for Crowd Stampede Prediction System.
Tests SMTP connection and sends a test email to verify credentials.
"""
import sys
from config import Config
from utils.email_service import send_alert_email


def test_smtp_configuration():
    print("=" * 60)
    print("CROWD STAMPEDE PREDICTION SYSTEM - SMTP DIAGNOSTIC TOOL")
    print("=" * 60)
    print(f"MAIL_SERVER:     {Config.MAIL_SERVER}")
    print(f"MAIL_PORT:       {Config.MAIL_PORT}")
    print(f"MAIL_USE_TLS:    {Config.MAIL_USE_TLS}")
    print(f"MAIL_USERNAME:   {Config.MAIL_USERNAME}")
    print(f"MAIL_PASSWORD:   {'*' * len(Config.MAIL_PASSWORD) if Config.MAIL_PASSWORD else 'NOT SET'}")
    print(f"ALERT_RECIPIENT: {Config.ALERT_EMAIL_RECIPIENT}")
    print(f"ENABLE_ALERTS:   {Config.ENABLE_ALERT_EMAILS}")
    print("-" * 60)

    target_email = Config.ALERT_EMAIL_RECIPIENT or Config.MAIL_USERNAME
    print(f"Attempting to send Test Emergency Alert Email to: {target_email} ...")

    test_alert = {
        "alert_type": "STAMPEDE_RISK",
        "severity": "CRITICAL",
        "risk_level": "RED",
        "message": "This is a verified test alert from your Crowd Stampede Prediction System to confirm that automatic email notifications are working successfully!",
        "people_count": 48,
        "session_id": "TEST-01",
        "timestamp": "Now (Live Test)"
    }

    success = send_alert_email(test_alert, target_email)

    if success:
        print("\n[SUCCESS] Emergency Alert email sent successfully!")
        print(f"Please check your inbox at: {target_email}")
        return True
    else:
        print("\n[ERROR] Email sending failed. Check credentials above or network connection.")
        return False


if __name__ == "__main__":
    success = test_smtp_configuration()
    sys.exit(0 if success else 1)
