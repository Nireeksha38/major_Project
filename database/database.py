import os
import sqlite3
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def migrate_db(db_path):
    """
    Safely inspects SQLite database schema and adds any missing columns
    so existing databases are automatically migrated without losing data.
    """
    if not os.path.exists(db_path):
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Helper to check and add column if missing
        def ensure_column(table_name, col_name, col_type):
            cursor.execute(f"PRAGMA table_info({table_name})")
            existing_cols = [row[1] for row in cursor.fetchall()]
            if existing_cols and col_name not in existing_cols:
                try:
                    cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}")
                except Exception as e:
                    print(f"[DB Migration Warning] {table_name}.{col_name}: {e}")

        # 1. Check users table
        ensure_column("users", "username", "VARCHAR(80)")
        ensure_column("users", "role", "VARCHAR(20) DEFAULT 'operator'")
        ensure_column("users", "otp", "VARCHAR(10)")
        ensure_column("users", "otp_expiry", "DATETIME")

        # 2. Check alerts table
        ensure_column("alerts", "session_id", "INTEGER")
        ensure_column("alerts", "severity", "VARCHAR(20) DEFAULT 'WARNING'")
        ensure_column("alerts", "risk_level", "VARCHAR(20) DEFAULT 'YELLOW'")
        ensure_column("alerts", "timestamp", "DATETIME DEFAULT CURRENT_TIMESTAMP")
        ensure_column("alerts", "acknowledged", "BOOLEAN DEFAULT 0")
        ensure_column("alerts", "acknowledged_at", "DATETIME")
        ensure_column("alerts", "acknowledged_by", "VARCHAR(100)")

        # If timestamp was added and created_at exists, backfill values
        try:
            cursor.execute("PRAGMA table_info(alerts)")
            alert_cols = [row[1] for row in cursor.fetchall()]
            if "timestamp" in alert_cols and "created_at" in alert_cols:
                cursor.execute("UPDATE alerts SET timestamp = created_at WHERE timestamp IS NULL")
        except Exception:
            pass

        # 3. Check crowd_data table
        ensure_column("crowd_data", "session_id", "INTEGER")
        ensure_column("crowd_data", "relative_speed", "FLOAT DEFAULT 0.0")
        ensure_column("crowd_data", "motion_entropy", "FLOAT DEFAULT 0.0")
        ensure_column("crowd_data", "motion_intensity", "FLOAT DEFAULT 0.0")
        ensure_column("crowd_data", "panic_score", "FLOAT DEFAULT 0.0")
        ensure_column("crowd_data", "risk_score", "FLOAT DEFAULT 0.0")
        ensure_column("crowd_data", "falls_detected", "INTEGER DEFAULT 0")
        ensure_column("crowd_data", "frame_number", "INTEGER DEFAULT 0")

        # 4. Check reports table
        ensure_column("reports", "session_id", "INTEGER")
        ensure_column("reports", "file_path", "VARCHAR(255)")
        ensure_column("reports", "summary", "TEXT")

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB Migration Exception] {e}")