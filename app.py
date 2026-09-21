import os
from flask import Flask, render_template
from flask_login import LoginManager
from config import Config
from database.database import db, migrate_db
from extensions import mail
from models.user import User

# Import Blueprints
from routes.auth import auth
from routes.dashboard import dashboard
from routes.detection import detection
from routes.history import history
from routes.model_performance import model_perf
from routes.architecture import architecture
from routes.reports import reports
from routes.profile import profile
from routes.api import api

# Create Flask Application
app = Flask(__name__)

# Load Configuration
app.config.from_object(Config)

# Initialize Extensions
db.init_app(app)
mail.init_app(app)

# Secret Key
app.secret_key = app.config["SECRET_KEY"]

# -------------------------------
# Flask Login Configuration
# -------------------------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "auth.login"
login_manager.login_message_category = "warning"


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# -------------------------------
# Register Blueprints
# -------------------------------
app.register_blueprint(auth)
app.register_blueprint(dashboard)
app.register_blueprint(detection)
app.register_blueprint(history)
app.register_blueprint(model_perf)
app.register_blueprint(architecture)
app.register_blueprint(reports)
app.register_blueprint(profile)
app.register_blueprint(api)


# -------------------------------
# Home & Error Handlers
# -------------------------------
@app.route("/")
def home():
    return render_template("home.html")


@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", error_code=404, message="Page not found"), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("error.html", error_code=500, message="Internal Server Error"), 500


# -------------------------------
# Initialize Database & Seed Demo User
# -------------------------------
def initialize_database():
    db_file_path = os.path.join(Config.BASE_DIR, "database.db")
    migrate_db(db_file_path)

    with app.app_context():
        db.create_all()
        # Seed default Demo Account if it does not exist
        try:
            demo_user = User.query.filter(
                (User.email == Config.DEMO_EMAIL) | (User.username == Config.DEMO_USERNAME)
            ).first()

            if not demo_user:
                demo_user = User(
                    full_name=Config.DEMO_NAME,
                    username=Config.DEMO_USERNAME,
                    email=Config.DEMO_EMAIL,
                    role="admin"
                )
                demo_user.set_password(Config.DEMO_PASSWORD)
                db.session.add(demo_user)
                db.session.commit()
                print(f"[Database] Default Demo account created: '{Config.DEMO_USERNAME}' with password '{Config.DEMO_PASSWORD}'")
        except Exception as e:
            print(f"[Database Init Warning] {e}")


initialize_database()

# -------------------------------
# Run Application
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False, threaded=True)