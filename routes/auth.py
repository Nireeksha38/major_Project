from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user
from models.user import User
from database.database import db
from utils.email_service import send_otp_email
import random
from datetime import datetime, timedelta, timezone

auth = Blueprint("auth", __name__)


# ================= SIGNUP =================
@auth.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not full_name or not email or not password:
            flash("Please fill in all required fields.", "warning")
            return redirect(url_for("auth.signup"))

        if password != confirm_password:
            flash("Passwords do not match!", "danger")
            return redirect(url_for("auth.signup"))

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("Email already registered!", "warning")
            return redirect(url_for("auth.signup"))

        user = User(
            full_name=full_name,
            email=email,
            username=email.split("@")[0]
        )
        user.set_password(password)

        db.session.add(user)
        db.session.commit()

        flash("Account created successfully! Please log in.", "success")
        return redirect(url_for("auth.login"))

    return render_template("signup.html")


# ================= LOGIN =================
@auth.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        login_input = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        # Look up by email or username
        user = User.query.filter(
            (User.email == login_input) | (User.username == login_input)
        ).first()

        if user and user.check_password(password):
            login_user(user)
            flash("Login Successful! Welcome to Crowd Surveillance Dashboard.", "success")
            return redirect(url_for("dashboard.dashboard_home"))
        else:
            flash("Invalid Username/Email or Password!", "danger")

    return render_template("login.html")


# ================= LOGOUT =================
@auth.route("/logout")
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("auth.login"))


# ================= FORGOT PASSWORD =================
@auth.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        user = User.query.filter_by(email=email).first()

        if not user:
            flash("Email is not registered!", "danger")
            return redirect(url_for("auth.forgot_password"))

        # Generate 6-digit OTP
        otp = str(random.randint(100000, 999999))
        user.otp = otp
        user.otp_expiry = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=5)
        db.session.commit()

        # Send OTP email
        try:
            success = send_otp_email(email, otp)
            if success:
                flash("OTP sent successfully to your email!", "success")
            else:
                flash(f"Could not send email automatically. For testing, OTP is: {otp}", "warning")
        except Exception as e:
            flash(f"Could not send email automatically ({str(e)}). For local testing, OTP is: {otp}", "warning")

        return redirect(url_for("auth.verify_otp", email=email))

    return render_template("forgot_password.html")


# ================= VERIFY OTP =================
@auth.route("/verify-otp", methods=["GET", "POST"])
def verify_otp():
    email = request.args.get("email", "")

    if request.method == "POST":
        entered_otp = request.form.get("otp", "").strip()
        user = User.query.filter_by(email=email).first()

        if not user:
            flash("User not found!", "danger")
            return redirect(url_for("auth.forgot_password"))

        if user.otp != entered_otp:
            flash("Invalid OTP! Please check and try again.", "danger")
            return render_template("otp_verification.html", email=email)

        now_naive = datetime.now(timezone.utc).replace(tzinfo=None)
        if user.otp_expiry and now_naive > user.otp_expiry:
            flash("OTP has expired! Please request a new OTP.", "danger")
            return render_template("otp_verification.html", email=email)

        return redirect(url_for("auth.reset_password", email=email))

    return render_template("otp_verification.html", email=email)


# ================= RESET PASSWORD =================
@auth.route("/reset-password", methods=["GET", "POST"])
def reset_password():
    email = request.args.get("email", "")

    if request.method == "POST":
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if password != confirm_password:
            flash("Passwords do not match!", "danger")
            return redirect(url_for("auth.reset_password", email=email))

        user = User.query.filter_by(email=email).first()
        if not user:
            flash("User not found!", "danger")
            return redirect(url_for("auth.forgot_password"))

        user.set_password(password)
        user.otp = None
        user.otp_expiry = None
        db.session.commit()

        flash("Password updated successfully! Please login with your new password.", "success")
        return redirect(url_for("auth.login"))

    return render_template("reset_password.html", email=email)


# ================= PROFILE =================
@auth.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        new_password = request.form.get("new_password", "")

        if full_name:
            current_user.full_name = full_name
        if new_password:
            current_user.set_password(new_password)

        db.session.commit()
        flash("Profile updated successfully!", "success")

    return render_template("profile.html", user=current_user)