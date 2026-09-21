from flask import Blueprint, render_template, request, flash, redirect, url_for
from flask_login import login_required, current_user
from database.database import db

profile = Blueprint("profile", __name__)


@profile.route("/profile", methods=["GET", "POST"])
@login_required
def profile_page():
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        new_password = request.form.get("new_password", "")
        confirm_password = request.form.get("confirm_password", "")

        if full_name:
            current_user.full_name = full_name

        if new_password:
            if new_password != confirm_password:
                flash("Passwords do not match!", "danger")
                return redirect(url_for("profile.profile_page"))
            current_user.set_password(new_password)
            flash("Password updated successfully!", "success")

        db.session.commit()
        flash("Profile updated successfully!", "success")
        return redirect(url_for("profile.profile_page"))

    return render_template("profile.html", user=current_user)
