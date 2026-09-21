from flask import Blueprint, render_template
from flask_login import login_required

architecture = Blueprint("architecture", __name__)


@architecture.route("/architecture")
@login_required
def architecture_page():
    return render_template("architecture.html")
