from flask import Blueprint, jsonify, redirect, render_template, request, session, url_for, current_app

from app.config import APP_USERNAME, MODEL_PATH, SCALER_PATH, THRESHOLD
from app.extensions import limiter
from app.persistence.storage import get_user_by_username
from app.routes import wants_json
from app.services.scoring_service import sanitize_text
from feature_engineering import MODEL_FEATURES

auth_bp = Blueprint("auth", __name__)


@auth_bp.get("/")
def home():
    return render_template(
        "index.html",
        authenticated=bool(session.get("authenticated")),
        username=session.get("username", ""),
        default_username=APP_USERNAME,
        threshold=THRESHOLD,
    )


@auth_bp.post("/login")
@limiter.limit("5 per minute")
def login():
    payload = request.get_json(silent=True) if request.is_json else request.form
    username = sanitize_text(payload.get("username") if payload else None, "")
    password = sanitize_text(payload.get("password") if payload else None, "")

    user = get_user_by_username(username)
    if user:
        bcrypt = current_app.bcrypt
        if bcrypt.check_password_hash(user["password_hash"], password):
            session["authenticated"] = True
            session["username"] = username
            if wants_json():
                return jsonify({"status": "ok", "username": username})
            return redirect(url_for("auth.home"))

    # Generic error to prevent user enumeration
    if wants_json():
        return jsonify({"error": "Invalid username or password."}), 401
    return render_template(
        "index.html",
        authenticated=False,
        username="",
        default_username=APP_USERNAME,
        threshold=THRESHOLD,
        login_error="Invalid username or password.",
    ), 401


@auth_bp.post("/logout")
def logout():
    session.clear()
    if wants_json():
        return jsonify({"status": "ok"})
    return redirect(url_for("auth.home"))


@auth_bp.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_file": MODEL_PATH.name,
        "scaler_file": SCALER_PATH.name,
        "threshold": THRESHOLD,
        "feature_count": len(MODEL_FEATURES),
        "features": [
            "Transaction Monitoring",
            "Machine Learning Prediction",
            "Real-Time Alerts",
            "User Authentication & Verification",
            "Risk Scoring System",
            "Behavior Analysis",
            "Data Preprocessing",
            "Dashboard & Reports",
            "CSV Batch Detection",
        ],
    })
    