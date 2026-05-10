"""
app.py  – Fraud Shield Flask application
"""

import atexit
import csv
import io
import json
import logging
import math
import os
import uuid
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path

import joblib
import pandas as pd
from flask import (Flask, Response, jsonify, redirect, render_template,
                   request, session, url_for)

from feature_engineering import engineer_single_transaction, MODEL_FEATURES

# ── App setup ────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "change-this-demo-secret")

BASE_DIR    = Path(__file__).resolve().parent
MODEL_PATH  = BASE_DIR / "fraud_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
STORE_PATH  = BASE_DIR / "data_store.json"    # persistence file

THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.5"))

APP_USERNAME = os.getenv("APP_USERNAME", "admin")
APP_PASSWORD = os.getenv("APP_PASSWORD", "admin123")

REQUIRED_FIELDS = [
    "transaction_date",    # YYYY-MM-DD
    "transaction_time",    # HH:MM:SS
    "amount",
    "merchant_category",
    "country",
    "channel",
]
OPTIONAL_FIELDS = ["customer_id", "card_last4", "merchant", "device_id"]

MAX_RECORDS  = 250
MAX_CSV_ROWS = 500

# ── In-memory stores ─────────────────────────────────────────────────────────

TRANSACTIONS:     list  = []
ALERTS:           list  = []
BEHAVIOR_PROFILES: dict = {}

# ── Persistence ──────────────────────────────────────────────────────────────

def _profile_to_dict(p: dict) -> dict:
    """Make a behavior profile JSON-serialisable (sets → sorted lists)."""
    return {
        **p,
        "devices":             sorted(p["devices"]),
        "countries":           sorted(p["countries"]),
        "merchant_categories": sorted(p["merchant_categories"]),
    }


def _dict_to_profile(d: dict) -> dict:
    """Restore a behavior profile from a loaded dict (lists → sets)."""
    return {
        **d,
        "devices":             set(d.get("devices", [])),
        "countries":           set(d.get("countries", [])),
        "merchant_categories": set(d.get("merchant_categories", [])),
    }


def save_state() -> None:
    """Persist TRANSACTIONS, ALERTS, and BEHAVIOR_PROFILES to disk."""
    try:
        state = {
            "transactions":      TRANSACTIONS,
            "alerts":            ALERTS,
            "behavior_profiles": {
                k: _profile_to_dict(v) for k, v in BEHAVIOR_PROFILES.items()
            },
        }
        STORE_PATH.write_text(
            json.dumps(state, indent=2, default=str), encoding="utf-8"
        )
    except Exception as exc:
        logger.warning("Could not save state: %s", exc)


def load_state() -> None:
    """Load persisted state from disk on startup."""
    if not STORE_PATH.exists():
        return
    try:
        state = json.loads(STORE_PATH.read_text(encoding="utf-8"))
        TRANSACTIONS[:] = state.get("transactions", [])
        ALERTS[:]       = state.get("alerts", [])
        BEHAVIOR_PROFILES.update(
            {k: _dict_to_profile(v)
             for k, v in state.get("behavior_profiles", {}).items()}
        )
        logger.info(
            "Loaded %d transactions, %d alerts, %d profiles.",
            len(TRANSACTIONS), len(ALERTS), len(BEHAVIOR_PROFILES),
        )
    except Exception as exc:
        logger.warning("Could not load state: %s", exc)


# Load on startup; save when the process exits cleanly.
load_state()
atexit.register(save_state)

# ── Model loading ─────────────────────────────────────────────────────────────

def load_artifacts():
    missing = [p.name for p in (MODEL_PATH, SCALER_PATH) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required file(s): {', '.join(missing)}")
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)


model, scaler = load_artifacts()

# ── Auth helpers ─────────────────────────────────────────────────────────────

def require_auth(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not session.get("authenticated"):
            return jsonify({"error": "Authentication required."}), 401
        return func(*args, **kwargs)
    return wrapper


def wants_json() -> bool:
    return request.is_json or "application/json" in request.headers.get("Accept", "")


# ── Utilities ─────────────────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sanitize_text(value: object, default: str) -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    cleaned = str(value).strip()
    return cleaned if cleaned else default


# ── Behavior helpers ──────────────────────────────────────────────────────────

def _count_recent_both(profile: dict) -> tuple[int, int]:
    """
    Count transactions in the last 1 h and 24 h in a **single pass**,
    avoiding repeated datetime parsing that the old two-call approach caused.

    Timestamps are stored as UTC unix floats for O(1) comparison.
    """
    timestamps = profile.get("recent_timestamps", [])
    if not timestamps:
        return 0, 0

    now        = datetime.now(timezone.utc).timestamp()
    cutoff_1h  = now - 3600
    cutoff_24h = now - 86400

    count_1h = count_24h = 0
    for ts in timestamps:
        if ts >= cutoff_24h:
            count_24h += 1
            if ts >= cutoff_1h:
                count_1h += 1
    return count_1h, count_24h


def get_behavior_profile(customer_id: str) -> dict:
    return BEHAVIOR_PROFILES.setdefault(customer_id, {
        "transaction_count":    0,
        "amount_total":         0.0,
        "amount_avg":           0.0,
        "amount_max":           0.0,
        "recent_timestamps":    [],   # list of UTC unix timestamps (floats)
        "devices":              set(),
        "countries":            set(),
        "merchant_categories":  set(),
    })


def analyze_behavior(txn: dict, metadata: dict, profile: dict) -> dict:
    """
    Compute behavioural risk signals for the current transaction.

    The returned dict is passed to engineer_single_transaction, so the keys
    must satisfy the contract documented in feature_engineering.py:
        amount_avg, is_new_country, is_new_category,
        txn_count_last_1h, txn_count_last_24h
    """
    amount  = txn["amount"]
    signals = []
    behavior_points = 0

    txn_1h, txn_24h = _count_recent_both(profile)

    if profile["transaction_count"] == 0:
        signals.append("New customer profile")
        behavior_points += 4
        # Use current amount as the "avg" so feature_engineering produces
        # a neutral amount_ratio of 1.0 rather than using 0 as the average.
        effective_avg = amount
    else:
        effective_avg = profile["amount_avg"]

        if effective_avg > 0 and amount > max(effective_avg * 3, 500):
            signals.append("Amount much higher than customer average")
            behavior_points += 14
        if metadata.get("device_id") not in profile["devices"]:
            signals.append("New device for this customer")
            behavior_points += 10
        if txn["country"] not in profile["countries"]:
            signals.append("New country for this customer")
            behavior_points += 12
        if txn["merchant_category"] not in profile["merchant_categories"]:
            signals.append("New merchant category for this customer")
            behavior_points += 6
        if txn_1h >= 3:
            signals.append(f"Rapid transactions: {txn_1h} in last hour")
            behavior_points += 8

    if amount >= 1000:
        signals.append("High transaction amount")
        behavior_points += 10

    try:
        hour = int(str(txn.get("transaction_time", "12:00:00")).split(":")[0])
        if 0 <= hour <= 5:
            signals.append("Night-time transaction (00:00–05:59)")
            behavior_points += 6
    except (ValueError, AttributeError, IndexError):
        pass

    if not signals:
        signals.append("Behavior matches known customer pattern")

    return {
        "customer_id":              metadata["customer_id"],
        "signals":                  signals,
        "behavior_points":          behavior_points,
        "profile_transaction_count": profile["transaction_count"],
        "txn_count_last_1h":        txn_1h,
        "txn_count_last_24h":       txn_24h,
        "amount_avg":               effective_avg,      # neutral on first txn
        "is_new_country":           int(txn["country"] not in profile["countries"]),
        "is_new_category":          int(txn["merchant_category"] not in profile["merchant_categories"]),
    }


def update_behavior_profile(txn: dict, metadata: dict) -> None:
    """Called *after* scoring so the current transaction doesn't bias its own score."""
    profile = get_behavior_profile(metadata["customer_id"])
    amount  = txn["amount"]
    count   = profile["transaction_count"] + 1

    profile["transaction_count"] = count
    profile["amount_total"]      += amount
    profile["amount_avg"]        = profile["amount_total"] / count
    profile["amount_max"]        = max(profile["amount_max"], amount)

    profile["recent_timestamps"].append(datetime.now(timezone.utc).timestamp())
    if len(profile["recent_timestamps"]) > 200:
        profile["recent_timestamps"] = profile["recent_timestamps"][-200:]

    profile["devices"].add(metadata.get("device_id", "unknown"))
    profile["countries"].add(txn["country"])
    profile["merchant_categories"].add(txn["merchant_category"])


# ── ML helpers ─────────────────────────────────────────────────────────────────

def preprocess_transaction(txn: dict, behavior: dict):
    raw_features = engineer_single_transaction(txn, behavior)
    input_df     = pd.DataFrame([raw_features], columns=MODEL_FEATURES)
    scaled       = scaler.transform(input_df)
    prepared_df  = pd.DataFrame(scaled, columns=MODEL_FEATURES)

    return prepared_df, {
        "feature_count":      len(MODEL_FEATURES),
        "hour":               raw_features["hour"],
        "is_night":           raw_features["is_night"],
        "is_weekend":         raw_features["is_weekend"],
        "amount_ratio":       round(raw_features["amount_ratio"], 4),
        "txn_count_last_24h": raw_features["txn_count_last_24h"],
    }


def score_ml(prepared_df) -> tuple[float, int]:
    probability = float(model.predict_proba(prepared_df)[0, 1])
    prediction  = int(probability >= THRESHOLD)
    return probability, prediction


def calculate_risk(probability: float, prediction: int, behavior: dict) -> tuple[int, str]:
    ml_component       = probability * 72
    model_flag         = 10 if prediction else 0
    behavior_component = min(behavior["behavior_points"], 28)
    risk_score         = min(100, round(ml_component + model_flag + behavior_component))

    if risk_score >= 85:
        risk_level = "Critical"
    elif risk_score >= 65:
        risk_level = "High"
    elif risk_score >= 35:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return risk_score, risk_level


def create_alert_if_needed(record: dict):
    if record["prediction"] == 0 and record["risk_score"] < 65:
        return None

    alert = {
        "id":             str(uuid.uuid4())[:8],
        "transaction_id": record["id"],
        "created_at":     record["created_at"],
        "severity":       record["risk_level"],
        "message": (
            f"{record['risk_level']} risk transaction for "
            f"{record['metadata']['customer_id']} at {record['metadata']['merchant']}"
        ),
        "risk_score":   record["risk_score"],
        "acknowledged": False,
    }
    ALERTS.insert(0, alert)
    del ALERTS[MAX_RECORDS:]
    return alert


# ── Transaction pipeline ─────────────────────────────────────────────────────

def parse_transaction(payload: object) -> tuple[dict, dict]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")

    try:
        amount = float(payload["amount"])
    except (TypeError, ValueError):
        raise ValueError("amount must be a numeric value.")
    if not math.isfinite(amount) or amount < 0:
        raise ValueError("amount must be a non-negative finite number.")

    txn = {
        "transaction_date": sanitize_text(payload.get("transaction_date"), "2024-01-01"),
        "transaction_time": sanitize_text(payload.get("transaction_time"), "12:00:00"),
        "amount":           amount,
        "merchant_category": sanitize_text(payload.get("merchant_category"), "other"),
        "country":          sanitize_text(payload.get("country"), "US"),
        "channel":          sanitize_text(payload.get("channel"),  "web"),
    }
    metadata = {
        "customer_id":     sanitize_text(payload.get("customer_id"),  "CUST-DEMO"),
        "card_last4":      sanitize_text(payload.get("card_last4"),   "0000"),
        "merchant":        sanitize_text(payload.get("merchant"),     "Unknown Merchant"),
        "merchant_category": txn["merchant_category"],
        "country":         txn["country"],
        "channel":         txn["channel"],
        "device_id":       sanitize_text(payload.get("device_id"),    "web-demo"),
    }
    return txn, metadata


def monitor_transaction(payload: object) -> tuple[dict, object]:
    """
    Full pipeline:
        parse → behavior analysis → preprocess → ML score →
        risk score → alert → update profile → persist
    """
    txn, metadata = parse_transaction(payload)

    # 1. Read profile BEFORE updating it, so the current transaction doesn't
    #    skew its own behavioral features.
    profile  = get_behavior_profile(metadata["customer_id"])
    behavior = analyze_behavior(txn, metadata, profile)

    # 2. Preprocess and score
    prepared_df, preprocessing = preprocess_transaction(txn, behavior)
    probability, prediction    = score_ml(prepared_df)
    risk_score, risk_level     = calculate_risk(probability, prediction, behavior)

    record = {
        "id":                str(uuid.uuid4())[:8],
        "created_at":        now_iso(),
        "operator":          session.get("username", "api"),
        "metadata":          metadata,
        "amount":            txn["amount"],
        "transaction_date":  txn["transaction_date"],
        "transaction_time":  txn["transaction_time"],
        "fraud_probability": round(probability, 6),
        "prediction":        prediction,
        "label":             "Fraud" if prediction else "Normal",
        "threshold":         THRESHOLD,
        "risk_score":        risk_score,
        "risk_level":        risk_level,
        "behavior":          behavior,
        "preprocessing":     preprocessing,
        "verification_status": "Pending Review" if risk_score >= 65 else "Auto Cleared",
        "verification_note": "",
        "verified_by":       "",
        "verified_at":       "",
    }

    alert = create_alert_if_needed(record)

    # 3. Update profile AFTER scoring
    update_behavior_profile(txn, metadata)

    TRANSACTIONS.insert(0, record)
    del TRANSACTIONS[MAX_RECORDS:]

    # 4. Persist to disk
    save_state()

    return record, alert


# ── Dashboard helper ──────────────────────────────────────────────────────────

def dashboard_payload() -> dict:
    total          = len(TRANSACTIONS)
    high_risk      = sum(1 for t in TRANSACTIONS if t["risk_score"] >= 65)
    fraud_preds    = sum(1 for t in TRANSACTIONS if t["prediction"] == 1)
    verified       = sum(1 for t in TRANSACTIONS if t["verification_status"] != "Pending Review")
    average_risk   = round(sum(t["risk_score"] for t in TRANSACTIONS) / total, 1) if total else 0
    open_alerts    = sum(1 for a in ALERTS if not a["acknowledged"])

    return {
        "stats": {
            "total_transactions":    total,
            "fraud_predictions":     fraud_preds,
            "high_risk_transactions": high_risk,
            "open_alerts":           open_alerts,
            "verified_transactions": verified,
            "average_risk":          average_risk,
        },
        "transactions": TRANSACTIONS[:25],
        "alerts":       ALERTS[:25],
        "profiles": {
            cid: {
                "transaction_count": p["transaction_count"],
                "amount_average":    round(p.get("amount_avg", 0), 2),
                "amount_max":        round(p["amount_max"], 2),
                "devices":           sorted(p["devices"]),
                "countries":         sorted(p["countries"]),
            }
            for cid, p in BEHAVIOR_PROFILES.items()
        },
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def home():
    return render_template(
        "index.html",
        authenticated=bool(session.get("authenticated")),
        username=session.get("username", ""),
        default_username=APP_USERNAME,
        threshold=THRESHOLD,
    )


@app.post("/login")
def login():
    payload  = request.get_json(silent=True) if request.is_json else request.form
    username = sanitize_text(payload.get("username") if payload else None, "")
    password = sanitize_text(payload.get("password") if payload else None, "")

    if username == APP_USERNAME and password == APP_PASSWORD:
        session["authenticated"] = True
        session["username"]      = username
        if wants_json():
            return jsonify({"status": "ok", "username": username})
        return redirect(url_for("home"))

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


@app.post("/logout")
def logout():
    session.clear()
    if wants_json():
        return jsonify({"status": "ok"})
    return redirect(url_for("home"))


@app.get("/health")
def health():
    return jsonify({
        "status":        "ok",
        "model_file":    MODEL_PATH.name,
        "scaler_file":   SCALER_PATH.name,
        "threshold":     THRESHOLD,
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


@app.get("/api/dashboard")
@require_auth
def api_dashboard():
    return jsonify(dashboard_payload())


@app.get("/api/transactions")
@require_auth
def api_transactions():
    return jsonify({"transactions": TRANSACTIONS})


@app.get("/api/alerts")
@require_auth
def api_alerts():
    return jsonify({"alerts": ALERTS})


@app.post("/api/alerts/<alert_id>/acknowledge")
@require_auth
def api_acknowledge_alert(alert_id: str):
    for alert in ALERTS:
        if alert["id"] == alert_id:
            alert["acknowledged"] = True
            save_state()
            return jsonify(alert)
    return jsonify({"error": "Alert not found."}), 404


@app.post("/api/transactions/<transaction_id>/verify")
@require_auth
def api_verify_transaction(transaction_id: str):
    payload = request.get_json(silent=True) or {}
    status  = sanitize_text(payload.get("status"), "Reviewed")
    note    = sanitize_text(payload.get("note"),   "")
    allowed = {"Approved", "Blocked", "Reviewed", "False Positive"}
    if status not in allowed:
        return jsonify({"error": f"Status must be one of: {', '.join(sorted(allowed))}."}), 400

    for record in TRANSACTIONS:
        if record["id"] == transaction_id:
            record["verification_status"] = status
            record["verification_note"]   = note
            record["verified_by"]         = session.get("username", "")
            record["verified_at"]         = now_iso()
            for alert in ALERTS:
                if alert["transaction_id"] == transaction_id:
                    alert["acknowledged"] = True
            save_state()
            return jsonify(record)

    return jsonify({"error": "Transaction not found."}), 404


@app.get("/api/sample-csv")
@require_auth
def api_sample_csv():
    sample_path = BASE_DIR / "sample_upload.csv"
    if not sample_path.exists():
        return jsonify({"error": "sample_upload.csv was not found."}), 404
    return Response(
        sample_path.read_text(encoding="utf-8"),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=sample_upload.csv"},
    )


@app.post("/api/upload-csv")
@require_auth
def api_upload_csv():
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"error": "Upload a CSV file first."}), 400

    try:
        df = pd.read_csv(upload)
    except Exception as exc:
        return jsonify({"error": f"Could not read CSV file: {exc}"}), 400

    if df.empty:
        return jsonify({"error": "CSV file has no transaction rows."}), 400

    if len(df) > MAX_CSV_ROWS:
        return jsonify({
            "error": f"CSV file contains {len(df)} rows. Limit is {MAX_CSV_ROWS} rows."
        }), 400

    missing = [f for f in REQUIRED_FIELDS if f not in df.columns]
    if missing:
        return jsonify({
            "error":            f"CSV is missing required column(s): {', '.join(missing)}.",
            "required_columns": REQUIRED_FIELDS,
            "optional_columns": OPTIONAL_FIELDS,
        }), 400

    records, errors, alerts_created = [], [], 0

    for index, row in df.iterrows():
        payload = {field: row[field] for field in REQUIRED_FIELDS}
        for field in OPTIONAL_FIELDS:
            if field in df.columns:
                payload[field] = row[field]

        try:
            record, alert = monitor_transaction(payload)
            alerts_created += 1 if alert else 0
            records.append({
                "row":               int(index) + 2,
                "transaction_id":    record["id"],
                "customer_id":       record["metadata"]["customer_id"],
                "amount":            record["amount"],
                "label":             record["label"],
                "fraud_probability": record["fraud_probability"],
                "risk_score":        record["risk_score"],
                "risk_level":        record["risk_level"],
                "verification_status": record["verification_status"],
            })
        except Exception as exc:
            errors.append({"row": int(index) + 2, "error": str(exc)})

    return jsonify({
        "processed":      len(records),
        "failed":         len(errors),
        "alerts_created": alerts_created,
        "records":        records,
        "errors":         errors,
        "dashboard":      dashboard_payload(),
    })


@app.get("/api/report.csv")
@require_auth
def api_report_csv():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "id", "created_at", "customer_id", "merchant", "amount",
        "fraud_probability", "prediction", "risk_score", "risk_level",
        "verification_status", "behavior_signals",
    ])
    for record in TRANSACTIONS:
        writer.writerow([
            record["id"],
            record["created_at"],
            record["metadata"]["customer_id"],
            record["metadata"]["merchant"],
            record["amount"],
            record["fraud_probability"],
            record["prediction"],
            record["risk_score"],
            record["risk_level"],
            record["verification_status"],
            "; ".join(record["behavior"]["signals"]),
        ])
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=fraud_report.csv"},
    )


@app.post("/predict")
@require_auth
def predict():
    try:
        record, alert = monitor_transaction(request.get_json(silent=True))
        return jsonify({
            "fraud_probability":   record["fraud_probability"],
            "prediction":          record["prediction"],
            "label":               record["label"],
            "threshold":           record["threshold"],
            "risk_score":          record["risk_score"],
            "risk_level":          record["risk_level"],
            "behavior":            record["behavior"],
            "preprocessing":       record["preprocessing"],
            "verification_status": record["verification_status"],
            "transaction":         record,
            "alert":               alert,
            "dashboard":           dashboard_payload(),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="127.0.0.1", port=5000, debug=debug)