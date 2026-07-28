import math
import uuid
from datetime import datetime, timezone

import joblib
import pandas as pd
from flask import session

from app.config import MAX_RECORDS, MODEL_PATH, NPR_BLOCK_AMOUNT, NPR_REVIEW_AMOUNT, REQUIRED_FIELDS, SCALER_PATH, THRESHOLD
from app.persistence.storage import TRANSACTIONS, save_state
from app.services.alert_service import create_alert_if_needed
from app.services.behavior_service import (
    analyze_behavior,
    get_behavior_profile,
    update_behavior_profile,
)
from feature_engineering import MODEL_FEATURES, engineer_single_transaction


def load_artifacts():
    missing = [p.name for p in (MODEL_PATH, SCALER_PATH) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required file(s): {', '.join(missing)}")
    loaded_model = joblib.load(MODEL_PATH)
    loaded_scaler = joblib.load(SCALER_PATH)
    if hasattr(loaded_model, "set_params"):
        try:
            loaded_model.set_params(n_jobs=1)
        except ValueError:
            pass
    return loaded_model, loaded_scaler


model, scaler = load_artifacts()


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


def require_text(value: object, field: str) -> str:
    cleaned = sanitize_text(value, "")
    if not cleaned:
        raise ValueError(f"{field} must not be empty.")
    return cleaned


def normalize_transaction_date(value: object) -> str:
    cleaned = require_text(value, "transaction_date")
    try:
        datetime.strptime(cleaned, "%Y-%m-%d")
    except ValueError:
        raise ValueError("transaction_date must use YYYY-MM-DD format.")
    return cleaned


def normalize_transaction_time(value: object) -> str:
    cleaned = require_text(value, "transaction_time")
    parts = cleaned.split(":")
    if len(parts) == 2:
        cleaned = f"{parts[0].zfill(2)}:{parts[1].zfill(2)}:00"
    elif len(parts) == 3:
        cleaned = ":".join(part.zfill(2) for part in parts)
    else:
        raise ValueError("transaction_time must use HH:MM or HH:MM:SS format.")

    try:
        datetime.strptime(cleaned, "%H:%M:%S")
    except ValueError:
        raise ValueError("transaction_time must use HH:MM or HH:MM:SS format.")
    return cleaned


def parse_transaction(payload: object) -> tuple[dict, dict]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    missing = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")

    try:
        amount = float(payload["amount"])
    except (TypeError, ValueError):
        raise ValueError("amount must be a numeric value.")
    if not math.isfinite(amount) or amount < 0:
        raise ValueError("amount must be a non-negative finite number.")

    txn = {
        "transaction_date": normalize_transaction_date(payload.get("transaction_date")),
        "transaction_time": normalize_transaction_time(payload.get("transaction_time")),
        "amount": amount,
        "merchant_category": require_text(payload.get("merchant_category"), "merchant_category"),
        "country": require_text(payload.get("country"), "country"),
        "channel": require_text(payload.get("channel"), "channel"),
    }
    metadata = {
        "customer_id": sanitize_text(payload.get("customer_id"), "CUST-DEMO"),
        "card_last4": sanitize_text(payload.get("card_last4"), "0000"),
        "merchant": sanitize_text(payload.get("merchant"), "Unknown Merchant"),
        "merchant_category": txn["merchant_category"],
        "country": txn["country"],
        "channel": txn["channel"],
        "device_id": sanitize_text(payload.get("device_id"), "web-demo"),
    }
    return txn, metadata


def preprocess_transaction(txn: dict, behavior: dict):
    raw_features = engineer_single_transaction(txn, behavior)
    input_df = pd.DataFrame([raw_features], columns=MODEL_FEATURES)
    scaled = scaler.transform(input_df)
    prepared_df = pd.DataFrame(scaled, columns=MODEL_FEATURES)

    return prepared_df, {
        "feature_count": len(MODEL_FEATURES),
        "hour": raw_features["hour"],
        "is_night": raw_features["is_night"],
        "is_weekend": raw_features["is_weekend"],
        "amount_ratio": round(raw_features["amount_ratio"], 4),
        "txn_count_last_24h": raw_features["txn_count_last_24h"],
    }


def score_ml(prepared_df) -> tuple[float, int]:
    model_input = prepared_df.to_numpy() if hasattr(prepared_df, "to_numpy") else prepared_df
    probability = float(model.predict_proba(model_input)[0, 1])
    prediction = int(probability >= THRESHOLD)
    return probability, prediction


def calculate_risk(probability: float, prediction: int, behavior: dict) -> tuple[int, str]:
    ml_component = probability * 72
    model_flag = 10 if prediction else 0
    behavior_component = min(behavior["behavior_points"], 28)
    risk_score = min(100, round(ml_component + model_flag + behavior_component))

    if risk_score >= 85:
        risk_level = "Critical"
    elif risk_score >= 65:
        risk_level = "High"
    elif risk_score >= 35:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return risk_score, risk_level


def monitor_transaction(payload: object) -> tuple[dict, object]:
    txn, metadata = parse_transaction(payload)
    profile = get_behavior_profile(metadata["customer_id"])

    # 1. Analyze behavior and ML
    behavior = analyze_behavior(txn, metadata, profile)  # Includes 3x average rule
    prepared_df, preprocessing = preprocess_transaction(txn, behavior)
    probability, prediction = score_ml(prepared_df)
    base_risk_score, base_risk_level = calculate_risk(probability, prediction, behavior)

    amount = txn["amount"]

    # 2. Apply NPR Business Rules to determine outcome (Normal / Review Required / Blocked / Fraud)
    final_label = "Normal"
    final_risk_level = base_risk_level
    final_risk_score = base_risk_score
    verification_status = "Auto Cleared"

    if amount >= NPR_BLOCK_AMOUNT:
        final_label = "Blocked"
        final_risk_level = "Critical"
        final_risk_score = 100
        verification_status = "Blocked – Limit Exceeded"
    
    elif amount >= NPR_REVIEW_AMOUNT:
        if prediction == 1:
            final_label = "Fraud"
        else:
            final_label = "Review Required"
        verification_status = "Pending Review"
        final_risk_level = "High"
        final_risk_score = max(base_risk_score, 65)

    else:
        # Amount < 100,000 NPR
        if prediction == 1:
            final_label = "Fraud"
            verification_status = "Pending Review"
            final_risk_level = "Critical"
            final_risk_score = max(base_risk_score, 65)
        else:
            # Check for strong behavior signals (like relative 3x rule or multiple anomalies)
            if behavior["behavior_points"] >= 20:
                final_label = "Review Required"
                verification_status = "Pending Review"
                final_risk_level = "High"
                final_risk_score = max(base_risk_score, 65)
            else:
                final_label = "Normal"
                verification_status = "Auto Cleared"

    # 3. Construct the record
    record = {
        "id": str(uuid.uuid4())[:8],
        "created_at": now_iso(),
        "operator": session.get("username", "api"),
        "metadata": metadata,
        "amount": txn["amount"],
        "transaction_date": txn["transaction_date"],
        "transaction_time": txn["transaction_time"],
        "fraud_probability": round(probability, 6),
        "prediction": prediction,
        "label": final_label,          # Normal, Review Required, Blocked, Fraud
        "threshold": THRESHOLD,
        "risk_score": final_risk_score,
        "risk_level": final_risk_level,
        "behavior": behavior,
        "preprocessing": preprocessing,
        "verification_status": verification_status,
        "verification_note": "",
        "verified_by": "",
        "verified_at": "",
    }

    alert = create_alert_if_needed(record)

    # 4. Update Profile ONLY if transaction is NOT Blocked
    if final_label != "Blocked":
        update_behavior_profile(txn, metadata)

    # 5. Persist to memory and DB
    TRANSACTIONS.insert(0, record)
    del TRANSACTIONS[MAX_RECORDS:]
    save_state()

    return record, alert