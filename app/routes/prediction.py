import csv
import io

import pandas as pd
from flask import Blueprint, Response, jsonify, request

from app.config import BASE_DIR, MAX_CSV_ROWS, OPTIONAL_FIELDS, REQUIRED_FIELDS
from app.extensions import limiter
from app.persistence.storage import TRANSACTIONS
from app.routes import require_auth
from app.routes.dashboard import dashboard_payload
from app.services.scoring_service import monitor_transaction

prediction_bp = Blueprint("prediction", __name__)


@prediction_bp.get("/api/sample-csv")
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


@prediction_bp.post("/api/upload-csv")
@require_auth
@limiter.limit("10 per hour")
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

    missing = [field for field in REQUIRED_FIELDS if field not in df.columns]
    if missing:
        return jsonify({
            "error": f"CSV is missing required column(s): {', '.join(missing)}.",
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
                "row": int(index) + 2,
                "transaction_id": record["id"],
                "customer_id": record["metadata"]["customer_id"],
                "amount": record["amount"],
                "label": record["label"],
                "fraud_probability": record["fraud_probability"],
                "risk_score": record["risk_score"],
                "risk_level": record["risk_level"],
                "verification_status": record["verification_status"],
            })
        except Exception as exc:
            errors.append({"row": int(index) + 2, "error": str(exc)})

    return jsonify({
        "processed": len(records),
        "failed": len(errors),
        "alerts_created": alerts_created,
        "records": records,
        "errors": errors,
        "dashboard": dashboard_payload(),
    })


@prediction_bp.get("/api/report.csv")
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


@prediction_bp.post("/predict")
@require_auth
@limiter.limit("60 per minute")
def predict():
    try:
        record, alert = monitor_transaction(request.get_json(silent=True))
        return jsonify({
            "fraud_probability": record["fraud_probability"],
            "prediction": record["prediction"],
            "label": record["label"],
            "threshold": record["threshold"],
            "risk_score": record["risk_score"],
            "risk_level": record["risk_level"],
            "behavior": record["behavior"],
            "preprocessing": record["preprocessing"],
            "verification_status": record["verification_status"],
            "transaction": record,
            "alert": alert,
            "dashboard": dashboard_payload(),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
        