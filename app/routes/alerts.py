from flask import Blueprint, jsonify, request, session

from app.persistence.storage import ALERTS, TRANSACTIONS, save_state
from app.routes import require_auth
from app.services.scoring_service import now_iso, sanitize_text

alerts_bp = Blueprint("alerts", __name__)


@alerts_bp.get("/api/alerts")
@require_auth
def api_alerts():
    return jsonify({"alerts": ALERTS})


@alerts_bp.post("/api/alerts/<alert_id>/acknowledge")
@require_auth
def api_acknowledge_alert(alert_id: str):
    for alert in ALERTS:
        if alert["id"] == alert_id:
            alert["acknowledged"] = True
            save_state()
            return jsonify(alert)
    return jsonify({"error": "Alert not found."}), 404


@alerts_bp.post("/api/transactions/<transaction_id>/verify")
@require_auth
def api_verify_transaction(transaction_id: str):
    payload = request.get_json(silent=True) or {}
    status = sanitize_text(payload.get("status"), "Reviewed")
    note = sanitize_text(payload.get("note"), "")
    allowed = {"Approved", "Blocked", "Reviewed", "False Positive"}
    if status not in allowed:
        return jsonify({
            "error": f"Status must be one of: {', '.join(sorted(allowed))}."
        }), 400

    for record in TRANSACTIONS:
        if record["id"] == transaction_id:
            record["verification_status"] = status
            
            # ---- NEW: Update the static label to match the user's verdict ----
            if status == "Approved":
                record["label"] = "Approved"
            elif status == "Blocked":
                record["label"] = "Blocked"
            elif status == "False Positive":
                record["label"] = "False Positive"
            elif status == "Reviewed":
                record["label"] = "Reviewed"
            # ------------------------------------------------------------------

            record["verification_note"] = note
            record["verified_by"] = session.get("username", "")
            record["verified_at"] = now_iso()
            for alert in ALERTS:
                if alert["transaction_id"] == transaction_id:
                    alert["acknowledged"] = True
            save_state()
            return jsonify(record)

    return jsonify({"error": "Transaction not found."}), 404
    