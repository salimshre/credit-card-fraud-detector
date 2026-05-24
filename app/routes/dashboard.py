from flask import Blueprint, jsonify

from app.persistence.storage import ALERTS, BEHAVIOR_PROFILES, TRANSACTIONS
from app.routes import require_auth

dashboard_bp = Blueprint("dashboard", __name__)


def dashboard_payload() -> dict:
    total = len(TRANSACTIONS)
    high_risk = sum(1 for item in TRANSACTIONS if item["risk_score"] >= 65)
    fraud_preds = sum(1 for item in TRANSACTIONS if item["prediction"] == 1)
    verified = sum(
        1
        for item in TRANSACTIONS
        if item["verification_status"] != "Pending Review"
    )
    average_risk = (
        round(sum(item["risk_score"] for item in TRANSACTIONS) / total, 1)
        if total
        else 0
    )
    open_alerts = sum(1 for alert in ALERTS if not alert["acknowledged"])

    return {
        "stats": {
            "total_transactions": total,
            "fraud_predictions": fraud_preds,
            "high_risk_transactions": high_risk,
            "open_alerts": open_alerts,
            "verified_transactions": verified,
            "average_risk": average_risk,
        },
        "transactions": TRANSACTIONS[:25],
        "alerts": ALERTS[:25],
        "profiles": {
            customer_id: {
                "transaction_count": profile["transaction_count"],
                "amount_average": round(profile.get("amount_avg", 0), 2),
                "amount_max": round(profile["amount_max"], 2),
                "devices": sorted(profile["devices"]),
                "countries": sorted(profile["countries"]),
            }
            for customer_id, profile in BEHAVIOR_PROFILES.items()
        },
    }


@dashboard_bp.get("/api/dashboard")
@require_auth
def api_dashboard():
    return jsonify(dashboard_payload())


@dashboard_bp.get("/api/transactions")
@require_auth
def api_transactions():
    return jsonify({"transactions": TRANSACTIONS})
