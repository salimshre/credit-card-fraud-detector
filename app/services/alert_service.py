import uuid

from app.config import MAX_RECORDS
from app.persistence.storage import ALERTS


def create_alert_if_needed(record: dict):
    if record["prediction"] == 0 and record["risk_score"] < 65:
        return None

    alert = {
        "id": str(uuid.uuid4())[:8],
        "transaction_id": record["id"],
        "created_at": record["created_at"],
        "severity": record["risk_level"],
        "message": (
            f"{record['risk_level']} risk transaction for "
            f"{record['metadata']['customer_id']} at {record['metadata']['merchant']}"
        ),
        "risk_score": record["risk_score"],
        "acknowledged": False,
    }
    ALERTS.insert(0, alert)
    del ALERTS[MAX_RECORDS:]
    return alert
