from datetime import datetime, timezone

from app.persistence.storage import BEHAVIOR_PROFILES


def count_recent_both(profile: dict) -> tuple[int, int]:
    timestamps = profile.get("recent_timestamps", [])
    if not timestamps:
        return 0, 0

    now = datetime.now(timezone.utc).timestamp()
    cutoff_1h = now - 3600
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
        "transaction_count": 0,
        "amount_total": 0.0,
        "amount_avg": 0.0,
        "amount_max": 0.0,
        "recent_timestamps": [],
        "devices": set(),
        "countries": set(),
        "merchant_categories": set(),
    })


def analyze_behavior(txn: dict, metadata: dict, profile: dict) -> dict:
    amount = txn["amount"]
    signals = []
    behavior_points = 0

    txn_1h, txn_24h = count_recent_both(profile)

    if profile["transaction_count"] == 0:
        signals.append("New customer profile")
        behavior_points += 4
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

    # ---- NEW RELATIVE SPENDING RULE (3× Average) ----
    if profile["transaction_count"] > 0 and profile["amount_avg"] > 0:
        if amount > 3 * profile["amount_avg"]:
            signals.append(f"Amount is more than 3x customer's average ({profile['amount_avg']:.2f} NPR)")
            behavior_points += 12

    if amount >= 1000:
        signals.append("High transaction amount")
        behavior_points += 10

    try:
        hour = int(str(txn.get("transaction_time", "12:00:00")).split(":")[0])
        if 0 <= hour <= 5:
            signals.append("Night-time transaction (00:00-05:59)")
            behavior_points += 6
    except (ValueError, AttributeError, IndexError):
        pass

    if not signals:
        signals.append("Behavior matches known customer pattern")

    return {
        "customer_id": metadata["customer_id"],
        "signals": signals,
        "behavior_points": behavior_points,
        "profile_transaction_count": profile["transaction_count"],
        "txn_count_last_1h": txn_1h,
        "txn_count_last_24h": txn_24h,
        "amount_avg": effective_avg,
        "is_new_country": int(txn["country"] not in profile["countries"]),
        "is_new_category": int(txn["merchant_category"] not in profile["merchant_categories"]),
    }


def update_behavior_profile(txn: dict, metadata: dict) -> None:
    profile = get_behavior_profile(metadata["customer_id"])
    amount = txn["amount"]
    count = profile["transaction_count"] + 1

    profile["transaction_count"] = count
    profile["amount_total"] += amount
    profile["amount_avg"] = profile["amount_total"] / count
    profile["amount_max"] = max(profile["amount_max"], amount)

    profile["recent_timestamps"].append(datetime.now(timezone.utc).timestamp())
    if len(profile["recent_timestamps"]) > 200:
        profile["recent_timestamps"] = profile["recent_timestamps"][-200:]

    profile["devices"].add(metadata.get("device_id", "unknown"))
    profile["countries"].add(txn["country"])
    profile["merchant_categories"].add(txn["merchant_category"])
    