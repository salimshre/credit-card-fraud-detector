"""
feature_engineering.py
-----------------------
Shared feature constants and single-transaction engineering used by both
app.py (real-time inference) and train_model.py (batch training).

Behavior dict contract
~~~~~~~~~~~~~~~~~~~~~~
``engineer_single_transaction`` accepts an optional *behavior* mapping.
When called from app.py it is the dict returned by ``analyze_behavior``.
The keys it reads are:

    amount_avg         float  – customer's historical average spend
                                (0.0 ⟹ treated as "no history")
    is_new_country     int    – 1 if this country was never seen before
    is_new_category    int    – 1 if this merchant category is new
    txn_count_last_1h  int    – transactions in the past hour
    txn_count_last_24h int    – transactions in the past 24 hours

All keys default to safe values (0 / current amount) when absent.
"""

import logging
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)

# ── Shared category lists ────────────────────────────────────────────────────
# Order matters: the index becomes the encoded integer fed to the model.
# New values must be appended, never inserted, to keep existing model weights valid.

MERCHANT_CATEGORIES = [
    "retail", "grocery", "restaurant", "gas_station",
    "travel", "entertainment", "healthcare", "online",
    "cash", "utilities", "other",          # "other" must stay last (catch-all)
]

CHANNELS = [
    "web", "pos", "atm", "mobile", "other",   # "other" must stay last
]

COUNTRIES = [
    "US", "CA", "GB", "FR", "DE", "AU", "JP",
    "CN", "RU", "BR", "MX", "IN", "other",    # "other" must stay last
]

# ── Feature names (must match train_model.py and app.py) ────────────────────
MODEL_FEATURES = [
    "hour", "day_of_week", "is_weekend", "is_night",
    "amount", "amount_log",
    "merchant_category_enc", "channel_enc", "country_enc",
    "txn_count_last_1h", "txn_count_last_24h",
    "avg_amount_prev", "amount_ratio",
    "is_new_country", "is_new_category",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def encode_category(value: object, category_list: list) -> int:
    """Return the index of *value* in *category_list*, or the last index."""
    v = str(value).lower().strip()
    try:
        return category_list.index(v)
    except ValueError:
        return len(category_list) - 1          # catch-all "other" slot


def _parse_datetime(txn: dict) -> datetime:
    """
    Parse transaction_date + transaction_time into a datetime.

    Accepts HH:MM:SS, H:MM:SS, HH:MM, and H:MM formats.
    Falls back to datetime.now() with a logged warning rather than crashing.
    """
    raw_date = str(txn.get("transaction_date", "")).strip()
    raw_time = str(txn.get("transaction_time", "")).strip()

    # Normalise time to HH:MM:SS so strptime is unambiguous
    parts = raw_time.split(":")
    if len(parts) == 2:                        # HH:MM – append seconds
        raw_time = raw_time + ":00"
    elif len(parts) == 3:
        # Zero-pad hour so '9:05:00' → '09:05:00'
        raw_time = ":".join(p.zfill(2) for p in parts)

    try:
        return datetime.strptime(f"{raw_date} {raw_time}", "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError) as exc:
        logger.warning(
            "Could not parse datetime from date=%r time=%r (%s); "
            "using current time instead.",
            raw_date, raw_time, exc,
        )
        return datetime.now()


# ── Main entry point ─────────────────────────────────────────────────────────

def engineer_single_transaction(txn: dict, behavior: dict | None = None) -> dict:
    """
    Convert a raw transaction dict and an optional behavior profile into the
    15-feature dict expected by the model.

    Parameters
    ----------
    txn:
        Must contain at minimum: transaction_date, transaction_time, amount,
        merchant_category, channel, country.
    behavior:
        Dict with keys described in the module docstring.  When omitted or
        empty all behavioural features default to neutral values.

    Returns
    -------
    dict with exactly the keys in MODEL_FEATURES.
    """
    if behavior is None:
        behavior = {}

    dt          = _parse_datetime(txn)
    hour        = dt.hour
    day_of_week = dt.weekday()
    is_weekend  = int(day_of_week >= 5)
    is_night    = int(0 <= hour <= 5)

    amount     = float(txn.get("amount", 0))
    amount_log = np.log1p(amount)

    merc_enc    = encode_category(txn.get("merchant_category", "other"), MERCHANT_CATEGORIES)
    chan_enc    = encode_category(txn.get("channel",            "other"), CHANNELS)
    country_enc = encode_category(txn.get("country",           "US"),    COUNTRIES)

    # ── Behavioural features ─────────────────────────────────────────────────
    # amount_avg is 0.0 for brand-new customers (no history).
    # Using 0.0 as avg_prev would make amount_ratio = amount (a huge number),
    # so we fall back to the current amount to produce a neutral ratio of 1.0.
    raw_avg_prev = behavior.get("amount_avg", 0.0)
    avg_prev     = raw_avg_prev if raw_avg_prev > 0 else amount

    amount_ratio    = amount / max(avg_prev, 1.0)
    is_new_country  = int(behavior.get("is_new_country",  0))
    is_new_category = int(behavior.get("is_new_category", 0))
    txn_1h          = int(behavior.get("txn_count_last_1h",  0))
    txn_24h         = int(behavior.get("txn_count_last_24h", 0))

    return {
        "hour":                  hour,
        "day_of_week":           day_of_week,
        "is_weekend":            is_weekend,
        "is_night":              is_night,
        "amount":                amount,
        "amount_log":            amount_log,
        "merchant_category_enc": merc_enc,
        "channel_enc":           chan_enc,
        "country_enc":           country_enc,
        "txn_count_last_1h":     txn_1h,
        "txn_count_last_24h":    txn_24h,
        "avg_amount_prev":       avg_prev,
        "amount_ratio":          round(amount_ratio, 6),
        "is_new_country":        is_new_country,
        "is_new_category":       is_new_category,
    }