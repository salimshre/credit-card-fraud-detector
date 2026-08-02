"""
Synthetic credit-card transaction generator for model development.

The generator intentionally keeps fraud patterns probabilistic instead of
making a single field such as country, channel, or category determine the
label. This makes notebook metrics less inflated and closer to the behavior
expected from a real fraud model.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import CHANNELS, COUNTRIES, MERCHANT_CATEGORIES

ACTIVE_MERCHANT_CATEGORIES = [c for c in MERCHANT_CATEGORIES if c != "other"]
ACTIVE_CHANNELS = [c for c in CHANNELS if c != "other"]
ACTIVE_COUNTRIES = [c for c in COUNTRIES if c != "other"]

COMMON_HOME_COUNTRIES = ["US", "CA", "GB", "FR", "DE", "AU"]
HIGHER_RISK_COUNTRIES = ["RU", "CN", "BR", "MX"]


def _hour_weights() -> list[float]:
    weights = [1, 1, 1, 1, 2, 3, 4, 6, 8, 8, 8, 7, 7, 7, 6, 6, 5, 5, 5, 4, 3, 3, 2, 1]
    total = sum(weights)
    return [weight / total for weight in weights]


HOUR_WEIGHTS = _hour_weights()
FRAUD_HOUR_WEIGHTS = [5, 5, 5, 5, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 4, 5]
FRAUD_HOUR_PROBS = [weight / sum(FRAUD_HOUR_WEIGHTS) for weight in FRAUD_HOUR_WEIGHTS]


def _other_country(home_country: str) -> str:
    choices = [country for country in ACTIVE_COUNTRIES if country != home_country]
    return random.choice(choices)


def _card_profile(card_last4: int) -> dict:
    # Generate average amount in NPR (e.g., between 500 and 50,000 NPR)
    avg_amount = float(np.random.lognormal(8, 1.2))
    return {
        "customer_id": f"CUST-{card_last4}",
        "card_last4": card_last4,
        "home_country": random.choice(COMMON_HOME_COUNTRIES),
        "avg_amount": avg_amount,
        "usual_categories": random.sample(
            ACTIVE_MERCHANT_CATEGORIES,
            k=random.randint(3, 6),
        ),
        "usual_channel": random.choices(
            ACTIVE_CHANNELS,
            weights=[0.28, 0.45, 0.08, 0.19],
        )[0],
    }


def _transaction_datetime(base_date: datetime, day_offset: int, fraud: bool) -> datetime:
    hour_probs = FRAUD_HOUR_PROBS if fraud else HOUR_WEIGHTS
    hour = np.random.choice(range(24), p=hour_probs)
    return base_date + timedelta(
        days=day_offset,
        hours=int(hour),
        minutes=np.random.randint(0, 60),
        seconds=np.random.randint(0, 60),
    )


def _normal_country(profile: dict) -> str:
    bucket = random.choices(
        ["home", "other_known", "higher_risk"],
        weights=[0.88, 0.10, 0.02],
    )[0]
    if bucket == "home":
        return profile["home_country"]
    if bucket == "higher_risk":
        return random.choice(HIGHER_RISK_COUNTRIES)
    return _other_country(profile["home_country"])


def _fraud_country(profile: dict) -> str:
    bucket = random.choices(
        ["home", "other_known", "higher_risk"],
        weights=[0.45, 0.30, 0.25],
    )[0]
    if bucket == "home":
        return profile["home_country"]
    if bucket == "higher_risk":
        return random.choice(HIGHER_RISK_COUNTRIES)
    return _other_country(profile["home_country"])


def _normal_category(profile: dict) -> str:
    if random.random() < 0.82:
        return random.choice(profile["usual_categories"])
    return random.choice(ACTIVE_MERCHANT_CATEGORIES)


def _fraud_category(profile: dict) -> str:
    suspicious_categories = ["cash", "online", "travel"]
    if random.random() < 0.65:
        return random.choices(suspicious_categories, weights=[0.38, 0.40, 0.22])[0]
    return random.choice(ACTIVE_MERCHANT_CATEGORIES)


def _normal_amount(profile: dict) -> float:
    # Normal spending: stays close to the average (low sigma)
    amount = np.random.lognormal(
        mean=np.log(max(profile["avg_amount"], 100.0)),
        sigma=0.5, 
    )
    return round(max(1.0, float(amount)), 2)


def _fraud_amount(profile: dict) -> float:
    mode = random.choices(["high", "medium", "low"], weights=[0.55, 0.30, 0.15])[0]
    avg_amount = max(profile["avg_amount"], 100.0)

    if mode == "high":
        # High fraud amounts: 5x to 10x average
        amount = np.random.lognormal(mean=np.log(avg_amount * 6.0), sigma=0.6)
    elif mode == "medium":
        # Medium fraud amounts: 2x to 3x average
        amount = np.random.lognormal(mean=np.log(avg_amount * 2.5), sigma=0.4)
    else:
        # Low fraud amounts: just a bit above average
        amount = np.random.lognormal(mean=np.log(avg_amount * 1.3), sigma=0.4)

    return round(max(5.0, float(amount)), 2)


def _base_record(profile: dict, dt: datetime, fraud: bool) -> dict:
    return {
        "transaction_date": dt.strftime("%Y-%m-%d"),
        "transaction_time": dt.strftime("%H:%M:%S"),
        "customer_id": profile["customer_id"],
        "card_last4": profile["card_last4"],
        "merchant": f"Merchant_{random.randint(1, 400)}",
        "merchant_category": _fraud_category(profile) if fraud else _normal_category(profile),
        "amount": _fraud_amount(profile) if fraud else _normal_amount(profile),
        "country": _fraud_country(profile) if fraud else _normal_country(profile),
        "channel": random.choices(
            ACTIVE_CHANNELS,
            weights=[0.38, 0.20, 0.28, 0.14] if fraud else [0.28, 0.45, 0.08, 0.19],
        )[0],
        "is_fraud": int(fraud),
    }


def generate_dataset(
    n_cards: int = 600,
    avg_txn_per_card: int = 100,
    fraud_rate: float = 0.015,
    output_file: str | Path = "creditcard_raw.csv",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic transactions and write them to ``output_file``."""
    np.random.seed(seed)
    random.seed(seed)

    base_date = datetime(2023, 1, 1)
    card_numbers = random.sample(range(1000, 10000), n_cards)
    records: list[dict] = []

    for card_last4 in card_numbers:
        profile = _card_profile(card_last4)
        n_txn = max(10, int(np.random.normal(avg_txn_per_card, 20)))

        for _ in range(n_txn):
            fraud = random.random() < fraud_rate
            day_offset = random.randint(0, 364)
            dt = _transaction_datetime(base_date, day_offset, fraud)
            records.append(_base_record(profile, dt, fraud))

    df = pd.DataFrame(records).sample(frac=1, random_state=seed).reset_index(drop=True)
    output_path = Path(output_file)
    df.to_csv(output_path, index=False)

    fraud_count = int(df["is_fraud"].sum())
    print(
        f"Generated {len(df):,} transactions | "
        f"Fraud: {fraud_count:,} ({fraud_count / len(df):.2%})"
    )
    print(f"Saved {output_path}")
    return df


def summarize_label_leakage(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return per-column fraud-rate summaries for quick notebook inspection."""
    summaries = {}
    for column in ["country", "channel", "merchant_category"]:
        summaries[column] = (
            df.groupby(column)["is_fraud"]
            .agg(["count", "sum", "mean"])
            .sort_values("mean", ascending=False)
        )
    return summaries
    