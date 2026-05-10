"""
train_model.py
--------------
Retrains fraud_model.pkl and scaler.pkl from creditcard_raw.csv.

The feature set and model type exactly match what app.py + feature_engineering.py
expect at inference time:
  - 15 engineered features defined in feature_engineering.MODEL_FEATURES
  - RandomForestClassifier  (same as self.ipynb)
  - StandardScaler fit on those 15 features

Run:
    python train_model.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# Import the shared feature list and encoders so train and serve are identical.
from feature_engineering import CHANNELS, COUNTRIES, MERCHANT_CATEGORIES, MODEL_FEATURES

BASE_DIR   = Path(__file__).resolve().parent
DATA_PATH  = BASE_DIR / "creditcard_raw.csv"
MODEL_PATH = BASE_DIR / "fraud_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"


# ── Feature engineering (mirrors self.ipynb, kept local to avoid circular deps) ──

def _encode(value, category_list):
    v = str(value).lower().strip()
    return category_list.index(v) if v in category_list else len(category_list) - 1


def compute_per_card_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 15 MODEL_FEATURES columns to *df* in-place."""
    df = df.copy()
    df["datetime"] = pd.to_datetime(
        df["transaction_date"].astype(str) + " " + df["transaction_time"].astype(str)
    )
    df = df.sort_values("datetime").reset_index(drop=True)

    # Basic time / categorical features
    df["hour"]        = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["is_night"]    = ((df["hour"] >= 0) & (df["hour"] <= 5)).astype(int)
    df["amount"]      = df["amount"].astype(float)
    df["amount_log"]  = np.log1p(df["amount"])

    df["merchant_category_enc"] = df["merchant_category"].apply(
        lambda x: _encode(x, MERCHANT_CATEGORIES)
    )
    df["channel_enc"]  = df["channel"].apply(lambda x: _encode(x, CHANNELS))
    df["country_enc"]  = df["country"].apply(lambda x: _encode(x, COUNTRIES))

    # Per-card behavioural features (initialise with safe defaults)
    df["txn_count_last_1h"]  = 0
    df["txn_count_last_24h"] = 0
    df["avg_amount_prev"]    = df["amount"]   # first-txn default = own amount → ratio 1.0
    df["amount_ratio"]       = 1.0
    df["is_new_country"]     = 0
    df["is_new_category"]    = 0

    ONE_HOUR = np.timedelta64(1, "h")
    ONE_DAY  = np.timedelta64(24, "h")

    for _card, group in df.groupby("card_last4"):
        idxs       = group.index.tolist()
        datetimes  = group["datetime"].values
        amounts    = group["amount"].values
        countries  = group["country"].values
        categories = group["merchant_category"].values

        for i, row_idx in enumerate(idxs):
            if i == 0:
                continue                            # defaults already set above

            cur_dt   = datetimes[i]
            prev_dts = datetimes[:i]

            df.at[row_idx, "txn_count_last_1h"]  = int((prev_dts >= cur_dt - ONE_HOUR).sum())
            df.at[row_idx, "txn_count_last_24h"] = int((prev_dts >= cur_dt - ONE_DAY).sum())

            avg_prev = float(amounts[:i].mean())
            df.at[row_idx, "avg_amount_prev"] = avg_prev
            df.at[row_idx, "amount_ratio"]    = amounts[i] / max(avg_prev, 1.0)
            df.at[row_idx, "is_new_country"]  = int(countries[i]  not in set(countries[:i]))
            df.at[row_idx, "is_new_category"] = int(categories[i] not in set(categories[:i]))

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find '{DATA_PATH.name}'. "
            "Either run self.ipynb (which generates creditcard_raw.csv) "
            "or download it from Kaggle and rename to creditcard_raw.csv."
        )

    df_raw = pd.read_csv(DATA_PATH)

    required_cols = [
        "transaction_date", "transaction_time", "card_last4",
        "merchant_category", "amount", "country", "channel", "is_fraud",
    ]
    missing = [c for c in required_cols if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Dataset is missing column(s): {', '.join(missing)}")

    n_fraud = int(df_raw["is_fraud"].sum())
    print(
        f"Loaded {len(df_raw):,} transactions | "
        f"Fraud: {n_fraud:,} ({n_fraud / len(df_raw):.2%})"
    )

    print("Engineering features (this may take a minute for large datasets)…")
    df = compute_per_card_features(df_raw)

    X = df[MODEL_FEATURES].copy()
    y = df["is_fraud"].astype(int)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape[0]:,} samples  |  Test: {X_test.shape[0]:,} samples")

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("RandomForest trained.")

    y_pred = model.predict(X_test)
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    recall_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="recall")
    print(f"\n5-fold CV recall: {recall_scores.mean():.4f} ± {recall_scores.std():.4f}")

    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\nSaved {MODEL_PATH.name} and {SCALER_PATH.name}.")


if __name__ == "__main__":
    main()