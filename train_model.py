"""
Retrain fraud_model.pkl and scaler.pkl from creditcard_raw.csv.

The feature set and model type match what app.py expects at inference time:
  - 15 engineered features defined in feature_engineering.MODEL_FEATURES
  - RandomForestClassifier
  - StandardScaler fit only on the training split

Run:
    python train_model.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from feature_engineering import (
    CHANNELS,
    COUNTRIES,
    MERCHANT_CATEGORIES,
    MODEL_FEATURES,
    encode_category,
)
from synthetic_data import generate_dataset

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "creditcard_raw.csv"
MODEL_PATH = BASE_DIR / "fraud_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.json"


def compute_per_card_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 15 MODEL_FEATURES columns using only previous card history."""
    df = df.copy()
    df["datetime"] = pd.to_datetime(
        df["transaction_date"].astype(str) + " " + df["transaction_time"].astype(str)
    )
    df = df.sort_values("datetime").reset_index(drop=True)

    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"] = ((df["hour"] >= 0) & (df["hour"] <= 5)).astype(int)
    df["amount"] = df["amount"].astype(float)
    df["amount_log"] = np.log1p(df["amount"])

    df["merchant_category_enc"] = df["merchant_category"].apply(
        lambda x: encode_category(x, MERCHANT_CATEGORIES)
    )
    df["channel_enc"] = df["channel"].apply(lambda x: encode_category(x, CHANNELS))
    df["country_enc"] = df["country"].apply(lambda x: encode_category(x, COUNTRIES))

    df["txn_count_last_1h"] = 0
    df["txn_count_last_24h"] = 0
    df["avg_amount_prev"] = df["amount"]
    df["amount_ratio"] = 1.0
    df["is_new_country"] = 0
    df["is_new_category"] = 0

    one_hour = np.timedelta64(1, "h")
    one_day = np.timedelta64(24, "h")

    for _card, group in df.groupby("card_last4", sort=False):
        idxs = group.index.tolist()
        datetimes = group["datetime"].values
        amounts = group["amount"].values
        countries = group["country"].values
        categories = group["merchant_category"].values

        for i, row_idx in enumerate(idxs):
            if i == 0:
                continue

            cur_dt = datetimes[i]
            prev_dts = datetimes[:i]

            df.at[row_idx, "txn_count_last_1h"] = int((prev_dts >= cur_dt - one_hour).sum())
            df.at[row_idx, "txn_count_last_24h"] = int((prev_dts >= cur_dt - one_day).sum())

            avg_prev = float(amounts[:i].mean())
            df.at[row_idx, "avg_amount_prev"] = avg_prev
            df.at[row_idx, "amount_ratio"] = amounts[i] / max(avg_prev, 1.0)
            df.at[row_idx, "is_new_country"] = int(countries[i] not in set(countries[:i]))
            df.at[row_idx, "is_new_category"] = int(categories[i] not in set(categories[:i]))

    return df


def chronological_train_validation_test_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Use time-based train/validation/test splits, with stratified fallback."""
    df = df.sort_values("datetime").reset_index(drop=True)
    train_end = int(len(df) * 0.7)
    validation_end = int(len(df) * 0.85)
    train_df = df.iloc[:train_end].copy()
    validation_df = df.iloc[train_end:validation_end].copy()
    test_df = df.iloc[validation_end:].copy()

    splits_have_both_classes = all(
        split["is_fraud"].nunique() >= 2
        for split in (train_df, validation_df, test_df)
    )
    if splits_have_both_classes:
        return train_df, validation_df, test_df, "chronological_70_15_15"

    train_validation_df, test_df = train_test_split(
        df,
        test_size=0.15,
        random_state=42,
        stratify=df["is_fraud"].astype(int),
    )
    train_df, validation_df = train_test_split(
        train_validation_df,
        test_size=0.15 / 0.85,
        random_state=42,
        stratify=train_validation_df["is_fraud"].astype(int),
    )
    return (
        train_df.copy(),
        validation_df.copy(),
        test_df.copy(),
        "stratified_70_15_15",
    )


def choose_threshold(
    y_true: pd.Series,
    y_score: np.ndarray,
    *,
    beta: float = 2.0,
    min_precision: float = 0.55,
) -> tuple[float, dict]:
    """
    Select a probability threshold from validation predictions.

    F-beta with beta=2 favors fraud recall while the minimum precision guard
    prevents the threshold from creating an unusable alert flood.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5, {
            "selection_rule": "fallback_no_thresholds",
            "beta": beta,
            "min_precision": min_precision,
        }

    precision = precision[:-1]
    recall = recall[:-1]
    beta_sq = beta**2
    fbeta = (1 + beta_sq) * precision * recall / np.maximum(
        (beta_sq * precision) + recall,
        1e-12,
    )

    candidates = np.where(precision >= min_precision)[0]
    if len(candidates) == 0:
        candidates = np.arange(len(thresholds))
        selection_rule = "max_fbeta_no_min_precision_candidate"
    else:
        selection_rule = "max_fbeta_with_min_precision"

    best_idx = int(candidates[np.nanargmax(fbeta[candidates])])
    threshold = float(thresholds[best_idx])
    return threshold, {
        "selection_rule": selection_rule,
        "beta": beta,
        "min_precision": min_precision,
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        "fbeta": float(fbeta[best_idx]),
    }


def evaluate_scores(y_true: pd.Series, y_score: np.ndarray, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }


def chronological_cv_metrics(
    X: pd.DataFrame,
    y: pd.Series,
    model_params: dict,
    threshold: float,
    *,
    n_splits: int = 5,
) -> dict:
    """Run time-ordered CV on training data without shuffled future leakage."""
    splitter = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    for fold, (train_idx, validation_idx) in enumerate(splitter.split(X), start=1):
        y_train_fold = y.iloc[train_idx]
        y_validation_fold = y.iloc[validation_idx]
        if y_train_fold.nunique() < 2 or y_validation_fold.nunique() < 2:
            continue

        cv_pipeline = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(**model_params),
        )
        cv_pipeline.fit(X.iloc[train_idx], y_train_fold)
        validation_score = cv_pipeline.predict_proba(X.iloc[validation_idx])[:, 1]
        metrics = evaluate_scores(y_validation_fold, validation_score, threshold)
        metrics["fold"] = fold
        metrics["validation_rows"] = int(len(validation_idx))
        metrics["validation_fraud_rows"] = int(y_validation_fold.sum())
        fold_metrics.append(metrics)

    if not fold_metrics:
        return {"folds": [], "recall_mean": None, "recall_std": None}

    recalls = np.array([fold["recall"] for fold in fold_metrics])
    pr_aucs = np.array([fold["pr_auc"] for fold in fold_metrics])
    return {
        "folds": fold_metrics,
        "recall_mean": float(recalls.mean()),
        "recall_std": float(recalls.std()),
        "pr_auc_mean": float(pr_aucs.mean()),
        "pr_auc_std": float(pr_aucs.std()),
    }


def main() -> None:
    if not DATA_PATH.exists():
        print(f"Cannot find {DATA_PATH.name}; generating synthetic training data.")
        generate_dataset(output_file=DATA_PATH)

    df_raw = pd.read_csv(DATA_PATH)

    required_cols = [
        "transaction_date",
        "transaction_time",
        "card_last4",
        "merchant_category",
        "amount",
        "country",
        "channel",
        "is_fraud",
    ]
    missing = [c for c in required_cols if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Dataset is missing column(s): {', '.join(missing)}")

    n_fraud = int(df_raw["is_fraud"].sum())
    print(
        f"Loaded {len(df_raw):,} transactions | "
        f"Fraud: {n_fraud:,} ({n_fraud / len(df_raw):.2%})"
    )

    print("Engineering features (this may take a minute for large datasets)...")
    df = compute_per_card_features(df_raw)
    train_df, validation_df, test_df, split_name = chronological_train_validation_test_split(df)
    print(f"Using {split_name} split.")

    X_train = train_df[MODEL_FEATURES].copy()
    y_train = train_df["is_fraud"].astype(int)
    X_validation = validation_df[MODEL_FEATURES].copy()
    y_validation = validation_df["is_fraud"].astype(int)
    X_test = test_df[MODEL_FEATURES].copy()
    y_test = test_df["is_fraud"].astype(int)

    print(
        f"Train: {X_train.shape[0]:,} samples | "
        f"Validation: {X_validation.shape[0]:,} samples | "
        f"Test: {X_test.shape[0]:,} samples"
    )

    model_params = {
        "n_estimators": 200,
        "class_weight": "balanced",
        "max_depth": 12,
        "random_state": 42,
        "n_jobs": -1,
    }

    selection_scaler = StandardScaler()
    X_train_scaled = selection_scaler.fit_transform(X_train)
    X_validation_scaled = selection_scaler.transform(X_validation)

    model = RandomForestClassifier(**model_params)
    model.fit(X_train_scaled, y_train)
    validation_score = model.predict_proba(X_validation_scaled)[:, 1]
    recommended_threshold, threshold_selection = choose_threshold(
        y_validation,
        validation_score,
    )
    validation_metrics = evaluate_scores(
        y_validation,
        validation_score,
        recommended_threshold,
    )
    print(
        "Selected threshold "
        f"{recommended_threshold:.4f} from validation "
        f"(precision={validation_metrics['precision']:.3f}, "
        f"recall={validation_metrics['recall']:.3f}, "
        f"PR-AUC={validation_metrics['pr_auc']:.3f})."
    )

    X_train_validation = pd.concat([X_train, X_validation], axis=0)
    y_train_validation = pd.concat([y_train, y_validation], axis=0)

    scaler = StandardScaler()
    X_train_validation_scaled = scaler.fit_transform(X_train_validation)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(**model_params)
    model.fit(X_train_validation_scaled, y_train_validation)
    print("RandomForest trained on train + validation data.")

    y_score = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_score >= recommended_threshold).astype(int)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report_text = classification_report(y_test, y_pred, digits=3, zero_division=0)
    report_dict = classification_report(
        y_test,
        y_pred,
        digits=3,
        output_dict=True,
        zero_division=0,
    )

    print("\nConfusion matrix:")
    print(conf_matrix)
    print("\nClassification report:")
    print(report_text)

    test_metrics = evaluate_scores(y_test, y_score, recommended_threshold)
    print(
        f"Test PR-AUC: {test_metrics['pr_auc']:.4f} | "
        f"ROC-AUC: {test_metrics['roc_auc']:.4f} | "
        f"F2: {test_metrics['f2']:.4f}"
    )

    cv_metrics = chronological_cv_metrics(
        X_train_validation,
        y_train_validation,
        model_params,
        recommended_threshold,
    )
    if cv_metrics["recall_mean"] is not None:
        print(
            "\n5-fold chronological CV recall: "
            f"{cv_metrics['recall_mean']:.4f} +/- {cv_metrics['recall_std']:.4f}"
        )

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": DATA_PATH.name,
        "rows": int(len(df)),
        "fraud_rows": int(df["is_fraud"].sum()),
        "split": split_name,
        "features": MODEL_FEATURES,
        "model": "RandomForestClassifier",
        "model_params": model_params,
        "recommended_threshold": recommended_threshold,
        "threshold_selection": threshold_selection,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "confusion_matrix": conf_matrix.tolist(),
        "classification_report": report_dict,
        "chronological_cv": cv_metrics,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nSaved {MODEL_PATH.name}, {SCALER_PATH.name}, and {METADATA_PATH.name}.")


if __name__ == "__main__":
    main()
