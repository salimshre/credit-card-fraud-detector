# Fraud Shield - Credit Card Fraud Detection Dashboard

Fraud Shield is a runnable Flask dashboard for credit-card fraud monitoring. It combines a trained Random Forest classifier with transaction monitoring, behavior signals, risk scoring, alerts, analyst verification, CSV batch scanning, and downloadable reports.

The model uses 15 engineered features from transaction time, amount, merchant category, channel, country, and customer behavior history.

## Features

- Single transaction fraud prediction with fraud probability and risk score.
- Customer behavior profiling for devices, countries, categories, amount spikes, and velocity.
- Real-time alert creation for model-flagged or high-risk transactions.
- Analyst verification workflow: approve, block, review, or mark false positive.
- CSV batch upload for manual fraud detection.
- Dashboard metrics, recent transaction history, and CSV report export.
- Retraining script that saves model artifacts and model metadata.

## Live Demo

The app is deployed on Render:

```text
https://credit-card-fraud-detector-j4ms.onrender.com
```

Deployment notes are documented in:

```text
Documentation/render-deployment-guide.md
```

## Demo Login

Default local credentials:

```text
Username: admin
Password: admin123
```

Override them before running the app:

```powershell
$env:APP_USERNAME = "analyst"
$env:APP_PASSWORD = "strong-password"
$env:SECRET_KEY = "replace-with-a-random-secret"
```

## Project Files

| File / Folder | Purpose |
| --- | --- |
| `app.py` | Thin compatibility launcher for `python app.py` |
| `app/app.py` | Flask application factory and blueprint registration |
| `app/routes/` | Auth, dashboard, alert, and prediction route modules |
| `app/services/` | Scoring, behavior analysis, and alert creation services |
| `app/persistence/storage.py` | JSON-backed runtime state loading and saving |
| `app/templates/index.html` | Dashboard UI |
| `feature_engineering.py` | Shared feature constants and single-transaction feature engineering |
| `synthetic_data.py` | Synthetic transaction generator used for model development |
| `train_model.py` | Retrains the model from `creditcard_raw.csv` |
| `fraud_model.pkl` | Saved Random Forest model |
| `scaler.pkl` | Saved `StandardScaler` |
| `model_metadata.json` | Generated training metrics and model metadata |
| `creditcard_raw.csv` | Synthetic training dataset used by the current model |
| `sample_transactions.json` | Normal and fraud demo payloads |
| `sample_upload.csv` | Example CSV batch upload |
| `smoke_test.py` | End-to-end and regression checks |
| `Documentation/CreditCard/` | LaTeX report source, bibliography, and generated PDF |

Runtime state is saved to `instance/data_store.json` by default. A legacy root-level `data_store.json` is read once if present, but new writes go to `instance/`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```powershell
python app.py
```

Open `http://127.0.0.1:5000/`.

## Test

```powershell
python smoke_test.py
```

The smoke test checks health, login, country encoding, sample predictions, validation errors, CSV upload, dashboard data, report download, logout, and authenticated route protection.

## Retrain

```powershell
python train_model.py
```

The training script:

- Engineers the same 15 features used by the app.
- Generates `creditcard_raw.csv` if it is missing.
- Uses chronological train/validation/test splits when each split has both classes.
- Fits preprocessing only on training data during threshold selection.
- Tunes a fraud probability threshold on validation data.
- Trains a balanced Random Forest.
- Writes `fraud_model.pkl`, `scaler.pkl`, and `model_metadata.json`.

At runtime the app uses `FRAUD_THRESHOLD` when it is set. Otherwise it falls back to the `recommended_threshold` saved in `model_metadata.json`.

## API Summary

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | App status and feature list |
| `POST` | `/login` | Authenticate |
| `POST` | `/logout` | Clear session |
| `POST` | `/predict` | Score one transaction |
| `GET` | `/api/dashboard` | Metrics, alerts, transactions, and profiles |
| `GET` | `/api/transactions` | All monitored transactions |
| `GET` | `/api/alerts` | Alert list |
| `POST` | `/api/alerts/<alert_id>/acknowledge` | Acknowledge an alert |
| `POST` | `/api/transactions/<transaction_id>/verify` | Update verification status |
| `GET` | `/api/sample-csv` | Download sample upload CSV |
| `POST` | `/api/upload-csv` | Scan a CSV batch |
| `GET` | `/api/report.csv` | Download monitored transaction report |

## Production Gaps

This is an educational/demo project. The next serious upgrades are:

- Replace JSON persistence with SQLite or PostgreSQL.
- Use hashed passwords, role-based access control, and environment-only secrets.
- Add audit logs for verification and alert actions.
- Add model threshold tuning, drift monitoring, and model version comparisons.
- Move runtime state from JSON files to SQLite or PostgreSQL before real multi-user use.
