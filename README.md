<div align="center">

# 🛡️ Fraud Shield

**Credit Card Fraud Detection & Monitoring Dashboard**

A Flask dashboard that combines a trained Random Forest classifier with real-time transaction monitoring, customer behavior profiling, risk scoring, alerts, analyst verification, CSV batch scanning, and downloadable reports.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/flask-web%20app-black)](https://flask.palletsprojects.com/)
[![Model](https://img.shields.io/badge/model-RandomForest-green)](#model)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](#license)

[Live Demo](#live-demo) · [Setup](#setup) · [API](#api-summary) · [FAQ](#faq) · [Contributors](#fraud-shield-contributors)

</div>

---

> [!TIP]
> **New here?** Jump straight to [Setup](#setup) to run it locally, or check the [Live Demo](#live-demo) to try it without installing anything.

## Overview

Fraud Shield turns raw transaction data into a fraud decision in a few steps: validate the transaction, analyze the customer's behavior history, engineer 15 model features, scale and score with a Random Forest classifier, combine the model's probability with behavior-based risk points, and surface the result through a dashboard with alerts and an analyst verification workflow.

The model uses 15 engineered features drawn from transaction time, amount, merchant category, channel, country, and customer behavior history.

## Features

- 🔍 **Single transaction scoring** — fraud probability, binary label, and a 0–100 risk score.
- 🧠 **Behavior profiling** — tracks devices, countries, categories, amount spikes, and transaction velocity per customer.
- 🚨 **Real-time alerts** — created automatically for model-flagged or high-risk transactions.
- ✅ **Analyst verification** — approve, block, review, or mark a transaction as a false positive.
- 📄 **CSV batch upload** — scan many transactions at once.
- 📊 **Dashboard metrics** — totals, fraud counts, open alerts, average risk, and recent transaction history.
- 📥 **CSV report export** — download monitored transactions for offline analysis.
- 🔁 **Retraining script** — regenerates model artifacts and metadata from scratch.

## Live Demo

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
$env:DEFAULT_ADMIN_PASSWORD = "strong-password"
$env:SECRET_KEY = "replace-with-a-random-secret"
```

> [!NOTE]
> The seeded admin password is controlled by `DEFAULT_ADMIN_PASSWORD`, not `APP_PASSWORD`. Passwords are stored bcrypt-hashed in the `users` table — never in plain text.

## Project Files

| File / Folder | Purpose |
| --- | --- |
| `app.py` | Thin compatibility launcher for `python app.py` |
| `app/app.py` | Flask application factory and blueprint registration |
| `app/routes/` | Auth, dashboard, alert, and prediction route modules |
| `app/services/` | Scoring, behavior analysis, and alert creation services |
| `app/persistence/storage.py` | SQLite-backed runtime state (transactions, alerts, behavior profiles, users), with one-time legacy JSON import |
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

Runtime state is persisted to an SQLite database at `instance/fraud_shield.db`. A legacy root-level `data_store.json`, if present, is imported once on first startup when the database is empty; all subsequent reads and writes use SQLite.

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
| `GET` | `/health` | App status, threshold, model files, feature count, and capability list |
| `POST` | `/login` | Authenticate (rate-limited, bcrypt-verified) |
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

## FAQ

<details>
<summary><strong>Is this trained on real bank data?</strong></summary>
<br>
No. The model is trained on <code>creditcard_raw.csv</code>, a synthetic dataset (59,978 transactions, ~1.54% fraud) generated by <code>synthetic_data.py</code>. Results reflect performance on this synthetic benchmark and should not be treated as representative of real banking fraud.
</details>

<details>
<summary><strong>Why does a "3x average" transaction sometimes pass and sometimes get flagged?</strong></summary>
<br>
The Random Forest model looks at all 15 features together, not <code>amount_ratio</code> in isolation — the same ratio can be legit in one context (daytime, familiar merchant, home country) and fraud in another (night, cash/ATM, unfamiliar country). See <code>feature_engineering.py</code> for the full feature list.
</details>

<details>
<summary><strong>What happens if the model and the behavior rules disagree?</strong></summary>
<br>
Both run independently and are combined in <code>calculate_risk()</code>: <code>risk_score = probability × 0.72 + flag(+10 if fraud) + min(behavior_points, 28)</code>. The rule engine can escalate a transaction the model considers low-risk (e.g. a brand-new device), but it cannot downgrade a transaction the model flags as fraud.
</details>

<details>
<summary><strong>Where is data actually stored?</strong></summary>
<br>
SQLite, at <code>instance/fraud_shield.db</code>. A legacy <code>data_store.json</code> is imported once on first run if the database is empty; it is not the ongoing storage mechanism.
</details>

<details>
<summary><strong>Can I use my own transaction data?</strong></summary>
<br>
Yes — run <code>train_model.py</code> against your own CSV in the same schema as <code>creditcard_raw.csv</code> (see <code>feature_engineering.py</code> for required fields), or submit transactions to <code>/predict</code> / <code>/api/upload-csv</code> against the existing model.
</details>

<details>
<summary><strong>Is this production-ready?</strong></summary>
<br>
No — see <a href="#production-gaps">Production Gaps</a> below.
</details>

## Production Gaps

This is an educational/demo project. The next serious upgrades are:

- Add a persistent audit-log table of operator actions (verification status is currently overwritten, not history-tracked).
- Add role-based access control — all authenticated operators currently share equal access.
- Add TLS/HTTPS termination for encrypted transport.
- Add model drift monitoring and automated retraining triggers.
- Migrate from SQLite to a networked database (e.g. PostgreSQL) for real multi-user, multi-instance deployment.

## Fraud Shield Contributors

Fraud Shield is made possible by these contributors.

<a href="https://github.com/YOUR-USERNAME/YOUR-REPO/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=YOUR-USERNAME/YOUR-REPO"/>
</a>

## License

This project is provided for academic and educational purposes as part of a Bachelor in Computer Engineering coursework submission.