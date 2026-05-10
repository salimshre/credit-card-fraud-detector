# Fraud Shield – Credit Card Fraud Detection Dashboard

A runnable Flask application for credit‑card fraud monitoring. It combines a trained **Random Forest** classifier with operational layers such as authentication, transaction monitoring, risk scoring, alerts, verification, behaviour analysis, and downloadable reports.

The model uses **15 engineered features** (time, amount, category encodings, behavioural velocity, etc.) – not raw PCA components.

---

## Features

- **Transaction Monitoring** – every predicted transaction is saved with full metadata.
- **Machine Learning Prediction** – `fraud_model.pkl` returns a fraud probability and binary label.
- **Real‑Time Alerts** – high‑risk or model‑flagged transactions automatically create dashboard alerts.
- **User Authentication & Verification** – operators must sign in; they can approve, block, or mark transactions as false positives.
- **Risk Scoring System** – fuses model probability, model threshold, and behaviour signals into a **0‑100 risk score**.
- **Behavior Analysis** – customer‑level profiles track devices, countries, merchant categories, timing, and amount spikes.
- **Data Preprocessing** – the dashboard shows extracted features (hour, amount ratio, etc.) for each prediction.
- **Dashboard & Reports** – metrics, recent transactions, alert queue, behavioural profiles, and a downloadable CSV report.
- **CSV Batch Detection** – upload a CSV file to scan multiple transactions at once.

---

## Demo Login
Username: admin
Password: admin123


Change them with environment variables:

```powershell
# PowerShell
$env:APP_USERNAME = "analyst"
$env:APP_PASSWORD = "strong-password"
```

---

## Project Files

| File / Folder               | Purpose |
|-----------------------------|---------|
| `app.py`                    | Flask server, authentication, monitoring, scoring, alerts, verification, and reporting endpoints |
| `templates/index.html`      | Dashboard UI (login, prediction form, batch upload, verification queue) |
| `fraud_model.pkl`           | Trained Random Forest classifier (15 features) |
| `scaler.pkl`                | Fitted `StandardScaler` for those 15 features |
| `feature_engineering.py`    | Shared function `engineer_single_transaction()` and constant feature lists (used by both `app.py` and training) |
| `train_model.py`            | Standalone retraining script using `creditcard_raw.csv` |
| `self.ipynb`                | Jupyter notebook that generates synthetic training data, engineers features, trains the model, and evaluates it |
| `sample_transactions.json`  | Demo payloads for normal / fraud transactions |
| `sample_upload.csv`         | Ready‑to‑upload CSV batch example |
| `smoke_test.py`             | End‑to‑end API test suite |
| `data_store.json`           | Persistence file (transactions, alerts, behaviour profiles) – auto‑created/updated at runtime |
| `requirements.txt`          | Python dependencies |

> **Note:** The legacy files `real_life_transactions.csv`, `readme.txt`, `server.err`, `server.out` have been removed because they were either outdated or not part of the application.

---

## Setup

### 1. Create and activate a virtual environment

```powershell
# PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Running the Application

```powershell
python app.py
```

Open [http://127.0.0.1:5000/](http://127.0.0.1:5000/).

---

## Running the Smoke Tests

```powershell
python smoke_test.py
```

The test suite verifies:
- `/health` returns `200`
- login succeeds / fails correctly
- normal sample predicts `Normal`
- fraud sample predicts `Fraud`
- CSV batch upload processes the sample file
- dashboard and report endpoints respond
- authenticated endpoints reject unauthenticated requests

---

## API Endpoints

| Method | Endpoint                                     | Description |
|--------|----------------------------------------------|-------------|
| `GET`  | `/health`                                    | App status and feature list |
| `POST` | `/login`                                     | Authenticate (JSON body: `username`, `password`) |
| `POST` | `/logout`                                    | Clear session |
| `POST` | `/predict`                                   | Submit a single transaction for scoring. **Required JSON fields:** `transaction_date` (YYYY‑MM‑DD), `transaction_time` (HH:MM:SS), `amount` (non‑negative float), `merchant_category`, `country`, `channel`. **Optional:** `customer_id`, `card_last4`, `merchant`, `device_id`. |
| `GET`  | `/api/dashboard`                             | Returns metrics, alerts, recent transactions, and behaviour profiles |
| `GET`  | `/api/transactions`                          | All monitored transactions |
| `GET`  | `/api/alerts`                                | All active alerts |
| `POST` | `/api/alerts/<alert_id>/acknowledge`        | Acknowledge an alert |
| `POST` | `/api/transactions/<transaction_id>/verify`  | Approve, Block, or mark `False Positive` (JSON body: `status`, optional `note`) |
| `GET`  | `/api/sample-csv`                            | Download a sample CSV for batch upload |
| `POST` | `/api/upload-csv`                            | Upload a CSV file; processes each row and returns results, errors, and updated dashboard data |
| `GET`  | `/api/report.csv`                            | Download a CSV report of all monitored transactions |

---

## CSV Batch Upload

In the dashboard, use the **Manual CSV Fraud Detection** section. The CSV must contain these columns:

| Required columns |
|------------------|
| `transaction_date` |
| `transaction_time` |
| `amount`           |
| `merchant_category`|
| `country`          |
| `channel`          |

Optional metadata columns:  
`customer_id`, `card_last4`, `merchant`, `device_id`

The included `sample_upload.csv` shows the exact format. Each row goes through the full pipeline: preprocessing → ML prediction → risk scoring → behaviour update → alert generation → verification queue → reports.

---

## Model Performance (from `self.ipynb`)

The Random Forest was trained on **60,994 synthetic transactions** (1.46% fraud) with 15 engineered features.

**Test set (20% stratified split):**

```
Confusion Matrix:
[[12005    16]
 [   12   166]]

Classification Report:
              precision    recall  f1-score   support
           0      0.999     0.999     0.999     12021
           1      0.912     0.933     0.922       178

    accuracy                          0.998     12199
   macro avg      0.956     0.966     0.961     12199
weighted avg      0.998     0.998     0.998     12199
```

**5‑fold Cross‑Validation Recall (on training set):**  
`0.9242 ± 0.0271`

The model catches most frauds while keeping false positives manageable – exactly why the verification queue exists.

---

## Retraining the Model

1. Generate or obtain a CSV named **`creditcard_raw.csv`** (must contain `transaction_date`, `transaction_time`, `card_last4`, `merchant_category`, `amount`, `country`, `channel`, `is_fraud`).  
   The notebook `self.ipynb` can create a synthetic dataset for you.

2. Run the training script:

```powershell
python train_model.py
```

3. The script will:
   - Engineer the 15 features (same code as `feature_engineering.py` and `app.py`)
   - Scale the features
   - Train a `RandomForestClassifier` (200 trees, class‑weight balanced, max depth 12)
   - Print evaluation metrics and 5‑fold CV recall
   - Overwrite `fraud_model.pkl` and `scaler.pkl`

---

## Security & Production Notes

This is a **demo / educational project** and not a production fraud system.  
To harden it for real deployments, you would add:

- A proper database (PostgreSQL, MySQL) instead of in‑memory lists and `data_store.json`
- Hashed passwords and token‑based authentication (Flask‑JWT, OAuth)
- HTTPS, secure session configuration, and environment‑only secrets
- Model versioning, drift monitoring, and a CI/CD pipeline
- Audit logs, role‑based access control, and rate limiting
```

**What changed?**

- The model feature description now reflects the 15 engineered features.
- The “Project Files” table lists actual current files and notes that legacy files have been removed.
- API `/predict` documentation lists the required fields correctly.
- CSV batch upload required columns are updated.
- Model performance is taken from `self.ipynb` (the Random Forest that matches `fraud_model.pkl`).
- Retraining instructions point to `creditcard_raw.csv` and `train_model.py`.
- All references to the old PCA dataset (`V1`–`V28`, `creditcard.csv`) have been removed.

Replace your existing `README.md` with the content above, and the project will be fully in sync.