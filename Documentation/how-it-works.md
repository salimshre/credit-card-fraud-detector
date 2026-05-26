# How Fraud Shield Works

Fraud Shield is a Flask web dashboard for credit-card fraud detection. It lets an analyst enter transaction data, score fraud risk with a trained machine-learning model, review alerts, verify decisions, upload CSV batches, and download reports.

Live app:

```text
https://credit-card-fraud-detector-j4ms.onrender.com
```

## Login

Default demo credentials:

```text
Username: admin
Password: admin123
```

For a public deployment, change these values in the hosting environment variables:

```text
APP_USERNAME
APP_PASSWORD
SECRET_KEY
```

## Basic Workflow

1. Open the web app.
2. Log in as an analyst.
3. Enter a transaction manually or upload a CSV file.
4. The app validates the transaction fields.
5. The app converts the transaction into model-ready features.
6. The trained model predicts fraud probability.
7. The dashboard displays:
   - label: `Normal` or `Fraud`
   - fraud probability
   - risk score
   - risk level
   - behavior signals
8. Risky transactions create alerts.
9. The analyst reviews alerts and verifies transactions.
10. The analyst can download a CSV report.

## Manual Transaction Input

The single-transaction form accepts credit-card transaction details.

Required fields:

```text
transaction_date
transaction_time
amount
merchant_category
country
channel
```

Optional fields:

```text
customer_id
card_last4
merchant
device_id
```

### Example Input

```json
{
  "transaction_date": "2026-05-26",
  "transaction_time": "14:30",
  "amount": 249.99,
  "merchant_category": "electronics",
  "country": "US",
  "channel": "online",
  "customer_id": "CUST-1001",
  "card_last4": "4242",
  "merchant": "Demo Store",
  "device_id": "device-abc"
}
```

## What Happens After Data Entry

When a transaction is submitted, the app performs these steps:

1. Validates required fields.
2. Normalizes values such as country and channel.
3. Builds engineered model features.
4. Scales numeric values with the saved scaler.
5. Sends the feature vector to the trained Random Forest model.
6. Reads the fraud probability.
7. Compares the probability with the configured fraud threshold.
8. Creates a dashboard transaction record.
9. Creates an alert if the transaction is high-risk.

## Machine Learning Files

The prediction system depends on these saved files:

```text
fraud_model.pkl
scaler.pkl
model_metadata.json
```

Purpose:

| File | Purpose |
| --- | --- |
| `fraud_model.pkl` | Trained Random Forest classifier |
| `scaler.pkl` | Saved numeric feature scaler |
| `model_metadata.json` | Training metrics and recommended fraud threshold |

## Risk Output

After prediction, the app returns:

| Output | Meaning |
| --- | --- |
| `label` | Final decision: `Normal` or `Fraud` |
| `fraud_probability` | Model probability that the transaction is fraud |
| `risk_score` | Human-friendly score from `0` to `100` |
| `risk_level` | Risk category used by the dashboard |
| `behavior_signals` | Extra signals from customer behavior analysis |

## Behavior Signals

The app also tracks customer behavior patterns. These signals help explain why a transaction may look risky.

Examples:

```text
new device
new country
new merchant category
unusual amount
high transaction velocity
```

These behavior checks are combined with model output to support dashboard alerts and analyst review.

## Alerts

Alerts are created when a transaction is model-flagged or has high risk.

An analyst can:

```text
acknowledge an alert
review a transaction
approve a transaction
block a transaction
mark a transaction as false positive
```

This turns the app into a simple fraud-monitoring workflow instead of only a prediction form.

## CSV Batch Upload

The app can scan multiple transactions from a CSV file.

Use the sample file:

```text
sample_upload.csv
```

The CSV should include the same required fields used by the manual form:

```text
transaction_date,transaction_time,amount,merchant_category,country,channel
```

Optional columns can also be included:

```text
customer_id,card_last4,merchant,device_id
```

Each uploaded row is processed and returned with:

```text
row number
transaction ID
label
fraud probability
risk score
risk level
```

## Dashboard

The dashboard shows:

```text
total transactions
fraud count
normal count
risk metrics
recent transactions
alerts
customer behavior profiles
```

This gives the analyst a quick overview of monitored transaction activity.

## Reports

The app can export monitored transaction data as CSV from:

```text
/api/report.csv
```

Use this for offline review, sharing, or record keeping.

## API Endpoints

Main endpoints:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Check app status |
| `POST` | `/login` | Log in |
| `POST` | `/logout` | Log out |
| `POST` | `/predict` | Score one transaction |
| `GET` | `/api/dashboard` | Dashboard metrics and activity |
| `GET` | `/api/transactions` | List transactions |
| `GET` | `/api/alerts` | List alerts |
| `POST` | `/api/upload-csv` | Upload and scan CSV |
| `GET` | `/api/report.csv` | Download report |

## Data Storage

Local development stores runtime data in:

```text
instance/data_store.json
```

The Render deployment uses:

```text
/tmp/fraud-shield-data-store.json
```

On Render's free plan, this storage is temporary. Runtime data can disappear after restarts or redeploys.

For production, replace JSON storage with:

```text
PostgreSQL
SQLite
another persistent database
```

## Important Limitations

This is an educational/demo project, not a production fraud platform.

Before using it seriously:

1. Replace demo credentials.
2. Hash passwords instead of storing plain values.
3. Add role-based access control.
4. Use a persistent database.
5. Add audit logs.
6. Monitor model drift.
7. Retrain and validate the model on real, approved data.
8. Add stronger security around uploads and reports.

## Short Summary

Fraud Shield takes transaction data, engineers features, applies a trained fraud model, calculates risk, creates alerts, and gives an analyst tools to review and verify suspicious transactions.
