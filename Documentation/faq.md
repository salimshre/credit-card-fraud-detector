# FAQ

## Why does the first entered transaction show Fraud, but the same data later shows Normal?

This can happen because the app is stateful. The same visible form data is not always the same full model input.

Fraud Shield stores customer behavior history after each prediction. When the same `customer_id` is used again, the app compares the new transaction against the customer's updated history.

### Simple Explanation

After the app first loads, the customer may have no behavior history yet.

First submit:

```text
customer_id = CUST-1
country = US
device_id = web-demo
merchant_category = online

History: none
Signals: new customer, new country, new device, new category
Possible result: Fraud
```

Second submit with the same visible data:

```text
customer_id = CUST-1
country = US
device_id = web-demo
merchant_category = online

History: already seen from the first submit
Signals: behavior matches known customer pattern
Possible result: Normal
```

The visible form data is the same, but the hidden behavior history is different.

### What Happens

1. The first transaction for a customer is treated as a new behavior profile.
2. The app may add behavior signals such as:

```text
New customer profile
New device for this customer
New country for this customer
New merchant category for this customer
High transaction amount
Night-time transaction
```

3. These behavior signals affect model features and risk score.
4. After prediction, the app saves the transaction into the customer's behavior profile.
5. When the same transaction is submitted again, the customer is no longer new.
6. The country, merchant category, and device may now be known.
7. Because the behavior features changed, the model input changes.
8. The result can change from `Fraud` to `Normal`.

### Features That Can Change Between Submissions

The model uses behavior-related features such as:

```text
txn_count_last_1h
txn_count_last_24h
avg_amount_prev
amount_ratio
is_new_country
is_new_category
```

Even if the form values are identical, these derived features can be different on the second submission.

### Relevant Code

Behavior profile is loaded before prediction:

```text
app/services/scoring_service.py
```

```python
profile = get_behavior_profile(metadata["customer_id"])
behavior = analyze_behavior(txn, metadata, profile)
```

The transaction is scored using behavior-aware features:

```python
prepared_df, preprocessing = preprocess_transaction(txn, behavior)
probability, prediction = score_ml(prepared_df)
```

After scoring, the behavior profile is updated:

```python
update_behavior_profile(txn, metadata)
```

The behavior features are generated in:

```text
feature_engineering.py
```

Important derived fields include:

```python
amount_ratio
is_new_country
is_new_category
txn_count_last_1h
txn_count_last_24h
```

### Is This a Bug?

Not exactly. It is expected behavior with the current design because the app learns customer behavior as transactions are submitted.

This is useful for fraud monitoring because a transaction can be suspicious the first time but less suspicious after it matches a known customer pattern.

### When Is This Confusing?

It can be confusing during demos or testing because the user may expect identical form data to always return the same prediction.

In this app, identical form data can produce different results if the customer behavior profile has changed.

### How To Get Consistent Results While Testing

Use a new `customer_id` for each independent test.

Example:

```text
CUST-TEST-001
CUST-TEST-002
CUST-TEST-003
```

Or clear/reset the runtime data store before testing.

Local runtime data is stored in:

```text
instance/data_store.json
```

The Render deployment uses:

```text
/tmp/fraud-shield-data-store.json
```

### Possible Future Fix

If deterministic predictions are required, add a stateless prediction mode.

In stateless mode:

1. Prediction would not use saved customer behavior history.
2. Prediction would not update the behavior profile.
3. The same input would always produce the same model result.

This could be implemented as a separate endpoint or toggle, for example:

```text
/predict?mode=stateless
```

or a dashboard checkbox:

```text
Do not update customer behavior profile
```
