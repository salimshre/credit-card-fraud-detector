"""
End-to-end regression checks for the Fraud Shield Flask app.

Run:
    python smoke_test.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SAMPLES_PATH = BASE_DIR / "sample_transactions.json"
SAMPLE_CSV_PATH = BASE_DIR / "sample_upload.csv"
TEST_STORE_PATH = Path(tempfile.gettempdir()) / "fraud_shield_smoke_test_state.json"

if TEST_STORE_PATH.exists():
    TEST_STORE_PATH.unlink()
os.environ["FRAUD_STORE_PATH"] = str(TEST_STORE_PATH)

import app as fraud_app
from app.persistence import storage
from feature_engineering import COUNTRIES, encode_category

# Use the same password we set in the environment
ADMIN_PASSWORD = os.getenv("DEFAULT_ADMIN_PASSWORD", "9705550012")

app = fraud_app.app

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {name}")
        return
    msg = f"{name}" + (f" - {detail}" if detail else "")
    print(f"  FAIL  {msg}")
    failures.append(msg)


def reset_test_state() -> None:
    storage.STORE_PATH = TEST_STORE_PATH
    if TEST_STORE_PATH.exists():
        TEST_STORE_PATH.unlink()
    storage.TRANSACTIONS.clear()
    storage.ALERTS.clear()
    storage.BEHAVIOR_PROFILES.clear()


def main() -> None:
    samples = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))
    reset_test_state()

    print("\n[0] Feature encoding")
    check("US encodes to US index", encode_category("US", COUNTRIES) == COUNTRIES.index("US"))
    check("lowercase country encodes correctly", encode_category("ru", COUNTRIES) == COUNTRIES.index("RU"))

    with app.test_client() as client:
        print("\n[1] Health check")
        resp = client.get("/health")
        data = resp.get_json()
        check("status 200", resp.status_code == 200)
        check("status field is ok", data.get("status") == "ok", f"got {data.get('status')!r}")
        check("feature_count > 0", data.get("feature_count", 0) > 0)

        print("\n[2] Login")
        # Use the admin credentials (password from environment)
        resp = client.post(
            "/login",
            json={"username": fraud_app.APP_USERNAME, "password": ADMIN_PASSWORD},
        )
        data = resp.get_json()
        check("status 200", resp.status_code == 200, f"got {resp.status_code}")
        check("status field is ok", data.get("status") == "ok", f"got {data.get('status')!r}")

        resp_bad = client.post(
            "/login",
            json={"username": fraud_app.APP_USERNAME, "password": "wrong"},
        )
        check("bad credentials -> 401", resp_bad.status_code == 401, f"got {resp_bad.status_code}")

        client.post(
            "/login",
            json={"username": fraud_app.APP_USERNAME, "password": ADMIN_PASSWORD},
        )

        print("\n[3] Predictions")
        expected_labels = {"normal": "Normal", "fraud": "Fraud"}
        for name, payload in samples.items():
            resp = client.post("/predict", json=payload)
            data = resp.get_json()
            check(f"{name}: status 200", resp.status_code == 200, f"got {resp.status_code}")

            label = data.get("label")
            expected = expected_labels[name]
            check(f"{name}: label is {expected}", label == expected, f"got {label!r}")

            prob = data.get("fraud_probability")
            check(f"{name}: fraud_probability in [0,1]", prob is not None and 0.0 <= prob <= 1.0, f"got {prob}")

            risk = data.get("risk_score")
            check(f"{name}: risk_score in [0,100]", risk is not None and 0 <= risk <= 100, f"got {risk}")
            check(f"{name}: behavior signals present", bool(data.get("behavior", {}).get("signals")))
            check(f"{name}: preprocessing keys present", "feature_count" in data.get("preprocessing", {}))

        bad_payload = dict(samples["normal"])
        bad_payload["transaction_date"] = "2024/01/15"
        resp_bad_date = client.post("/predict", json=bad_payload)
        check("invalid transaction_date -> 400", resp_bad_date.status_code == 400, f"got {resp_bad_date.status_code}")

        print("\n[4] Dashboard")
        resp = client.get("/api/dashboard")
        data = resp.get_json()
        check("status 200", resp.status_code == 200)
        stats = data.get("stats", {})
        check("total_transactions >= 2", stats.get("total_transactions", 0) >= 2, f"got {stats.get('total_transactions')}")
        check("transactions list present", isinstance(data.get("transactions"), list))
        check("alerts list present", isinstance(data.get("alerts"), list))

        print("\n[5] Sample CSV download")
        resp = client.get("/api/sample-csv")
        check("status 200", resp.status_code == 200, f"got {resp.status_code}")
        check("mimetype is text/csv", resp.mimetype == "text/csv", f"got {resp.mimetype!r}")

        print("\n[6] CSV batch upload")
        with SAMPLE_CSV_PATH.open("rb") as csv_file:
            resp = client.post(
                "/api/upload-csv",
                data={"file": (csv_file, "sample_upload.csv")},
                content_type="multipart/form-data",
            )
        data = resp.get_json()
        check("status 200", resp.status_code == 200, f"got {resp.status_code}")
        check("processed > 0", data.get("processed", 0) > 0, f"got {data.get('processed')}")
        check("failed == 0", data.get("failed", -1) == 0, f"got {data.get('failed')}")
        check("records list present", isinstance(data.get("records"), list))

        if data.get("records"):
            rec = data["records"][0]
            for key in ("row", "transaction_id", "label", "risk_score", "risk_level"):
                check(f"CSV record has {key}", key in rec)

        print("\n[7] Report CSV download")
        resp = client.get("/api/report.csv")
        check("status 200", resp.status_code == 200, f"got {resp.status_code}")
        check("mimetype is text/csv", resp.mimetype == "text/csv", f"got {resp.mimetype!r}")
        check("non-empty body", len(resp.data) > 0)

        print("\n[8] Logout")
        resp = client.post("/logout", headers={"Accept": "application/json"})
        check("status 200", resp.status_code == 200)

        resp_unauth = client.get("/api/dashboard")
        check("dashboard requires auth after logout", resp_unauth.status_code == 401, f"got {resp_unauth.status_code}")

    print()
    if failures:
        print(f"  {len(failures)} check(s) FAILED:")
        for failure in failures:
            print(f"    - {failure}")
        sys.exit(1)

    print("  All checks passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
    