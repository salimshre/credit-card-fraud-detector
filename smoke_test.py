"""
smoke_test.py
-------------
End-to-end test suite for the Fraud Shield Flask app.

Run:
    python smoke_test.py

Every check now uses explicit assertions so a failure causes a non-zero exit
and prints a clear message – previously all checks only printed results without
verifying them.
"""

import json
import sys
from pathlib import Path

from app import app

BASE_DIR         = Path(__file__).resolve().parent
SAMPLES_PATH     = BASE_DIR / "sample_transactions.json"
SAMPLE_CSV_PATH  = BASE_DIR / "sample_upload.csv"

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  {PASS}  {name}")
    else:
        msg = f"{name}" + (f" – {detail}" if detail else "")
        print(f"  {FAIL}  {msg}")
        failures.append(msg)


def main() -> None:
    samples = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))

    with app.test_client() as client:

        # ── /health ───────────────────────────────────────────────────────────
        print("\n[1] Health check")
        resp = client.get("/health")
        data = resp.get_json()
        check("status 200",             resp.status_code == 200)
        check("status field is 'ok'",   data.get("status") == "ok",
              f"got {data.get('status')!r}")
        check("feature_count > 0",      data.get("feature_count", 0) > 0)

        # ── /login ────────────────────────────────────────────────────────────
        print("\n[2] Login")
        resp = client.post("/login", json={"username": "admin", "password": "admin123"})
        data = resp.get_json()
        check("status 200",           resp.status_code == 200,
              f"got {resp.status_code}")
        check("status field is 'ok'", data.get("status") == "ok",
              f"got {data.get('status')!r}")

        # Wrong credentials
        resp_bad = client.post("/login", json={"username": "admin", "password": "wrong"})
        check("bad credentials → 401", resp_bad.status_code == 401,
              f"got {resp_bad.status_code}")

        # Re-login correctly for the rest of the tests
        client.post("/login", json={"username": "admin", "password": "admin123"})

        # ── /predict ──────────────────────────────────────────────────────────
        print("\n[3] Predictions")
        expected_labels = {"normal": "Normal", "fraud": "Fraud"}
        for name, payload in samples.items():
            resp = client.post("/predict", json=payload)
            data = resp.get_json()
            check(f"{name}: status 200", resp.status_code == 200,
                  f"got {resp.status_code}")

            label  = data.get("label")
            exp    = expected_labels[name]
            check(f"{name}: label is '{exp}'", label == exp,
                  f"got {label!r}")

            prob = data.get("fraud_probability")
            check(f"{name}: fraud_probability in [0,1]",
                  prob is not None and 0.0 <= prob <= 1.0,
                  f"got {prob}")

            risk = data.get("risk_score")
            check(f"{name}: risk_score in [0,100]",
                  risk is not None and 0 <= risk <= 100,
                  f"got {risk}")

            check(f"{name}: behavior signals present",
                  bool(data.get("behavior", {}).get("signals")))

            check(f"{name}: preprocessing keys present",
                  "feature_count" in data.get("preprocessing", {}))

        # ── /api/dashboard ────────────────────────────────────────────────────
        print("\n[4] Dashboard")
        resp = client.get("/api/dashboard")
        data = resp.get_json()
        check("status 200", resp.status_code == 200)
        stats = data.get("stats", {})
        check("total_transactions >= 2",
              stats.get("total_transactions", 0) >= 2,
              f"got {stats.get('total_transactions')}")
        check("transactions list present", isinstance(data.get("transactions"), list))
        check("alerts list present",       isinstance(data.get("alerts"),       list))

        # ── /api/sample-csv ───────────────────────────────────────────────────
        print("\n[5] Sample CSV download")
        resp = client.get("/api/sample-csv")
        check("status 200",       resp.status_code == 200,
              f"got {resp.status_code}")
        check("mimetype is text/csv", resp.mimetype == "text/csv",
              f"got {resp.mimetype!r}")

        # ── /api/upload-csv ───────────────────────────────────────────────────
        print("\n[6] CSV batch upload")
        with SAMPLE_CSV_PATH.open("rb") as csv_file:
            resp = client.post(
                "/api/upload-csv",
                data={"file": (csv_file, "sample_upload.csv")},
                content_type="multipart/form-data",
            )
        data = resp.get_json()
        check("status 200",     resp.status_code == 200,
              f"got {resp.status_code}")
        check("processed > 0",  data.get("processed", 0) > 0,
              f"got {data.get('processed')}")
        check("failed == 0",    data.get("failed", -1) == 0,
              f"got {data.get('failed')}")
        check("records list present", isinstance(data.get("records"), list))

        # Validate first record structure
        if data.get("records"):
            rec = data["records"][0]
            for key in ("row", "transaction_id", "label", "risk_score", "risk_level"):
                check(f"CSV record has '{key}'", key in rec)

        # ── /api/report.csv ───────────────────────────────────────────────────
        print("\n[7] Report CSV download")
        resp = client.get("/api/report.csv")
        check("status 200",           resp.status_code == 200,
              f"got {resp.status_code}")
        check("mimetype is text/csv", resp.mimetype == "text/csv",
              f"got {resp.mimetype!r}")
        check("non-empty body",       len(resp.data) > 0)

        # ── /logout ───────────────────────────────────────────────────────────
        print("\n[8] Logout")
        resp = client.post("/logout")
        check("status 200", resp.status_code == 200)

        # Authenticated endpoints should now return 401
        resp_unauth = client.get("/api/dashboard")
        check("dashboard requires auth after logout",
              resp_unauth.status_code == 401,
              f"got {resp_unauth.status_code}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    if failures:
        print(f"  {len(failures)} check(s) FAILED:")
        for f in failures:
            print(f"    • {f}")
        sys.exit(1)
    else:
        print("  All checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()