import json
import logging
import sqlite3
from pathlib import Path

from app.config import DB_PATH, DEFAULT_STORE_PATH, LEGACY_STORE_PATH

logger = logging.getLogger(__name__)

# In‑memory globals – keep exactly as before
TRANSACTIONS: list = []
ALERTS: list = []
BEHAVIOR_PROFILES: dict = {}


# --- Helper serialization for JSON fields ---

def _json_dumps(obj):
    return json.dumps(obj, default=str)

def _json_loads(data):
    if data is None:
        return {}
    return json.loads(data)


# --- Database initialisation ---

def init_db() -> None:
    """Create tables if they don't exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id TEXT PRIMARY KEY,
                created_at TEXT,
                operator TEXT,
                metadata TEXT,
                amount REAL,
                transaction_date TEXT,
                transaction_time TEXT,
                fraud_probability REAL,
                prediction INTEGER,
                label TEXT,
                threshold REAL,
                risk_score INTEGER,
                risk_level TEXT,
                behavior TEXT,
                preprocessing TEXT,
                verification_status TEXT,
                verification_note TEXT,
                verified_by TEXT,
                verified_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                transaction_id TEXT,
                created_at TEXT,
                severity TEXT,
                message TEXT,
                risk_score INTEGER,
                acknowledged INTEGER
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS behavior_profiles (
                customer_id TEXT PRIMARY KEY,
                transaction_count INTEGER,
                amount_total REAL,
                amount_avg REAL,
                amount_max REAL,
                recent_timestamps TEXT,
                devices TEXT,
                countries TEXT,
                merchant_categories TEXT
            )
        """)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.commit()
    logger.info(f"Database initialized at {DB_PATH}")


# --- Profile conversion (set↔list) ---

def profile_to_dict(profile: dict) -> dict:
    return {
        **profile,
        "devices": sorted(profile["devices"]),
        "countries": sorted(profile["countries"]),
        "merchant_categories": sorted(profile["merchant_categories"]),
    }

def dict_to_profile(data: dict) -> dict:
    return {
        **data,
        "devices": set(data.get("devices", [])),
        "countries": set(data.get("countries", [])),
        "merchant_categories": set(data.get("merchant_categories", [])),
    }


# --- Load from SQLite ---

def load_state() -> None:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        # Load transactions
        rows = conn.execute("SELECT * FROM transactions").fetchall()
        TRANSACTIONS.clear()
        for row in rows:
            rec = dict(row)
            rec["metadata"] = _json_loads(rec["metadata"])
            rec["behavior"] = _json_loads(rec["behavior"])
            rec["preprocessing"] = _json_loads(rec["preprocessing"])
            TRANSACTIONS.append(rec)

        # Load alerts
        rows = conn.execute("SELECT * FROM alerts").fetchall()
        ALERTS.clear()
        for row in rows:
            rec = dict(row)
            rec["acknowledged"] = bool(rec["acknowledged"])
            ALERTS.append(rec)

        # Load behavior profiles
        rows = conn.execute("SELECT * FROM behavior_profiles").fetchall()
        BEHAVIOR_PROFILES.clear()
        for row in rows:
            rec = dict(row)
            rec["recent_timestamps"] = _json_loads(rec["recent_timestamps"])
            rec["devices"] = set(_json_loads(rec["devices"]))
            rec["countries"] = set(_json_loads(rec["countries"]))
            rec["merchant_categories"] = set(_json_loads(rec["merchant_categories"]))
            BEHAVIOR_PROFILES[rec["customer_id"]] = rec

    logger.info(f"Loaded {len(TRANSACTIONS)} transactions, {len(ALERTS)} alerts, {len(BEHAVIOR_PROFILES)} profiles from DB.")


# --- Save to SQLite (full replace) ---

def save_state() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        # Clear existing data
        conn.execute("DELETE FROM transactions")
        conn.execute("DELETE FROM alerts")
        conn.execute("DELETE FROM behavior_profiles")

        # Insert transactions
        for rec in TRANSACTIONS:
            rec_copy = rec.copy()
            rec_copy["metadata"] = _json_dumps(rec_copy["metadata"])
            rec_copy["behavior"] = _json_dumps(rec_copy["behavior"])
            rec_copy["preprocessing"] = _json_dumps(rec_copy["preprocessing"])
            placeholders = ", ".join(["?"] * len(rec_copy))
            columns = ", ".join(rec_copy.keys())
            conn.execute(
                f"INSERT INTO transactions ({columns}) VALUES ({placeholders})",
                list(rec_copy.values())
            )

        # Insert alerts
        for rec in ALERTS:
            rec_copy = rec.copy()
            rec_copy["acknowledged"] = int(rec_copy["acknowledged"])
            placeholders = ", ".join(["?"] * len(rec_copy))
            columns = ", ".join(rec_copy.keys())
            conn.execute(
                f"INSERT INTO alerts ({columns}) VALUES ({placeholders})",
                list(rec_copy.values())
            )

        # Insert behavior profiles
        for customer_id, profile in BEHAVIOR_PROFILES.items():
            rec = profile_to_dict(profile)
            rec["customer_id"] = customer_id
            rec["recent_timestamps"] = _json_dumps(rec["recent_timestamps"])
            rec["devices"] = _json_dumps(rec["devices"])
            rec["countries"] = _json_dumps(rec["countries"])
            rec["merchant_categories"] = _json_dumps(rec["merchant_categories"])
            placeholders = ", ".join(["?"] * len(rec))
            columns = ", ".join(rec.keys())
            conn.execute(
                f"INSERT INTO behavior_profiles ({columns}) VALUES ({placeholders})",
                list(rec.values())
            )

        conn.commit()
    logger.debug("State saved to SQLite.")


# --- Optional: migrate from old JSON (if exists and DB empty) ---
def _migrate_from_json_if_needed() -> None:
    """One‑time migration of legacy JSON data into SQLite."""
    load_path = DEFAULT_STORE_PATH
    if (
        not load_path.exists()
        and DEFAULT_STORE_PATH == DEFAULT_STORE_PATH
        and LEGACY_STORE_PATH.exists()
    ):
        load_path = LEGACY_STORE_PATH

    if not load_path.exists():
        return

    with sqlite3.connect(DB_PATH) as conn:
        # Check if DB already has data
        count = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        if count > 0:
            return  # already has data

        # Load JSON
        try:
            import json
            state = json.loads(load_path.read_text(encoding="utf-8"))
            transactions = state.get("transactions", [])
            alerts = state.get("alerts", [])
            profiles = state.get("behavior_profiles", {})

            # Populate globals (they will be written to DB by save_state)
            TRANSACTIONS[:] = transactions
            ALERTS[:] = alerts
            BEHAVIOR_PROFILES.update({
                key: dict_to_profile(value)
                for key, value in profiles.items()
            })
            # Save to SQLite
            save_state()
            logger.info(f"Migrated {len(TRANSACTIONS)} transactions, {len(ALERTS)} alerts, {len(BEHAVIOR_PROFILES)} profiles from JSON.")
        except Exception as exc:
            logger.warning(f"Could not migrate JSON data: {exc}")


# Invoke the migration on first load
_migrate_from_json_if_needed()
