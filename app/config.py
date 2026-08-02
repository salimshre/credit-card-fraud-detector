import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "fraud_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.json"

DEFAULT_STORE_PATH = BASE_DIR / "instance" / "data_store.json"
LEGACY_STORE_PATH = BASE_DIR / "data_store.json"
DEFAULT_DB_PATH = BASE_DIR / "instance" / "fraud_shield.db"

APP_USERNAME = os.getenv("APP_USERNAME", "admin")

# --- CRITICAL: Enforce a strong SECRET_KEY ---
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY or SECRET_KEY == "change-this-demo-secret":
    raise ValueError(
        "SECRET_KEY environment variable is not set or is using the insecure default. "
        "Generate a new key with: python -c 'import secrets; print(secrets.token_hex(32))'"
    )

REQUIRED_FIELDS = [
    "transaction_date",
    "transaction_time",
    "amount",
    "merchant_category",
    "country",
    "channel",
]
OPTIONAL_FIELDS = ["customer_id", "card_last4", "merchant", "device_id"]

MAX_RECORDS = 250
MAX_CSV_ROWS = 500

# ---- NEW NPR BUSINESS LIMITS ----
NPR_REVIEW_AMOUNT = 100_000      # Above this -> Review Required
NPR_BLOCK_AMOUNT = 300_000       # Above this -> Blocked


def resolve_store_path() -> Path:
    configured = os.getenv("FRAUD_STORE_PATH")
    if not configured:
        return DEFAULT_STORE_PATH
    path = Path(configured).expanduser()
    return path if path.is_absolute() else BASE_DIR / path


def resolve_db_path() -> Path:
    configured = os.getenv("FRAUD_DB_PATH")
    if not configured:
        return DEFAULT_DB_PATH
    path = Path(configured).expanduser()
    return path if path.is_absolute() else BASE_DIR / path


def load_threshold() -> float:
    configured = os.getenv("FRAUD_THRESHOLD")
    if configured not in (None, ""):
        return float(configured)

    if not METADATA_PATH.exists():
        return 0.5

    try:
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        return float(metadata.get("recommended_threshold", 0.5))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("Could not load model threshold from metadata: %s", exc)
        return 0.5


THRESHOLD = load_threshold()
DB_PATH = resolve_db_path()
