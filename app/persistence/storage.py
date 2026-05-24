import json
import logging
from pathlib import Path

from app.config import DEFAULT_STORE_PATH, LEGACY_STORE_PATH, resolve_store_path

logger = logging.getLogger(__name__)

STORE_PATH: Path = resolve_store_path()

TRANSACTIONS: list = []
ALERTS: list = []
BEHAVIOR_PROFILES: dict = {}


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


def save_state() -> None:
    try:
        STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "transactions": TRANSACTIONS,
            "alerts": ALERTS,
            "behavior_profiles": {
                key: profile_to_dict(value)
                for key, value in BEHAVIOR_PROFILES.items()
            },
        }
        STORE_PATH.write_text(
            json.dumps(state, indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("Could not save state: %s", exc)


def load_state() -> None:
    load_path = STORE_PATH
    if (
        not load_path.exists()
        and STORE_PATH == DEFAULT_STORE_PATH
        and LEGACY_STORE_PATH.exists()
    ):
        load_path = LEGACY_STORE_PATH

    if not load_path.exists():
        return

    try:
        state = json.loads(load_path.read_text(encoding="utf-8"))
        TRANSACTIONS[:] = state.get("transactions", [])
        ALERTS[:] = state.get("alerts", [])
        BEHAVIOR_PROFILES.update(
            {
                key: dict_to_profile(value)
                for key, value in state.get("behavior_profiles", {}).items()
            }
        )
        logger.info(
            "Loaded %d transactions, %d alerts, %d profiles from %s.",
            len(TRANSACTIONS),
            len(ALERTS),
            len(BEHAVIOR_PROFILES),
            load_path,
        )
    except Exception as exc:
        logger.warning("Could not load state: %s", exc)
