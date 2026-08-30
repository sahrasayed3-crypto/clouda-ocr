import os
from .constants import API_KEY_FILE, LEGACY_API_KEY_FILE, SECRETS_DIR


def load_saved_api_key() -> str:
    try:
        if os.path.exists(API_KEY_FILE):
            with open(API_KEY_FILE, "r", encoding="utf-8") as f:
                return f.read().strip()
        if os.path.exists(LEGACY_API_KEY_FILE):
            with open(LEGACY_API_KEY_FILE, "r", encoding="utf-8") as f:
                legacy_key = f.read().strip()
            if legacy_key:
                save_api_key_local(legacy_key)
                return legacy_key
    except Exception:
        return ""
    return ""


def save_api_key_local(key: str) -> bool:
    try:
        os.makedirs(SECRETS_DIR, exist_ok=True)
        with open(API_KEY_FILE, "w", encoding="utf-8") as f:
            f.write((key or "").strip())
        return True
    except Exception:
        return False


def delete_saved_api_key() -> bool:
    try:
        if os.path.exists(API_KEY_FILE):
            os.remove(API_KEY_FILE)
        return True
    except Exception:
        return False
