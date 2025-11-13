"""Utility helpers for the FastAPI app."""
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Any

HISTORY_PATH = Path(__file__).resolve().parent / "history.json"


def save_history(entry: Dict[str, Any]):
    try:
        if HISTORY_PATH.exists():
            data = json.loads(HISTORY_PATH.read_text())
        else:
            data = []
    except Exception:
        data = []
    entry = {**entry, "ts": datetime.utcnow().isoformat()}
    data.append(entry)
    HISTORY_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def validate_input(body: Dict[str, Any]) -> str:
    text = (body.get("text") or "").strip()
    if not text:
        raise ValueError("`text` is required and cannot be empty")
    return text
