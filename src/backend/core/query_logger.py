from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


LOG_FILENAME = "llm_router.log"


def _resolve_log_path(log_dir: str) -> str:
    target_dir = log_dir or "log"
    os.makedirs(target_dir, exist_ok=True)
    return str(Path(target_dir) / LOG_FILENAME)


def clear_log_file(log_dir: str) -> str:
    path = _resolve_log_path(log_dir)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("")
    return path


def append_log(log_dir: str, event: Dict[str, Any]) -> str:
    path = _resolve_log_path(log_dir)
    payload = dict(event or {})
    payload["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    line = json.dumps(payload, ensure_ascii=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    return path
