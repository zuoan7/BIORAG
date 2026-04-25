from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...domain.config import AuditConfig


class AuditLogger:
    def __init__(self, config: AuditConfig):
        self.config = config
        self.path = Path(config.audit_log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, payload: dict[str, Any]) -> None:
        if not self.config.enabled:
            return
        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


def to_serializable(data: Any) -> Any:
    if hasattr(data, "__dataclass_fields__"):
        return {k: to_serializable(v) for k, v in asdict(data).items()}
    if isinstance(data, dict):
        return {k: to_serializable(v) for k, v in data.items()}
    if isinstance(data, list):
        return [to_serializable(item) for item in data]
    return data
