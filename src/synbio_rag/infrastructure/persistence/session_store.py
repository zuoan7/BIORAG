from __future__ import annotations

import json
from pathlib import Path

from ...domain.config import AuditConfig
from ...domain.schemas import ConversationTurn


class SessionStore:
    def __init__(self, config: AuditConfig):
        self.config = config
        self.path = Path(config.session_store_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append_turn(self, session_id: str, turn: ConversationTurn) -> None:
        data = self._read_all()
        turns = data.get(session_id, [])
        turns.append(turn.__dict__)
        data[session_id] = turns[-self.config.max_session_turns :]
        self._write_all(data)

    def get_turns(self, session_id: str) -> list[ConversationTurn]:
        data = self._read_all()
        return [ConversationTurn(**item) for item in data.get(session_id, [])]

    def _read_all(self) -> dict:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write_all(self, data: dict) -> None:
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
