from __future__ import annotations

import json
from typing import Any

import httpx


class OpenAICompatibleClient:
    def __init__(self, api_base: str, api_key: str, timeout_seconds: int = 30):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def is_enabled(self) -> bool:
        return bool(self.api_base and self.api_key)

    def chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        if not self.is_enabled():
            raise RuntimeError("OpenAI-compatible endpoint is not configured")

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            payload["response_format"] = response_format

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                f"{self.api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            body = response.json()
        return body["choices"][0]["message"]["content"]


def extract_json_block(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    start = min((idx for idx in (stripped.find("{"), stripped.find("[")) if idx != -1), default=-1)
    if start == -1:
        raise ValueError("No JSON block found in model output")
    return json.loads(stripped[start:])
