from __future__ import annotations

import httpx


class RerankerServiceClient:
    def __init__(self, service_url: str, timeout_seconds: int = 60):
        self.service_url = service_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def score(self, query: str, documents: list[str]) -> list[float]:
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                f"{self.service_url}/v1/rerank",
                json={"query": query, "documents": documents},
            )
            response.raise_for_status()
            payload = response.json()
        return [float(item["score"]) for item in payload["items"]]
