"""BIORAG Eval v1 — Qwen LLM-as-judge client with caching and error handling."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from .judge_prompts import PROMPT_BUILDERS, PROMPT_VERSIONS
from .schemas import metric_applicable

DEFAULT_MODEL = "qwen-plus"
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT = 60
DEFAULT_MAX_RETRIES = 2


class QwenJudgeClient:
    """Client for calling Qwen API as LLM judge.

    Features:
    - Explicit model/temperature/max_tokens
    - JSON-only output, no long rationale
    - Response cache (hash-based JSONL)
    - Parse error handling (doesn't treat error as score=0)
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        cache_path: str | Path | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_path = Path(cache_path) if cache_path else None

        self._cache: dict[str, dict[str, Any]] = {}
        self._client = None
        self._load_cache()

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            api_base = os.getenv("QWEN_CHAT_API_BASE", "")
            api_key = os.getenv("QWEN_CHAT_API_KEY", "")
            if not api_base or not api_key:
                raise RuntimeError("Missing QWEN_CHAT_API_BASE or QWEN_CHAT_API_KEY")
            self._client = OpenAI(
                base_url=api_base,
                api_key=api_key,
                timeout=self.timeout,
                max_retries=1,  # we do our own retry
            )
        return self._client

    # ── Cache ──────────────────────────────────────────────────────────
    def _cache_key(self, sample_id: str, metric_name: str, prompt: str) -> str:
        """Stable cache key: hash of inputs."""
        h = hashlib.sha256()
        h.update(sample_id.encode())
        h.update(metric_name.encode())
        h.update(PROMPT_VERSIONS.get(metric_name, "v1.0").encode())
        h.update(prompt.encode())
        return h.hexdigest()[:16]

    def _load_cache(self):
        if not self.cache_path or not self.cache_path.exists():
            return
        try:
            with open(self.cache_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    self._cache[entry.get("cache_key", "")] = entry
        except Exception:
            pass  # Corrupt cache → ignore, don't crash

    def _write_cache_entry(self, entry: dict[str, Any]):
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._cache[entry["cache_key"]] = entry

    # ── Core call ──────────────────────────────────────────────────────
    def judge(self, record: dict[str, Any], metric_name: str) -> dict[str, Any]:
        """Run a single judge metric on an EvalRecord.

        Returns dict with keys: metric_name, score, score_valid, cache_hit,
        judge_error_type, rationale, raw_preview.
        """
        sid = record["sample_id"]
        prompt = self._build_prompt(record, metric_name)
        ck = self._cache_key(sid, metric_name, prompt)

        # Cache hit
        if ck in self._cache:
            cached = self._cache[ck]
            if cached.get("metric_name") == metric_name:
                result = dict(cached)
                result["cache_hit"] = True
                return result

        # API call with retry
        client = self._get_client()
        for attempt in range(self.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                raw = response.choices[0].message.content or ""
                break
            except Exception as e:
                if attempt == self.max_retries:
                    entry = {
                        "cache_key": ck,
                        "sample_id": sid,
                        "metric_name": metric_name,
                        "score": None,
                        "score_valid": False,
                        "cache_hit": False,
                        "judge_error_type": f"api_error_after_{self.max_retries}_retries",
                        "rationale": str(e)[:80],
                        "raw_preview": str(e)[:200],
                    }
                    self._write_cache_entry(entry)
                    entry["cache_hit"] = False
                    return entry
                time.sleep(2**attempt)

        # Parse JSON
        try:
            parsed = self._parse_json(raw)
        except Exception:
            raw_preview = raw[:200]
            entry = {
                "cache_key": ck, "sample_id": sid, "metric_name": metric_name,
                "score": None, "score_valid": False, "cache_hit": False,
                "judge_error_type": "json_parse_error",
                "rationale": "failed to parse JSON from judge output",
                "raw_preview": raw_preview,
            }
            self._write_cache_entry(entry)
            entry["cache_hit"] = False
            return entry

        # Normalize primary score
        primary_key = self._primary_key(metric_name)
        score = parsed.get(primary_key)
        entry = {
            "cache_key": ck, "sample_id": sid, "metric_name": metric_name,
            "score": score, "score_valid": score is not None,
            "cache_hit": False, "judge_error_type": "",
            "rationale": str(parsed.get("rationale", ""))[:80],
            "raw_json": parsed,
            "prompt_version": PROMPT_VERSIONS.get(metric_name, "v1.0"),
        }
        self._write_cache_entry(entry)
        entry["cache_hit"] = False
        return entry

    # ── Helpers ────────────────────────────────────────────────────────
    def _build_prompt(self, record: dict[str, Any], metric_name: str) -> str:
        builder = PROMPT_BUILDERS.get(metric_name)
        if not builder:
            raise ValueError(f"Unknown metric: {metric_name}")
        return builder(record)

    def _parse_json(self, raw: str) -> dict[str, Any]:
        text = raw.strip()
        # Remove code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:])
            if text.endswith("```"):
                text = text[:-3]
        # Find first JSON object
        start = text.find("{")
        if start == -1:
            raise ValueError("No JSON object found")
        return json.loads(text[start:])

    def _primary_key(self, metric_name: str) -> str:
        """Map metric_name to primary score key in JSON output."""
        mapping = {
            "faithfulness": "faithfulness",
            "evidence_recall": "evidence_recall",
            "answer_accuracy": "answer_accuracy",
            "answer_relevance": "answer_relevance",
            "answer_completeness": "answer_completeness",
            "abstention_correctness": "abstention_correctness",
            "both_branches_covered": "both_branches_covered",
            "comparison_axis_covered": "comparison_axis_covered",
            "comparison_faithfulness": "comparison_faithfulness",
            "numeric_accuracy": "numeric_accuracy",
            "unit_correct": "unit_correct",
        }
        return mapping.get(metric_name, metric_name)

    def judge_all_applicable(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        """Run all applicable judge metrics for a record."""
        results = []
        for metric_name in PROMPT_BUILDERS:
            ok, reason = metric_applicable(record, metric_name)
            if not ok:
                results.append({
                    "sample_id": record["sample_id"], "metric_name": metric_name,
                    "score": None, "score_valid": False, "cache_hit": False,
                    "judge_error_type": "", "rationale": reason, "raw_preview": "",
                })
                continue
            results.append(self.judge(record, metric_name))
        return results
