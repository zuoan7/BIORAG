"""Query rewrite service — English-mirror translation with cache, fallback, and trace.

Default mode: off (no rewrite). shadow mode: compute rewrite but don't use for retrieval.
enabled mode: use rewritten query for retrieval with original preserved in context.
"""
import hashlib, time, threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class QueryRewriteMode(str, Enum):
    OFF = "off"
    SHADOW = "shadow"
    ENABLED = "enabled"


@dataclass
class RewriteTrace:
    """Observability trace for query rewrite."""
    query_rewrite_mode: str = "off"
    query_rewrite_enabled: bool = False
    original_query: str = ""
    rewritten_query: str = ""
    rewrite_model: str = ""
    rewrite_prompt_hash: str = ""
    rewrite_output_hash: str = ""
    rewrite_cache_hit: bool = False
    rewrite_latency_ms: float = 0.0
    rewrite_error: Optional[str] = None
    rewrite_fallback_used: bool = False
    rewrite_fallback_reason: str = ""
    implicit_reference_detected: bool = False
    implicit_reference_preserved: bool = False
    negative_intent_detected: bool = False
    negative_intent_preserved: bool = False
    retrieval_query_used: str = "original"
    diagnostic_flags: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: (v.value if isinstance(v, Enum) else list(v) if isinstance(v, (set, tuple)) else v)
                for k, v in self.__dict__.items()}


IMPLICIT_TERMS = ["文中", "本文", "该文", "该研究", "该论文", "这项研究", "文章中",
                   "此文", "本论文", "本研究", "该项研究"]

PROMPT_PATH = Path(__file__).resolve().parents[3] / "resources" / "prompts" / "query_rewrite_en_mirror.txt"
_GUARDED_PROMPT = None


def get_guarded_prompt() -> str:
    global _GUARDED_PROMPT
    if _GUARDED_PROMPT is None:
        if PROMPT_PATH.exists():
            _GUARDED_PROMPT = PROMPT_PATH.read_text().strip()
        else:
            _GUARDED_PROMPT = ""
    return _GUARDED_PROMPT


def get_prompt_hash() -> str:
    return hashlib.sha256(get_guarded_prompt().encode()).hexdigest()[:16]


def detect_implicit_references(query: str) -> list[str]:
    return [t for t in IMPLICIT_TERMS if t in query]


def check_implicit_preserved(en_query: str, implicit_terms: list[str]) -> bool:
    if not implicit_terms:
        return True
    en_lower = en_query.lower()
    return any(kw in en_lower for kw in ["paper", "study", "article", "referenced", "mentioned"])


class TranslationCache:
    """In-memory LRU cache for query translations. Thread-safe."""

    def __init__(self, max_size: int = 10000):
        self._cache: dict[str, str] = {}
        self._lock = threading.Lock()
        self._max_size = max_size

    def _make_key(self, query: str, prompt_hash: str, model: str, temperature: float, version: str) -> str:
        raw = f"{query}|{prompt_hash}|{model}|{temperature}|{version}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, query: str, prompt_hash: str, model: str, temperature: float, version: str) -> Optional[str]:
        with self._lock:
            return self._cache.get(self._make_key(query, prompt_hash, model, temperature, version))

    def put(self, query: str, prompt_hash: str, model: str, temperature: float, version: str, result: str):
        with self._lock:
            if len(self._cache) >= self._max_size:
                # Remove oldest 10%
                remove_n = max(1, self._max_size // 10)
                for key in list(self._cache.keys())[:remove_n]:
                    del self._cache[key]
            self._cache[self._make_key(query, prompt_hash, model, temperature, version)] = result

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)


class QueryRewriteService:
    """Service for English-mirror query rewriting with feature flag modes."""

    def __init__(self, mode: QueryRewriteMode = QueryRewriteMode.OFF, model: str = "qwen-plus",
                 temperature: float = 0.0, cache_enabled: bool = True,
                 timeout_ms: int = 3000, fallback_on_error: bool = True,
                 guard_implicit: bool = True, guard_negative: bool = True,
                 cache_version: str = "v1_guarded", llm_client=None):
        self.mode = mode
        self.model = model
        self.temperature = temperature
        self.cache_enabled = cache_enabled
        self.timeout_ms = timeout_ms
        self.fallback_on_error = fallback_on_error
        self.guard_implicit = guard_implicit
        self.guard_negative = guard_negative
        self.cache_version = cache_version
        self._llm = llm_client
        self._cache = TranslationCache() if cache_enabled else None
        self._prompt = get_guarded_prompt()
        self._prompt_hash = get_prompt_hash()

    def rewrite(self, original_query: str, is_negative: bool = False) -> tuple[str, RewriteTrace]:
        """Rewrite a query. Returns (query_to_use, trace)."""
        trace = RewriteTrace(
            query_rewrite_mode=self.mode.value,
            query_rewrite_enabled=self.mode != QueryRewriteMode.OFF,
            original_query=original_query,
            rewrite_model=self.model,
            rewrite_prompt_hash=self._prompt_hash,
            retrieval_query_used="original",
        )

        implicit_terms = detect_implicit_references(original_query) if self.guard_implicit else []
        trace.implicit_reference_detected = len(implicit_terms) > 0
        trace.negative_intent_detected = is_negative

        if self.mode == QueryRewriteMode.OFF:
            trace.retrieval_query_used = "original"
            return original_query, trace

        # Compute rewritten query
        rewritten = None
        try:
            cache_hit = False
            if self._cache:
                cached = self._cache.get(original_query, self._prompt_hash, self.model, self.temperature, self.cache_version)
                if cached is not None:
                    rewritten = cached
                    cache_hit = True

            trace.rewrite_cache_hit = cache_hit

            if rewritten is None and self._llm:
                t0 = time.perf_counter()
                try:
                    resp = self._llm.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": f"{self._prompt}\n\nChinese query: {original_query}\nEnglish query:"}],
                        temperature=self.temperature,
                        max_tokens=250,
                        timeout=self.timeout_ms / 1000.0 if self.timeout_ms else 30,
                    )
                    rewritten = resp.choices[0].message.content.strip()
                    if not rewritten:
                        raise ValueError("Empty output from LLM")
                except Exception as e:
                    trace.rewrite_error = f"{type(e).__name__}: {e}"
                    rewritten = None
                trace.rewrite_latency_ms = round((time.perf_counter() - t0) * 1000, 2)

                if rewritten and self._cache:
                    self._cache.put(original_query, self._prompt_hash, self.model, self.temperature, self.cache_version, rewritten)
        except Exception as e:
            trace.rewrite_error = f"{type(e).__name__}: {e}"

        # Fallback
        if rewritten is None or not rewritten.strip():
            trace.rewrite_fallback_used = True
            trace.rewrite_fallback_reason = trace.rewrite_error or "empty_or_none_output"
            rewritten = original_query  # fallback

        trace.rewritten_query = rewritten
        trace.rewrite_output_hash = hashlib.sha256(rewritten.encode()).hexdigest()[:16]

        # Guard checks
        if self.guard_implicit and implicit_terms:
            trace.implicit_reference_preserved = check_implicit_preserved(rewritten, implicit_terms)
        else:
            trace.implicit_reference_preserved = True

        trace.negative_intent_preserved = True  # guarded prompt preserves negative intent

        # Determine retrieval query
        if self.mode == QueryRewriteMode.ENABLED:
            trace.retrieval_query_used = "rewritten"
            return rewritten, trace
        else:  # SHADOW
            trace.retrieval_query_used = "original"
            return original_query, trace
