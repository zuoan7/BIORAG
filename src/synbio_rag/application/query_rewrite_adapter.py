from __future__ import annotations

from ..domain.config import Settings
from ..infrastructure.clients.openai_compatible import OpenAICompatibleClient
from ..rewrite.query_rewrite_service import QueryRewriteMode, QueryRewriteService


def build_query_rewrite_service(settings: Settings) -> QueryRewriteService:
    qrc = settings.query_rewrite
    rewrite_llm_client, rewrite_llm_error = _build_query_rewrite_llm_client(settings)
    return QueryRewriteService(
        mode=QueryRewriteMode(qrc.mode),
        model=qrc.model, temperature=qrc.temperature,
        cache_enabled=qrc.cache_enabled, timeout_ms=qrc.timeout_ms,
        fallback_on_error=qrc.fallback_on_error,
        guard_implicit=qrc.guard_implicit_reference,
        guard_negative=qrc.guard_negative_intent,
        cache_version=qrc.cache_key_version,
        llm_client=rewrite_llm_client,
        llm_client_error=rewrite_llm_error,
        eval_cache_path=qrc.eval_rewrite_cache_path,
        eval_require_cache=qrc.eval_rewrite_require_cache,
        eval_fail_fast_on_missing=qrc.eval_rewrite_fail_fast_on_missing,
    )


def _build_query_rewrite_llm_client(settings: Settings):
    qrc = settings.query_rewrite
    mode = QueryRewriteMode(qrc.mode)
    if mode == QueryRewriteMode.OFF:
        return None, ""

    api_base = settings.llm.api_base
    api_key = settings.llm.api_key
    if not api_base or not api_key:
        message = "query_rewrite_llm_client_unavailable: missing QWEN_CHAT_API_BASE or QWEN_CHAT_API_KEY"
        if mode == QueryRewriteMode.ENABLED and qrc.require_llm_for_eval:
            raise RuntimeError(message)
        return None, message

    timeout_seconds = qrc.timeout_ms / 1000.0 if qrc.timeout_ms else settings.llm.timeout_seconds
    try:
        client = OpenAICompatibleClient(
            api_base=api_base,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        message = f"query_rewrite_llm_client_init_failed: {type(exc).__name__}: {exc}"
        if mode == QueryRewriteMode.ENABLED and qrc.require_llm_for_eval:
            raise RuntimeError(message) from exc
        return None, message
    return client, ""
