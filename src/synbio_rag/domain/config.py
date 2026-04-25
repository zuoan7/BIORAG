from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class ModelEndpointConfig:
    provider: str = "qwen"
    model_name: str = "qwen-placeholder"
    api_base: str = ""
    api_key: str = ""
    timeout_seconds: int = 30
    temperature: float = 0.1
    model_path: str = ""
    service_url: str = ""
    batch_size: int = 8
    use_fp16: bool = True


@dataclass
class RetrievalConfig:
    milvus_uri: str = "./runtime/vectorstores/milvus/papers.db"
    collection_name: str = "synbio_papers"
    vector_field: str = "embedding"
    bm25_cache_path: str = "./data/paper_round1/chunks/bm25_index.json"
    search_limit: int = 40
    dense_limit: int = 40
    bm25_limit: int = 40
    rerank_top_k: int = 10
    final_top_k: int = 8
    metric_type: str = "COSINE"
    ef: int = 64
    score_floor: float = 0.05
    max_filter_items: int = 20
    hybrid_enabled: bool = True
    bm25_enabled: bool = True
    rrf_k: int = 60
    dense_rrf_weight: float = 1.0
    bm25_rrf_weight: float = 1.0
    cjk_query_bm25_weight: float = 0.25
    comparison_query_weight: float = 1.1
    comparison_subquery_weight: float = 0.9
    comparison_max_chunks_per_doc: int = 1
    comparison_rerank_max_chunks_per_doc: int = 1
    title_keyword_boost: float = 0.08
    rerank_subquery_aggregate_alpha: float = 0.15
    rerank_strategy_bonus: float = 0.1
    evidence_numeric_bonus: float = 0.20
    evidence_result_bonus: float = 0.22
    evidence_definition_bonus: float = 0.06
    section_results_bonus: float = 0.35
    section_discussion_bonus: float = 0.28
    section_abstract_bonus: float = 0.0
    section_introduction_penalty: float = -0.05
    rerank_score_floor_ratio: float = 0.4
    neighbor_expansion_enabled: bool = True
    neighbor_window_size: int = 2
    neighbor_expansion_max_chunks: int = 30
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    bm25_batch_size: int = 1000


@dataclass
class ConfidenceConfig:
    answer_threshold: float = 0.58
    citation_threshold: float = 0.45
    low_confidence_message: str = (
        "现有知识库证据不足，建议触发外部工具或人工复核。"
    )


@dataclass
class AuditConfig:
    enabled: bool = True
    audit_log_path: str = "./runtime/logs/rag_audit.jsonl"
    session_store_path: str = "./runtime/logs/session_store.json"
    max_session_turns: int = 12
    include_context_in_logs: bool = False


@dataclass
class ToolConfig:
    enable_pubmed: bool = True
    enable_crossref: bool = True
    pubmed_base: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    crossref_base: str = "https://api.crossref.org/works"
    max_results: int = 3


@dataclass
class Round8EvidencePolicy:
    score_strength_factoid: float = 0.34
    score_strength_summary: float = 0.32
    score_strength_comparison: float = 0.30
    partial_boundary_factoid: float = 0.46
    partial_boundary_summary: float = 0.44
    partial_boundary_comparison: float = 0.44
    evidence_unit_threshold_factoid: float = 0.34
    evidence_unit_threshold_summary: float = 0.30
    evidence_unit_threshold_comparison: float = 0.28
    evidence_signal_weak_threshold: float = 0.32
    comparison_branch_coverage_threshold: float = 0.40


@dataclass
class Round8PolicyConfig:
    enable_round8_policy: bool = False
    disable_comparison_single_doc_hard_refusal: bool = False
    enable_claim_fallback: bool = False
    enable_partial_answer: bool = False
    enable_route_specific_thresholds: bool = False
    enable_comparison_prompt_v2: bool = False
    enable_citation_expansion: bool = False
    enable_ragas_retry: bool = False
    evidence_policy: Round8EvidencePolicy = field(default_factory=Round8EvidencePolicy)


@dataclass
class KnowledgeBaseConfig:
    paper_dir: str = "./data/paper_round1/paper"
    parsed_raw_dir: str = "./data/paper_round1/parsed_raw"
    parsed_dir: str = "./data/paper_round1/parsed_clean"
    parsed_preview_dir: str = "./data/paper_round1/parsed_preview"
    chunk_dir: str = "./data/paper_round1/chunks"
    chunk_jsonl: str = "./data/paper_round1/chunks/chunks.jsonl"
    embedding_model_path: str = "./models/BAAI/bge-m3"
    embedding_dim: int = 1024
    chunk_size: int = 800
    chunk_overlap: int = 120


@dataclass
class Settings:
    app_name: str = "synbio-rag-service"
    env: str = "dev"
    domain: str = "synthetic_biology"
    kb: KnowledgeBaseConfig = field(default_factory=KnowledgeBaseConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)
    round8: Round8PolicyConfig = field(default_factory=Round8PolicyConfig)
    llm: ModelEndpointConfig = field(default_factory=lambda: ModelEndpointConfig(model_name="qwen-plus"))
    reranker: ModelEndpointConfig = field(default_factory=lambda: ModelEndpointConfig(model_name="qwen-rerank"))

    @classmethod
    def from_env(cls) -> "Settings":
        settings = cls()
        env_file = dotenv_values(PROJECT_ROOT / ".env")

        def get_value(key: str, default: str) -> str:
            return os.getenv(key, env_file.get(key, default))

        settings.env = get_value("APP_ENV", settings.env)
        settings.retrieval.milvus_uri = get_value(
            "SYNBIO_MILVUS_URI", get_value("MILVUS_URI", settings.retrieval.milvus_uri)
        )
        settings.retrieval.collection_name = get_value(
            "MILVUS_COLLECTION",
            settings.retrieval.collection_name,
        )
        settings.retrieval.score_floor = float(
            get_value("RETRIEVAL_SCORE_FLOOR", str(settings.retrieval.score_floor))
        )
        settings.retrieval.dense_rrf_weight = float(
            get_value("RETRIEVAL_DENSE_RRF_WEIGHT", str(settings.retrieval.dense_rrf_weight))
        )
        settings.retrieval.bm25_rrf_weight = float(
            get_value("RETRIEVAL_BM25_RRF_WEIGHT", str(settings.retrieval.bm25_rrf_weight))
        )
        settings.retrieval.cjk_query_bm25_weight = float(
            get_value("RETRIEVAL_CJK_QUERY_BM25_WEIGHT", str(settings.retrieval.cjk_query_bm25_weight))
        )
        settings.retrieval.comparison_query_weight = float(
            get_value(
                "RETRIEVAL_COMPARISON_QUERY_WEIGHT",
                str(settings.retrieval.comparison_query_weight),
            )
        )
        settings.retrieval.comparison_subquery_weight = float(
            get_value(
                "RETRIEVAL_COMPARISON_SUBQUERY_WEIGHT",
                str(settings.retrieval.comparison_subquery_weight),
            )
        )
        settings.retrieval.comparison_max_chunks_per_doc = int(
            get_value(
                "RETRIEVAL_COMPARISON_MAX_CHUNKS_PER_DOC",
                str(settings.retrieval.comparison_max_chunks_per_doc),
            )
        )
        settings.retrieval.comparison_rerank_max_chunks_per_doc = int(
            get_value(
                "RETRIEVAL_COMPARISON_RERANK_MAX_CHUNKS_PER_DOC",
                str(settings.retrieval.comparison_rerank_max_chunks_per_doc),
            )
        )
        settings.retrieval.title_keyword_boost = float(
            get_value(
                "RETRIEVAL_TITLE_KEYWORD_BOOST",
                str(settings.retrieval.title_keyword_boost),
            )
        )
        settings.retrieval.rerank_subquery_aggregate_alpha = float(
            get_value(
                "RETRIEVAL_RERANK_SUBQUERY_AGGREGATE_ALPHA",
                str(settings.retrieval.rerank_subquery_aggregate_alpha),
            )
        )
        settings.retrieval.rerank_strategy_bonus = float(
            get_value(
                "RETRIEVAL_RERANK_STRATEGY_BONUS",
                str(settings.retrieval.rerank_strategy_bonus),
            )
        )
        settings.retrieval.search_limit = int(
            get_value("RETRIEVAL_SEARCH_LIMIT", str(settings.retrieval.search_limit))
        )
        settings.retrieval.dense_limit = int(
            get_value("RETRIEVAL_DENSE_LIMIT", str(settings.retrieval.dense_limit))
        )
        settings.retrieval.bm25_limit = int(
            get_value("RETRIEVAL_BM25_LIMIT", str(settings.retrieval.bm25_limit))
        )
        settings.retrieval.rerank_top_k = int(
            get_value("RETRIEVAL_RERANK_TOP_K", str(settings.retrieval.rerank_top_k))
        )
        settings.retrieval.final_top_k = int(
            get_value("RETRIEVAL_FINAL_TOP_K", str(settings.retrieval.final_top_k))
        )
        settings.retrieval.evidence_numeric_bonus = float(
            get_value("RETRIEVAL_EVIDENCE_NUMERIC_BONUS", str(settings.retrieval.evidence_numeric_bonus))
        )
        settings.retrieval.evidence_result_bonus = float(
            get_value("RETRIEVAL_EVIDENCE_RESULT_BONUS", str(settings.retrieval.evidence_result_bonus))
        )
        settings.retrieval.evidence_definition_bonus = float(
            get_value("RETRIEVAL_EVIDENCE_DEFINITION_BONUS", str(settings.retrieval.evidence_definition_bonus))
        )
        settings.retrieval.section_results_bonus = float(
            get_value("RETRIEVAL_SECTION_RESULTS_BONUS", str(settings.retrieval.section_results_bonus))
        )
        settings.retrieval.section_discussion_bonus = float(
            get_value("RETRIEVAL_SECTION_DISCUSSION_BONUS", str(settings.retrieval.section_discussion_bonus))
        )
        settings.retrieval.section_abstract_bonus = float(
            get_value("RETRIEVAL_SECTION_ABSTRACT_BONUS", str(settings.retrieval.section_abstract_bonus))
        )
        settings.retrieval.section_introduction_penalty = float(
            get_value("RETRIEVAL_SECTION_INTRODUCTION_PENALTY", str(settings.retrieval.section_introduction_penalty))
        )
        settings.retrieval.rerank_score_floor_ratio = float(
            get_value("RETRIEVAL_RERANK_SCORE_FLOOR_RATIO", str(settings.retrieval.rerank_score_floor_ratio))
        )
        settings.retrieval.neighbor_expansion_enabled = _parse_bool(
            get_value(
                "RETRIEVAL_NEIGHBOR_EXPANSION_ENABLED",
                str(settings.retrieval.neighbor_expansion_enabled),
            )
        )
        settings.retrieval.neighbor_window_size = int(
            get_value("RETRIEVAL_NEIGHBOR_WINDOW_SIZE", str(settings.retrieval.neighbor_window_size))
        )
        settings.retrieval.neighbor_expansion_max_chunks = int(
            get_value(
                "RETRIEVAL_NEIGHBOR_EXPANSION_MAX_CHUNKS",
                str(settings.retrieval.neighbor_expansion_max_chunks),
            )
        )
        settings.kb.embedding_model_path = get_value(
            "BGE_M3_MODEL_PATH",
            settings.kb.embedding_model_path,
        )
        settings.llm.api_base = get_value("QWEN_CHAT_API_BASE", "")
        settings.llm.api_key = get_value("QWEN_CHAT_API_KEY", "")
        settings.reranker.api_base = get_value("QWEN_RERANK_API_BASE", "")
        settings.reranker.api_key = get_value("QWEN_RERANK_API_KEY", "")
        settings.reranker.model_path = get_value(
            "BGE_RERANKER_MODEL_PATH",
            "./models/BAAI/bge-reranker-v2-m3",
        )
        settings.reranker.service_url = get_value("RERANKER_SERVICE_URL", "")
        settings.audit.audit_log_path = get_value("AUDIT_LOG_PATH", settings.audit.audit_log_path)
        settings.audit.session_store_path = get_value("SESSION_STORE_PATH", settings.audit.session_store_path)
        settings.round8.enable_round8_policy = _parse_bool(
            get_value("ROUND8_ENABLE_ROUND8_POLICY", str(settings.round8.enable_round8_policy))
        )
        settings.round8.disable_comparison_single_doc_hard_refusal = _parse_bool(
            get_value(
                "ROUND8_DISABLE_COMPARISON_SINGLE_DOC_HARD_REFUSAL",
                str(settings.round8.disable_comparison_single_doc_hard_refusal),
            )
        )
        settings.round8.enable_claim_fallback = _parse_bool(
            get_value("ROUND8_ENABLE_CLAIM_FALLBACK", str(settings.round8.enable_claim_fallback))
        )
        settings.round8.enable_partial_answer = _parse_bool(
            get_value("ROUND8_ENABLE_PARTIAL_ANSWER", str(settings.round8.enable_partial_answer))
        )
        settings.round8.enable_route_specific_thresholds = _parse_bool(
            get_value(
                "ROUND8_ENABLE_ROUTE_SPECIFIC_THRESHOLDS",
                str(settings.round8.enable_route_specific_thresholds),
            )
        )
        settings.round8.enable_comparison_prompt_v2 = _parse_bool(
            get_value(
                "ROUND8_ENABLE_COMPARISON_PROMPT_V2",
                str(settings.round8.enable_comparison_prompt_v2),
            )
        )
        settings.round8.enable_citation_expansion = _parse_bool(
            get_value(
                "ROUND8_ENABLE_CITATION_EXPANSION",
                str(settings.round8.enable_citation_expansion),
            )
        )
        settings.round8.enable_ragas_retry = _parse_bool(
            get_value("ROUND8_ENABLE_RAGAS_RETRY", str(settings.round8.enable_ragas_retry))
        )
        settings.resolve_paths()
        settings.ensure_directories()
        return settings

    def resolve_paths(self) -> None:
        self.retrieval.milvus_uri = _resolve_local_path(self.retrieval.milvus_uri)
        self.retrieval.bm25_cache_path = _resolve_local_path(self.retrieval.bm25_cache_path)
        self.kb.paper_dir = _resolve_local_path(self.kb.paper_dir)
        self.kb.parsed_raw_dir = _resolve_local_path(self.kb.parsed_raw_dir)
        self.kb.parsed_dir = _resolve_local_path(self.kb.parsed_dir)
        self.kb.parsed_preview_dir = _resolve_local_path(self.kb.parsed_preview_dir)
        self.kb.chunk_dir = _resolve_local_path(self.kb.chunk_dir)
        self.kb.chunk_jsonl = _resolve_local_path(self.kb.chunk_jsonl)
        self.kb.embedding_model_path = _resolve_local_path(self.kb.embedding_model_path)
        self.reranker.model_path = _resolve_local_path(self.reranker.model_path)
        self.audit.audit_log_path = _resolve_local_path(self.audit.audit_log_path)
        self.audit.session_store_path = _resolve_local_path(self.audit.session_store_path)

    def ensure_directories(self) -> None:
        for path_str in (
            self.kb.parsed_raw_dir,
            self.kb.parsed_dir,
            self.kb.parsed_preview_dir,
            self.kb.chunk_dir,
            str(Path(self.kb.chunk_jsonl).parent),
            str(Path(self.retrieval.bm25_cache_path).parent),
            str(Path(self.audit.audit_log_path).parent),
            str(Path(self.audit.session_store_path).parent),
        ):
            Path(path_str).mkdir(parents=True, exist_ok=True)
        if _is_local_path(self.retrieval.milvus_uri):
            Path(self.retrieval.milvus_uri).parent.mkdir(parents=True, exist_ok=True)


def _is_local_path(value: str) -> bool:
    return bool(value) and not value.startswith(("http://", "https://", "unix://", "tcp://"))


def _resolve_local_path(value: str) -> str:
    if not _is_local_path(value):
        return value
    return str((PROJECT_ROOT / value).resolve()) if not Path(value).is_absolute() else value


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
