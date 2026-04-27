from __future__ import annotations

import time
from dataclasses import replace

from ..domain.confidence import ConfidenceScorer
from ..domain.config import Settings
from ..domain.router import QueryRouter
from ..domain.schemas import ConversationTurn, QueryAnalysis, QueryFilters, RAGResponse
from ..infrastructure.embedding.bge import BGEM3Embedder
from ..infrastructure.external_tools.literature_search import ExternalToolManager
from ..infrastructure.vectorstores.bm25 import BM25Retriever
from ..infrastructure.vectorstores.hybrid import HybridRetriever
from ..infrastructure.vectorstores.milvus import MilvusRetriever
from .context_builder import ContextBuilder
from .generation_v2 import GenerationV2Service
from .generation_service import QwenChatGenerator
from .neighbor_expansion import ChunkNeighborExpander
from .rerank_service import QwenReranker


class SynBioRAGPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedder = BGEM3Embedder(
            model_path=settings.kb.embedding_model_path,
            dim=settings.kb.embedding_dim,
            max_length=settings.kb.embedding_max_length,
        )
        self.router = QueryRouter(settings.retrieval)
        self.dense_retriever = MilvusRetriever(settings.retrieval, self.embedder)
        self.bm25_retriever = BM25Retriever(
            retrieval_config=settings.retrieval,
            kb_config=settings.kb,
            milvus_client=self.dense_retriever.client,
        )
        self.retriever = HybridRetriever(
            config=settings.retrieval,
            dense_retriever=self.dense_retriever,
            bm25_retriever=self.bm25_retriever,
        )
        self.reranker = QwenReranker(
            api_base=settings.reranker.api_base,
            api_key=settings.reranker.api_key,
            model_name=settings.reranker.model_name,
            model_path=settings.reranker.model_path,
            service_url=settings.reranker.service_url,
            batch_size=settings.reranker.batch_size,
            use_fp16=settings.reranker.use_fp16,
            retrieval_config=settings.retrieval,
        )
        self.context_builder = ContextBuilder()
        self.neighbor_expander = ChunkNeighborExpander(
            kb_config=settings.kb,
            retrieval_config=settings.retrieval,
        )
        self.generator = QwenChatGenerator(
            api_base=settings.llm.api_base,
            api_key=settings.llm.api_key,
            model_name=settings.llm.model_name,
            temperature=settings.llm.temperature,
            round8_config=settings.round8,
        )
        self.generator_v2 = GenerationV2Service(settings.llm)
        self.confidence_scorer = ConfidenceScorer(settings.confidence)
        self.external_tools = ExternalToolManager(settings.tools)

    def answer(
        self,
        question: str,
        session_id: str | None = None,
        history: list[ConversationTurn] | None = None,
        filters: QueryFilters | None = None,
    ) -> RAGResponse:
        start = time.perf_counter()
        analysis = self.router.analyze(question)
        retrieved, retrieval_debug = self._search_with_filter_fallback(
            question=question,
            analysis=analysis,
            filters=filters,
        )
        reranked = self.reranker.rerank(
            question,
            retrieved,
            top_k=analysis.rerank_top_k,
            analysis=analysis,
        )
        seed_chunks = reranked[: self.settings.retrieval.final_top_k]
        if self.settings.generation.version == "v2":
            final_chunks = seed_chunks
            confidence = self.confidence_scorer.score(seed_chunks)
            gen_result = self.generator_v2.run(
                question=question,
                analysis=analysis,
                seed_chunks=seed_chunks,
                config=self.settings.generation,
                history=history if self.settings.generation.v2_use_history else None,
            )
            return RAGResponse(
                answer=gen_result.answer,
                confidence=confidence,
                route=analysis.intent,
                citations=gen_result.citations,
                used_external_tool=False,
                tool_name=None,
                tool_result=None,
                session_id=session_id,
                external_references=[],
                debug={
                    "analysis_notes": analysis.notes,
                    "retrieved_count": len(retrieved),
                    "reranked_count": len(reranked),
                    "seed_context_count": len(seed_chunks),
                    "final_context_count": len(final_chunks),
                    "context_chars": 0,
                    "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                    "tenant_id": filters.tenant_id if filters else "default",
                    "hybrid_enabled": self.settings.retrieval.hybrid_enabled,
                    "bm25_enabled": self.settings.retrieval.bm25_enabled,
                    "retrieval_hits": getattr(self.retriever, "last_debug", {}),
                    "rerank_hits": getattr(self.reranker, "last_debug", {}),
                    "neighbor_expansion": {
                        "enabled": False,
                        "reason": "generation_v2_seed_only",
                        "input_count": len(seed_chunks),
                        "output_count": len(seed_chunks),
                    },
                    "filter_strategy": retrieval_debug,
                    "generation_v2": gen_result.debug,
                },
            )
        final_chunks = self.neighbor_expander.expand(seed_chunks)
        context = self.context_builder.build(question, final_chunks, history=history, intent=analysis.intent)
        evidence_quality = self.generator.assess_evidence(question, final_chunks, analysis=analysis)
        answer = self.generator.generate(
            question,
            context,
            final_chunks,
            analysis=analysis,
            history=history,
            assessment=evidence_quality,
        )
        confidence = self.confidence_scorer.score(final_chunks)
        tool_execution = self.external_tools.run_if_needed(
            question=question,
            analysis=analysis,
            low_confidence=self.confidence_scorer.needs_external_tool(confidence),
        )
        citations = self.generator.build_citations(final_chunks, evidence_quality)
        answer = self.generator.validate_generated_answer(answer, citations, evidence_quality)

        return RAGResponse(
            answer=answer,
            confidence=confidence,
            route=analysis.intent,
            citations=citations,
            used_external_tool=tool_execution.invoked,
            tool_name=tool_execution.tool_name,
            tool_result=tool_execution.result,
            session_id=session_id,
            external_references=tool_execution.references,
            debug={
                "analysis_notes": analysis.notes,
                "retrieved_count": len(retrieved),
                "reranked_count": len(reranked),
                "seed_context_count": len(seed_chunks),
                "final_context_count": len(final_chunks),
                "context_chars": len(context),
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                "tenant_id": filters.tenant_id if filters else "default",
                "hybrid_enabled": self.settings.retrieval.hybrid_enabled,
                "bm25_enabled": self.settings.retrieval.bm25_enabled,
                "retrieval_hits": getattr(self.retriever, "last_debug", {}),
                "rerank_hits": getattr(self.reranker, "last_debug", {}),
                "neighbor_expansion": getattr(self.neighbor_expander, "last_debug", {}),
                "filter_strategy": retrieval_debug,
                "evidence_quality": evidence_quality.__dict__,
            },
        )

    def _search_with_filter_fallback(
        self,
        question: str,
        analysis: QueryAnalysis,
        filters: QueryFilters | None,
    ) -> tuple[list, dict[str, object]]:
        attempts: list[dict[str, object]] = []
        filter_plan = _build_filter_plan(filters)
        for name, candidate_filters in filter_plan:
            retrieved = self.retriever.search(
                question,
                limit=analysis.search_limit,
                filters=candidate_filters,
                analysis=analysis,
            )
            attempts.append(
                {
                    "name": name,
                    "filters": candidate_filters.__dict__ if candidate_filters else None,
                    "retrieved_count": len(retrieved),
                }
            )
            if retrieved:
                return retrieved, {"selected": name, "attempts": attempts}
        return [], {"selected": "empty", "attempts": attempts}


def _build_filter_plan(filters: QueryFilters | None) -> list[tuple[str, QueryFilters | None]]:
    if not filters:
        return [("original", None)]
    plan: list[tuple[str, QueryFilters | None]] = []
    if filters.sections and not filters.doc_ids and not filters.source_files:
        plan.append(("drop_sections", replace(filters, sections=[])))
        plan.append(("original", filters))
    else:
        plan.append(("original", filters))
    if filters.sections:
        plan.append(("drop_sections", replace(filters, sections=[])))
    if filters.sections and filters.source_files:
        plan.append(("doc_ids_only", replace(filters, sections=[], source_files=[])))
    elif filters.source_files:
        plan.append(("drop_source_files", replace(filters, source_files=[])))
    deduped: list[tuple[str, QueryFilters | None]] = []
    seen: set[tuple] = set()
    for name, candidate in plan:
        key = (
            tuple(candidate.doc_ids) if candidate else (),
            tuple(candidate.sections) if candidate else (),
            tuple(candidate.source_files) if candidate else (),
            candidate.min_score if candidate else None,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append((name, candidate))
    return deduped
