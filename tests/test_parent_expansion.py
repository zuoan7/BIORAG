from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from src.synbio_rag.application.parent_expansion import ParentContextExpander
from src.synbio_rag.application.pipeline import SynBioRAGPipeline
from src.synbio_rag.domain.config import GenerationConfig, RetrievalConfig, Settings
from src.synbio_rag.domain.schemas import QueryAnalysis, QueryIntent, RetrievedChunk
from src.synbio_rag.infrastructure.index.parent_store import ParentStore


def _analysis(intent: QueryIntent) -> QueryAnalysis:
    return QueryAnalysis(intent=intent, requires_external_tools=False, search_limit=5, rerank_top_k=5)


def _chunk(
    chunk_id: str,
    doc_id: str,
    chunk_index: int,
    *,
    section: str = "Body",
    section_path: list[str] | None = None,
    page_numbers: list[int] | None = None,
    evidence_types: list[str] | None = None,
    table_caption: bool = False,
    figure_caption: bool = False,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "source_file": f"{doc_id}.pdf",
        "title": f"title-{doc_id}",
        "section": section,
        "section_path": section_path if section_path is not None else [section],
        "chunk_index": chunk_index,
        "text": f"text for {chunk_id}",
        "page_start": (page_numbers or [chunk_index])[0],
        "page_end": (page_numbers or [chunk_index])[-1],
        "page_numbers": page_numbers or [chunk_index],
        "block_ids": [f"b{chunk_index}"],
        "source_block_ids": [f"sb{chunk_index}"],
        "evidence_types": evidence_types or [],
        "contains_table_caption": table_caption,
        "contains_figure_caption": figure_caption,
        "contains_table_text": False,
        "contains_image": False,
        "contains_references": False,
        "contains_metadata": False,
        "contains_noise": False,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_store(tmp_path: Path, chunks: list[dict]) -> tuple[ParentStore, RetrievalConfig]:
    from scripts.ingestion.build_parent_index import build_parent_records

    chunk_path = tmp_path / "chunks.jsonl"
    parent_path = tmp_path / "parents.jsonl"
    parents = build_parent_records(chunks, window_size=1, caption_context_max_children=5)
    _write_jsonl(chunk_path, chunks)
    _write_jsonl(parent_path, parents)
    config = RetrievalConfig(
        parent_expansion_enabled=True,
        parent_index_path=str(parent_path),
        parent_expansion_max_total=12,
        parent_expansion_per_seed_limit=2,
    )
    return ParentStore.from_jsonl(parent_path, chunk_jsonl_path=chunk_path), config


def _build_custom_store(tmp_path: Path, chunks: list[dict], parents: list[dict]) -> tuple[ParentStore, RetrievalConfig]:
    chunk_path = tmp_path / "chunks.jsonl"
    parent_path = tmp_path / "parents.jsonl"
    _write_jsonl(chunk_path, chunks)
    _write_jsonl(parent_path, parents)
    config = RetrievalConfig(
        parent_expansion_enabled=True,
        parent_index_path=str(parent_path),
        parent_expansion_max_total=12,
        parent_expansion_per_seed_limit=2,
    )
    return ParentStore.from_jsonl(parent_path, chunk_jsonl_path=chunk_path), config


def _seed_from_chunk(chunk: dict) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk["chunk_id"],
        doc_id=chunk["doc_id"],
        source_file=chunk["source_file"],
        title=chunk["title"],
        section=chunk["section"],
        text=chunk["text"],
        page_start=chunk["page_start"],
        page_end=chunk["page_end"],
        vector_score=0.8,
        bm25_score=0.3,
        rerank_score=0.6,
        fusion_score=0.7,
        metadata={
            "chunk_index": chunk["chunk_index"],
            "page_numbers": chunk["page_numbers"],
            "section_path": chunk["section_path"],
            "evidence_types": chunk.get("evidence_types", []),
            "contains_table_caption": chunk.get("contains_table_caption", False),
            "contains_figure_caption": chunk.get("contains_figure_caption", False),
            "contains_table_text": chunk.get("contains_table_text", False),
            "contains_image": chunk.get("contains_image", False),
        },
    )


def test_disabled_returns_original_seed_order(tmp_path: Path):
    chunks = [_chunk("c1", "d1", 1), _chunk("c2", "d1", 2)]
    store, config = _build_store(tmp_path, chunks)
    config.parent_expansion_enabled = False
    expander = ParentContextExpander(store, config)
    seeds = [_seed_from_chunk(chunks[0]), _seed_from_chunk(chunks[1])]
    final_chunks, debug = expander.expand("what is this?", seeds, _analysis(QueryIntent.FACTOID))
    assert [chunk.chunk_id for chunk in final_chunks] == ["c1", "c2"]
    assert debug["reason"] == "disabled"


def test_factoid_uses_single_chunk_window_neighbor(tmp_path: Path):
    chunks = [_chunk("c1", "d1", 1), _chunk("c2", "d1", 2), _chunk("c3", "d1", 3)]
    store, config = _build_store(tmp_path, chunks)
    expander = ParentContextExpander(store, config)
    final_chunks, debug = expander.expand("what is the result?", [_seed_from_chunk(chunks[1])], _analysis(QueryIntent.FACTOID))
    assert [chunk.chunk_id for chunk in final_chunks] == ["c2", "c1"]
    assert debug["effective_max_total"] <= 10
    assert debug["effective_per_seed_limit"] == 1
    assert debug["added_parent_types"] == ["chunk_window"]


def test_multi_doc_table_hint_only_expands_chunk_window_for_top_doc(tmp_path: Path):
    chunks = [
        _chunk("d1_seed", "d1", 2),
        _chunk("d1_peer", "d1", 1),
        _chunk("d2_seed", "d2", 2),
        _chunk("d2_peer", "d2", 1),
    ]
    store, config = _build_store(tmp_path, chunks)
    config.parent_expansion_section_path_enabled = False
    expander = ParentContextExpander(store, config)
    seeds = [_seed_from_chunk(chunks[0]), _seed_from_chunk(chunks[2])]
    final_chunks, debug = expander.expand(
        "What are the specific activity, yield, and purification fold for the enzyme?",
        seeds,
        _analysis(QueryIntent.FACTOID),
    )
    assert debug["primary_doc_window_gating"] is True
    assert debug["primary_doc_local_context_gating"] is True
    assert debug["window_target_doc_id"] == "d1"
    assert debug["local_context_target_doc_id"] == "d1"
    assert "d2_seed" in debug["window_skipped_non_target_doc"]
    assert [chunk.chunk_id for chunk in final_chunks] == ["d1_seed", "d2_seed", "d1_peer"]


def test_multi_doc_table_hint_only_expands_section_path_for_top_doc(tmp_path: Path):
    chunks = [
        _chunk("d1_seed", "d1", 2, section="Results", section_path=["Results", "Assay"]),
        _chunk("d1_peer", "d1", 1, section="Results", section_path=["Results", "Assay"]),
        _chunk("d2_seed", "d2", 2, section="Results", section_path=["Results", "Assay"]),
        _chunk("d2_peer", "d2", 1, section="Results", section_path=["Results", "Assay"]),
    ]
    store, config = _build_store(tmp_path, chunks)
    config.parent_expansion_window_enabled = False
    expander = ParentContextExpander(store, config)
    seeds = [_seed_from_chunk(chunks[0]), _seed_from_chunk(chunks[2])]
    final_chunks, debug = expander.expand(
        "What are the specific activity, yield, and purification fold for the enzyme?",
        seeds,
        _analysis(QueryIntent.FACTOID),
    )
    assert debug["primary_doc_local_context_gating"] is True
    assert debug["local_context_target_doc_id"] == "d1"
    assert "d2_seed" in debug["local_context_skipped_non_target_doc"]
    assert "section_path" in debug["local_context_blocked_parent_types"]
    assert "d2_seed" in debug["section_path_skipped_non_target_doc"]
    assert [chunk.chunk_id for chunk in final_chunks] == ["d1_seed", "d2_seed", "d1_peer"]


def test_multi_doc_plain_factoid_does_not_enable_primary_doc_window_gating(tmp_path: Path):
    chunks = [
        _chunk("d1_seed", "d1", 2),
        _chunk("d1_peer", "d1", 1),
        _chunk("d2_seed", "d2", 2),
        _chunk("d2_peer", "d2", 1),
    ]
    store, config = _build_store(tmp_path, chunks)
    expander = ParentContextExpander(store, config)
    seeds = [_seed_from_chunk(chunks[0]), _seed_from_chunk(chunks[2])]
    _final_chunks, debug = expander.expand("What is this system?", seeds, _analysis(QueryIntent.FACTOID))
    assert debug["primary_doc_window_gating"] is False
    assert debug["primary_doc_local_context_gating"] is False


def test_summary_adds_results_or_conclusion_and_debug(tmp_path: Path):
    chunks = [
        _chunk("abs", "d1", 1, section="Abstract"),
        _chunk("res", "d1", 2, section="Results"),
        _chunk("conc", "d1", 3, section="Conclusion"),
    ]
    store, config = _build_store(tmp_path, chunks)
    expander = ParentContextExpander(store, config)
    final_chunks, debug = expander.expand("summary this paper", [_seed_from_chunk(chunks[0])], _analysis(QueryIntent.SUMMARY))
    final_ids = [chunk.chunk_id for chunk in final_chunks]
    assert final_ids[0] == "abs"
    assert "res" in final_ids
    assert "conc" in final_ids
    assert any("d1:results" == item for item in debug["summary_sections_added"])


def test_comparison_effective_limits_and_top4_only(tmp_path: Path):
    chunks = [
        _chunk("a1", "d1", 1), _chunk("a2", "d1", 2), _chunk("a3", "d1", 3),
        _chunk("b1", "d2", 1), _chunk("b2", "d2", 2), _chunk("b3", "d2", 3),
        _chunk("c1", "d3", 1), _chunk("c2", "d3", 2), _chunk("c3", "d3", 3),
    ]
    store, config = _build_store(tmp_path, chunks)
    expander = ParentContextExpander(store, config)
    seeds = [_seed_from_chunk(chunks[1]), _seed_from_chunk(chunks[4]), _seed_from_chunk(chunks[7]), _seed_from_chunk(chunks[0]), _seed_from_chunk(chunks[3])]
    final_chunks, debug = expander.expand("compare these systems", seeds, _analysis(QueryIntent.COMPARISON))
    assert debug["comparison_mode"] is True
    assert debug["effective_max_total"] <= 8
    assert debug["effective_per_seed_limit"] == 1
    assert debug["comparison_seed_considered"] == ["a2", "b2", "c2", "a1"]
    assert debug["comparison_seed_skipped_by_rank"] == ["b1"]
    assert debug["per_seed_added"].get("a2", 0) <= 1
    assert debug["per_seed_added"].get("b2", 0) <= 1
    assert all(value <= 1 for value in debug["per_doc_added"].values())
    assert len(final_chunks) <= 8


def test_ordinary_comparison_does_not_trigger_caption_context(tmp_path: Path):
    chunks = [
        _chunk("c1", "d1", 1),
        _chunk("cap", "d1", 2, figure_caption=True, page_numbers=[2]),
        _chunk("c3", "d1", 3),
    ]
    store, config = _build_store(tmp_path, chunks)
    expander = ParentContextExpander(store, config)
    _final_chunks, debug = expander.expand("比较两种 AOX1 表达盒 策略", [_seed_from_chunk(chunks[1])], _analysis(QueryIntent.COMPARISON))
    assert debug["comparison_caption_allowed"] is False
    assert "caption_context" not in debug["added_parent_types"]


def test_expression_cassette_queries_do_not_trigger_table_query(tmp_path: Path):
    chunks = [
        _chunk("cap", "d1", 1, table_caption=True, page_numbers=[1]),
        _chunk("peer", "d1", 2, table_caption=True, page_numbers=[1]),
    ]
    store, config = _build_store(tmp_path, chunks)
    expander = ParentContextExpander(store, config)
    final_chunks, debug = expander.expand("What does the expression cassette change?", [_seed_from_chunk(chunks[0])], _analysis(QueryIntent.FACTOID))
    assert debug["table_query"] is False
    assert debug["caption_mode"] is False
    assert debug["caption_mode_trigger_source"] == "disabled"
    assert debug["false_table_trigger_guarded"] is True
    assert [chunk.chunk_id for chunk in final_chunks] == ["cap", "peer"]


def test_chinese_expression_box_does_not_trigger_table_query(tmp_path: Path):
    chunks = [
        _chunk("cap", "d1", 1, table_caption=True, page_numbers=[1]),
        _chunk("peer", "d1", 2, table_caption=True, page_numbers=[1]),
    ]
    store, config = _build_store(tmp_path, chunks)
    expander = ParentContextExpander(store, config)
    _final_chunks, debug = expander.expand("比较 AOX1 表达盒 的拷贝策略", [_seed_from_chunk(chunks[0])], _analysis(QueryIntent.FACTOID))
    assert debug["table_query"] is False
    assert debug["caption_mode"] is False


def test_explicit_table_query_triggers_table_mode(tmp_path: Path):
    chunks = [_chunk("cap", "d1", 1, table_caption=True, page_numbers=[1])]
    store, config = _build_store(tmp_path, chunks)
    expander = ParentContextExpander(store, config)
    _final_chunks, debug = expander.expand("What is listed in Table 1 primer table?", [_seed_from_chunk(chunks[0])], _analysis(QueryIntent.FACTOID))
    assert debug["table_query"] is True
    assert debug["caption_query_type"] == "table"
    assert debug["caption_mode"] is True


def test_explicit_figure_query_triggers_figure_mode(tmp_path: Path):
    chunks = [_chunk("cap", "d1", 1, figure_caption=True, page_numbers=[1])]
    store, config = _build_store(tmp_path, chunks)
    expander = ParentContextExpander(store, config)
    _final_chunks, debug = expander.expand("What is shown in Figure 6?", [_seed_from_chunk(chunks[0])], _analysis(QueryIntent.FACTOID))
    assert debug["figure_query"] is True
    assert debug["caption_query_type"] == "figure"
    assert debug["caption_mode"] is True


def test_table_figure_comparison_can_trigger_caption_context(tmp_path: Path):
    chunks = [
        _chunk("c1", "d1", 1, page_numbers=[1]),
        _chunk("cap", "d1", 2, figure_caption=True, page_numbers=[2]),
        _chunk("c3", "d1", 3, page_numbers=[2]),
    ]
    store, config = _build_store(tmp_path, chunks)
    expander = ParentContextExpander(store, config)
    final_chunks, debug = expander.expand("compare figure 1 and figure 2", [_seed_from_chunk(chunks[1])], _analysis(QueryIntent.COMPARISON))
    assert debug["comparison_caption_allowed"] is True
    assert debug["primary_doc_local_context_gating"] is False
    assert "caption_context" in debug["added_parent_types"]
    assert [chunk.chunk_id for chunk in final_chunks] == ["cap", "c3"]


def test_figure_query_does_not_add_pure_table_caption_child(tmp_path: Path):
    chunks = [
        _chunk("seed", "d1", 1, figure_caption=True, page_numbers=[2]),
        _chunk("tbl", "d1", 2, table_caption=True, page_numbers=[2]),
        _chunk("fig", "d1", 3, figure_caption=True, page_numbers=[2]),
    ]
    parents = [
        {
            "parent_id": "cap-parent",
            "parent_type": "caption_context",
            "doc_id": "d1",
            "anchor_chunk_id": "seed",
            "child_chunk_ids": ["seed", "tbl", "fig"],
            "page_numbers": [2],
            "contains_figure_caption": True,
        }
    ]
    store, config = _build_custom_store(tmp_path, chunks, parents)
    expander = ParentContextExpander(store, config)
    final_chunks, debug = expander.expand("Describe Figure 2.", [_seed_from_chunk(chunks[0])], _analysis(QueryIntent.FACTOID))
    assert [chunk.chunk_id for chunk in final_chunks] == ["seed", "fig"]
    assert debug["caption_candidates_filtered_by_type"] >= 1


def test_table_query_does_not_add_pure_figure_caption_child(tmp_path: Path):
    chunks = [
        _chunk("seed", "d1", 1, table_caption=True, page_numbers=[2]),
        _chunk("fig", "d1", 2, figure_caption=True, page_numbers=[2]),
        _chunk("tbl", "d1", 3, table_caption=True, page_numbers=[2]),
    ]
    parents = [
        {
            "parent_id": "cap-parent",
            "parent_type": "caption_context",
            "doc_id": "d1",
            "anchor_chunk_id": "seed",
            "child_chunk_ids": ["seed", "fig", "tbl"],
            "page_numbers": [2],
            "contains_table_caption": True,
        }
    ]
    store, config = _build_custom_store(tmp_path, chunks, parents)
    expander = ParentContextExpander(store, config)
    final_chunks, debug = expander.expand("What is listed in Table 1?", [_seed_from_chunk(chunks[0])], _analysis(QueryIntent.FACTOID))
    assert [chunk.chunk_id for chunk in final_chunks] == ["seed", "tbl"]
    assert debug["caption_candidates_filtered_by_type"] >= 1


def test_caption_mode_same_doc_only(tmp_path: Path):
    chunks = [
        _chunk("cap", "d1", 1, figure_caption=True, page_numbers=[2]),
        _chunk("same", "d1", 2, figure_caption=True, page_numbers=[2]),
        _chunk("cross", "d2", 1, page_numbers=[2]),
    ]
    parents = [
        {
            "parent_id": "cap-parent",
            "parent_type": "caption_context",
            "doc_id": "d1",
            "anchor_chunk_id": "cap",
            "child_chunk_ids": ["cap", "cross", "same"],
            "page_numbers": [2],
            "contains_figure_caption": True,
        }
    ]
    store, config = _build_custom_store(tmp_path, chunks, parents)
    expander = ParentContextExpander(store, config)
    final_chunks, debug = expander.expand("what does figure 2 show?", [_seed_from_chunk(chunks[0])], _analysis(QueryIntent.FACTOID))
    assert [chunk.chunk_id for chunk in final_chunks] == ["cap", "same"]
    assert debug["same_doc_only"] is True
    assert debug["skipped_cross_doc"] >= 1


def test_caption_mode_only_expands_target_doc_ids(tmp_path: Path):
    chunks = [
        _chunk("seed1", "d1", 1, figure_caption=True, page_numbers=[2]),
        _chunk("d1peer", "d1", 2, figure_caption=True, page_numbers=[2]),
        _chunk("seed2", "d2", 1, page_numbers=[2]),
        _chunk("d2peer", "d2", 2, figure_caption=True, page_numbers=[2]),
    ]
    store, config = _build_store(tmp_path, chunks)
    expander = ParentContextExpander(store, config)
    seeds = [_seed_from_chunk(chunks[0]), _seed_from_chunk(chunks[2])]
    final_chunks, debug = expander.expand("What is shown in Figure 2?", seeds, _analysis(QueryIntent.FACTOID))
    assert "d1" in debug["caption_target_doc_ids"]
    assert debug["primary_doc_local_context_gating"] is False
    assert "seed2" in debug["skipped_non_target_doc"]
    assert "d2peer" not in [chunk.chunk_id for chunk in final_chunks]


def test_caption_context_limit_prevents_extra_chunk_window(tmp_path: Path):
    chunks = [
        _chunk("c1", "d1", 1, page_numbers=[1]),
        _chunk("cap", "d1", 2, figure_caption=True, page_numbers=[2]),
        _chunk("c3", "d1", 3, figure_caption=True, page_numbers=[2]),
        _chunk("c4", "d1", 4, page_numbers=[3]),
    ]
    store, config = _build_store(tmp_path, chunks)
    expander = ParentContextExpander(store, config)
    final_chunks, debug = expander.expand("what does figure 2 show?", [_seed_from_chunk(chunks[1])], _analysis(QueryIntent.FACTOID))
    assert [chunk.chunk_id for chunk in final_chunks] == ["cap", "c3"]
    assert debug["caption_context_added"] == 1
    assert "chunk_window" not in debug["added_parent_types"]
    assert debug["page_candidates_added"] == 0
    assert debug["page_fallback_used"] is False


def test_page_and_evidence_debug_fields_exist_and_skipped_reason_observable(tmp_path: Path):
    chunks = [_chunk("c1", "d1", 1), _chunk("c2", "d1", 2)]
    store, config = _build_store(tmp_path, chunks)
    expander = ParentContextExpander(store, config)
    _final_chunks, debug = expander.expand("what is this?", [_seed_from_chunk(chunks[0])], _analysis(QueryIntent.FACTOID))
    assert "page_candidates_found" in debug
    assert "page_skipped_reason" in debug
    assert "evidence_candidates_found" in debug
    assert "evidence_skipped_reason" in debug
    assert debug["page_skipped_reason"] == "no_query_trigger"
    assert debug["evidence_skipped_reason"] == "no_query_trigger"


def test_page_parent_can_add_same_page_context(tmp_path: Path):
    chunks = [
        _chunk("cap", "d1", 2, figure_caption=True, page_numbers=[1, 2]),
        _chunk("c3", "d1", 3, figure_caption=True, page_numbers=[2]),
    ]
    store, config = _build_store(tmp_path, chunks)
    config.parent_expansion_caption_enabled = False
    config.parent_expansion_window_enabled = False
    expander = ParentContextExpander(store, config)
    final_chunks, debug = expander.expand("figure details on page", [_seed_from_chunk(chunks[0])], _analysis(QueryIntent.FACTOID))
    assert len(final_chunks) == 2
    assert final_chunks[1].chunk_id == "c3"
    assert debug["page_candidates_added"] == 1


def test_page_fallback_does_not_add_plain_introduction_paragraph(tmp_path: Path):
    chunks = [
        _chunk("seed", "d1", 1, figure_caption=True, page_numbers=[2]),
        _chunk("intro", "d1", 2, section="Introduction", page_numbers=[2]),
    ]
    parents = [
        {
            "parent_id": "cap-parent",
            "parent_type": "caption_context",
            "doc_id": "d1",
            "anchor_chunk_id": "seed",
            "child_chunk_ids": ["seed"],
            "page_numbers": [2],
            "contains_figure_caption": True,
        },
        {
            "parent_id": "page-parent",
            "parent_type": "page",
            "doc_id": "d1",
            "anchor_chunk_id": "",
            "child_chunk_ids": ["seed", "intro"],
            "page_numbers": [2],
            "page_number": 2,
        },
    ]
    store, config = _build_custom_store(tmp_path, chunks, parents)
    config.parent_expansion_window_enabled = False
    expander = ParentContextExpander(store, config)
    final_chunks, debug = expander.expand("What is shown in Figure 2?", [_seed_from_chunk(chunks[0])], _analysis(QueryIntent.FACTOID))
    assert [chunk.chunk_id for chunk in final_chunks] == ["seed"]
    assert debug["page_candidates_added"] == 0
    assert debug["page_plain_paragraph_skipped"] >= 1


def test_evidence_context_can_support_method_query(tmp_path: Path):
    chunks = [
        _chunk("m1", "d1", 1, section="Materials and Methods", evidence_types=["method"]),
        _chunk("m2", "d1", 2, section="Materials and Methods", evidence_types=["method"]),
        _chunk("r1", "d1", 3, section="Results", evidence_types=["result"]),
    ]
    store, config = _build_store(tmp_path, chunks)
    config.parent_expansion_window_enabled = False
    config.parent_expansion_section_path_enabled = False
    expander = ParentContextExpander(store, config)
    final_chunks, debug = expander.expand("what is the method?", [_seed_from_chunk(chunks[0])], _analysis(QueryIntent.FACTOID))
    assert [chunk.chunk_id for chunk in final_chunks] == ["m1", "m2"]
    assert debug["evidence_candidates_added"] == 1
    assert debug["added_parent_types"] == ["evidence_type_context"]


def test_max_total_limit_and_metadata_tags_enforced(tmp_path: Path):
    chunks = [_chunk("c1", "d1", 1), _chunk("c2", "d1", 2), _chunk("c3", "d1", 3), _chunk("c4", "d1", 4)]
    store, config = _build_store(tmp_path, chunks)
    config.parent_expansion_max_total = 2
    expander = ParentContextExpander(store, config)
    final_chunks, _debug = expander.expand("what?", [_seed_from_chunk(chunks[1])], _analysis(QueryIntent.FACTOID))
    assert len(final_chunks) == 2
    assert final_chunks[1].metadata["parent_expansion"] is True
    assert final_chunks[1].metadata["expanded_from_chunk_id"] == "c2"
    assert final_chunks[1].metadata["expanded_from_parent_id"]
    assert final_chunks[1].metadata["expanded_from_parent_type"]
    assert final_chunks[1].metadata["parent_expansion_reason"]


def test_pipeline_default_off_behavior_unchanged():
    pipeline = SynBioRAGPipeline.__new__(SynBioRAGPipeline)
    pipeline.settings = Settings(
        retrieval=RetrievalConfig(final_top_k=2, parent_expansion_enabled=False),
        generation=GenerationConfig(version="v2"),
    )
    pipeline.router = SimpleNamespace(analyze=lambda question: _analysis(QueryIntent.FACTOID))
    retrieved = [_seed_from_chunk(_chunk("c1", "d1", 1))]
    pipeline._search_with_filter_fallback = lambda **kwargs: (retrieved, {"selected": "original", "attempts": []})
    pipeline.reranker = SimpleNamespace(rerank=lambda *args, **kwargs: retrieved, last_debug={})
    pipeline.retriever = SimpleNamespace(last_debug={})
    pipeline.neighbor_expander = SimpleNamespace(last_debug={})
    pipeline.confidence_scorer = SimpleNamespace(score=lambda chunks: 0.5, needs_external_tool=lambda confidence: False)
    pipeline.external_tools = SimpleNamespace()
    pipeline.generator_v2 = SimpleNamespace(run=lambda **kwargs: SimpleNamespace(answer="ok", citations=[], debug={}))

    response = pipeline.answer("what?")
    assert response.debug["parent_expansion"]["reason"] == "disabled"
    assert response.debug["seed_context_count"] == response.debug["final_context_count"]


def test_pipeline_enabled_shows_parent_expansion_debug(tmp_path: Path):
    chunks = [_chunk("c1", "d1", 1), _chunk("c2", "d1", 2), _chunk("c3", "d1", 3)]
    store, config = _build_store(tmp_path, chunks)
    pipeline = SynBioRAGPipeline.__new__(SynBioRAGPipeline)
    pipeline.settings = Settings(
        retrieval=config,
        generation=GenerationConfig(version="v2"),
    )
    pipeline.router = SimpleNamespace(analyze=lambda question: _analysis(QueryIntent.FACTOID))
    retrieved = [_seed_from_chunk(chunks[1])]
    pipeline._search_with_filter_fallback = lambda **kwargs: (retrieved, {"selected": "original", "attempts": []})
    pipeline.reranker = SimpleNamespace(rerank=lambda *args, **kwargs: retrieved, last_debug={})
    pipeline.retriever = SimpleNamespace(last_debug={})
    pipeline.neighbor_expander = SimpleNamespace(last_debug={})
    pipeline.parent_expander = ParentContextExpander(store, config)
    pipeline.confidence_scorer = SimpleNamespace(score=lambda chunks: float(len(chunks)), needs_external_tool=lambda confidence: False)
    pipeline.external_tools = SimpleNamespace()
    pipeline.generator_v2 = SimpleNamespace(run=lambda **kwargs: SimpleNamespace(answer="ok", citations=[], debug={}))

    response = pipeline.answer("what?")
    assert response.debug["parent_expansion"]["enabled"] is True
    assert response.debug["parent_expansion"]["effective_per_seed_limit"] == 1
    assert response.debug["parent_expansion"]["output_count"] >= response.debug["seed_context_count"]
