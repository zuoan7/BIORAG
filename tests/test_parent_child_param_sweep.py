from __future__ import annotations

import json
from argparse import Namespace

from scripts.evaluation.run_parent_child_param_sweep import (
    AB500_PROFILES,
    GENERATION_PROFILES,
    PARENT_PROFILES,
    RECALL_PROFILES,
    Phase5FSample,
    SweepProfile,
    add_gate_checks,
    aggregate_records,
    build_settings,
    chunk_matches_stable_parent,
    load_phase5f_samples,
    select_profiles,
)
from src.synbio_rag.domain.schemas import RetrievedChunk


def test_recall_matrix_matches_parent_child_plan() -> None:
    assert [profile.name for profile in RECALL_PROFILES] == [
        "baseline",
        "recall_80_rerank10",
        "recall_80_rerank20",
        "recall_80_rerank20_final12",
        "recall_100_rerank20_final12",
    ]
    assert RECALL_PROFILES[0].retrieval == {
        "dense_limit": 40,
        "bm25_limit": 40,
        "search_limit": 40,
        "rerank_top_k": 10,
        "final_top_k": 8,
    }
    assert RECALL_PROFILES[-1].retrieval == {
        "dense_limit": 100,
        "bm25_limit": 100,
        "search_limit": 100,
        "rerank_top_k": 20,
        "final_top_k": 12,
    }


def test_ab500_matrix_has_focused_default_and_candidate_profiles() -> None:
    assert [profile.name for profile in AB500_PROFILES] == [
        "baseline",
        "recall80_f8_parent_default",
        "recall80_f12_parent_disabled",
        "recall80_f12_parent_default",
        "recall80_f12_parent_preserve_seed_extra2",
        "recall80_f12_parent_preserve_seed_extra4",
    ]
    assert AB500_PROFILES[0].retrieval == {
        "dense_limit": 40,
        "bm25_limit": 40,
        "search_limit": 40,
        "rerank_top_k": 10,
        "final_top_k": 8,
    }
    assert AB500_PROFILES[2].retrieval["final_top_k"] == 12
    assert AB500_PROFILES[2].parent["parent_expansion_enabled"] is False
    assert AB500_PROFILES[4].parent["parent_expansion_preserve_seed_chunks"] is True
    assert AB500_PROFILES[4].parent["parent_expansion_max_added"] == 2
    assert AB500_PROFILES[5].parent["parent_expansion_max_added"] == 4


def test_parent_and_generation_matrices_match_plan() -> None:
    assert [profile.parent for profile in PARENT_PROFILES] == [
        {
            "parent_expansion_enabled": False,
            "parent_expansion_max_total": 12,
            "parent_expansion_per_seed_limit": 2,
        },
        {
            "parent_expansion_enabled": True,
            "parent_expansion_max_total": 12,
            "parent_expansion_per_seed_limit": 2,
        },
        {
            "parent_expansion_enabled": True,
            "parent_expansion_max_total": 8,
            "parent_expansion_per_seed_limit": 1,
        },
        {
            "parent_expansion_enabled": True,
            "parent_expansion_max_total": 16,
            "parent_expansion_per_seed_limit": 2,
        },
        {
            "parent_expansion_enabled": True,
            "parent_expansion_max_total": 18,
            "parent_expansion_per_seed_limit": 3,
        },
    ]
    assert [profile.generation for profile in GENERATION_PROFILES] == [
        {
            "v2_max_support_factoid": 3,
            "v2_max_support_summary": 5,
            "v2_max_support_comparison": 6,
            "v2_max_extractive_evidence_lines": 6,
        },
        {
            "v2_max_support_factoid": 2,
            "v2_max_support_summary": 5,
            "v2_max_support_comparison": 6,
            "v2_max_extractive_evidence_lines": 4,
        },
        {
            "v2_max_support_factoid": 3,
            "v2_max_support_summary": 6,
            "v2_max_support_comparison": 8,
            "v2_max_extractive_evidence_lines": 6,
        },
    ]


def test_parent_round_composes_recall_base_with_parent_profile() -> None:
    profiles = select_profiles(
        "parent",
        ["parent_wider"],
        base_profile_name="recall_80_rerank20_final12",
        parent_profile_name="parent_default",
    )

    assert profiles == [
        SweepProfile(
            name="parent_wider",
            retrieval={
                "dense_limit": 80,
                "bm25_limit": 80,
                "search_limit": 80,
                "rerank_top_k": 20,
                "final_top_k": 12,
            },
            parent={
                "parent_expansion_enabled": True,
                "parent_expansion_max_total": 16,
                "parent_expansion_per_seed_limit": 2,
            },
            generation={},
        )
    ]


def test_generation_round_composes_recall_parent_and_generation_profiles() -> None:
    profiles = select_profiles(
        "generation",
        ["gen_summary_wide"],
        base_profile_name="recall_80_rerank20_final12",
        parent_profile_name="parent_wider",
    )

    assert profiles[0].retrieval["final_top_k"] == 12
    assert profiles[0].parent["parent_expansion_max_total"] == 16
    assert profiles[0].generation["v2_max_support_summary"] == 6
    assert profiles[0].generation["v2_max_support_comparison"] == 8


def test_phase5f_loader_uses_query_field_and_stable_blocks(tmp_path) -> None:
    dataset = tmp_path / "strict.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "sample_id": "s1",
                "query": "rewritten query",
                "original_query": "original query",
                "query_type": "table_content",
                "target_doc_id": "doc_1",
                "stable_target_block_ids": ["b1"],
                "target_associated_block_id": "b2",
                "target_caption_block_id": "b1",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    samples = load_phase5f_samples(dataset)

    assert samples[0].query == "rewritten query"
    assert samples[0].stable_target_block_ids == ("b1", "b2")


def test_loader_accepts_smoke200_question_and_expected_doc_ids(tmp_path) -> None:
    dataset = tmp_path / "smoke200.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "sample_id": "s1",
                "question": "compare these papers",
                "category": "comparison",
                "expected_doc_ids": ["doc_1", "doc_2"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    samples = load_phase5f_samples(dataset)

    assert samples[0].query == "compare these papers"
    assert samples[0].query_type == "comparison"
    assert samples[0].target_doc_id == "doc_1"
    assert samples[0].target_doc_ids == ("doc_1", "doc_2")


def test_chunk_matches_stable_parent_from_parent_metadata() -> None:
    sample = Phase5FSample(
        sample_id="s1",
        query="q",
        query_type="figure_caption",
        target_doc_id="doc_1",
        stable_target_block_ids=("b_target",),
    )
    chunk = RetrievedChunk(
        chunk_id="parent_1",
        doc_id="doc_1",
        source_file="doc_1.pdf",
        title="",
        section="Results",
        text="",
        metadata={"source_block_ids": ["b_other", "b_target"]},
    )

    assert chunk_matches_stable_parent(chunk, sample)


def test_doc_hit_accepts_any_expected_doc_id_for_smoke200_sample() -> None:
    sample = Phase5FSample(
        sample_id="s1",
        query="q",
        query_type="comparison",
        target_doc_id="doc_1",
        target_doc_ids=("doc_1", "doc_2"),
        stable_target_block_ids=(),
    )
    chunk = RetrievedChunk(
        chunk_id="parent_1",
        doc_id="doc_2",
        source_file="doc_2.pdf",
        title="",
        section="Results",
        text="",
        metadata={},
    )

    assert chunk_matches_stable_parent(chunk, sample)


def test_chunk_matches_stable_parent_from_target_chunk_candidate_child_id() -> None:
    sample = Phase5FSample(
        sample_id="s1",
        query="q",
        query_type="table_content",
        target_doc_id="doc_1",
        stable_target_block_ids=("missing_block",),
        target_chunk_id_candidate="parent_1::child002",
    )
    chunk = RetrievedChunk(
        chunk_id="parent_1",
        doc_id="doc_1",
        source_file="doc_1.pdf",
        title="",
        section="Results",
        text="",
        metadata={"matched_child_chunk_ids": ["parent_1::child002"]},
    )

    assert chunk_matches_stable_parent(chunk, sample)


def test_aggregate_records_reports_required_metrics() -> None:
    record = {
        "query_type": "figure_caption",
        "retrieved_top5_doc": True,
        "retrieved_top5_stable_parent": True,
        "retrieved_top10_doc": True,
        "retrieved_top10_stable_parent": True,
        "rerank_top5_doc": True,
        "rerank_top5_stable_parent": False,
        "final_topK_doc": True,
        "final_topK_stable_parent": True,
        "materialized_top5_has_matched_child": True,
        "support_child_view_used": True,
        "cited_child_view_used": True,
        "citation_quote_has_child_marker": True,
        "all_citations_parent_ids": True,
        "legal_final_citation": True,
        "citation_target_doc_hit": True,
        "citation_count": 2,
        "answer_mode": "full",
        "latency_ms": 123.4,
        "rerank_candidate_count": 80,
        "support_pack_size": 3,
        "parent_expansion": {
            "enabled_config": True,
            "enabled_debug": True,
            "reason": "expanded",
            "input_count": 8,
            "output_count": 10,
            "added_count": 2,
            "added_chunk_ids": ["c2", "c3"],
            "added_parent_ids": ["p1", "p2"],
            "added_parent_types": ["caption_context", "page"],
            "effective_max_total": 12,
            "effective_per_seed_limit": 2,
            "over_budget": False,
            "disabled_but_added": False,
            "enabled_but_disabled_reason": False,
            "debug_consistent": True,
        },
        "error": "",
        "sample": {"sample_id": "s1"},
    }

    metrics = aggregate_records([record])

    assert metrics["overall"]["retrieved_top5_doc"] == {"hit": 1, "total": 1, "rate": 1.0}
    assert metrics["overall"]["rerank_top5_stable_parent"] == {
        "hit": 0,
        "total": 1,
        "rate": 0.0,
    }
    assert metrics["overall"]["cost"]["average_rerank_candidate_count"] == 80.0
    assert metrics["overall"]["cost"]["average_support_pack_size"] == 3.0
    assert metrics["overall"]["parent_expansion_audit"]["expanded_sample_count"] == 1
    assert metrics["overall"]["parent_expansion_audit"]["added_count"]["average"] == 2.0
    assert metrics["overall"]["parent_expansion_audit"]["added_parent_type_distribution"] == {
        "caption_context": 1,
        "page": 1,
    }
    assert metrics["by_query_type"]["figure_caption"]["final_topK_stable_parent"]["hit"] == 1


def test_gate_checks_compare_candidate_to_baseline() -> None:
    def result(name: str, final_stable: int, table_stable: int, p95: float) -> dict:
        base_record = {
            "sample_count": 2,
            "final_topK_doc": {"hit": final_stable, "total": 2, "rate": final_stable / 2},
            "final_topK_stable_parent": {
                "hit": final_stable,
                "total": 2,
                "rate": final_stable / 2,
            },
            "support_child_view_used": {"hit": 2, "total": 2, "rate": 1.0},
            "cited_child_view_used": {"hit": 2, "total": 2, "rate": 1.0},
            "all_citations_parent_ids": {"hit": 2, "total": 2, "rate": 1.0},
            "cost": {"latency_ms_p95": p95},
            "parent_expansion_audit": {
                "inconsistent_debug_count": 0,
                "over_budget_count": 0,
                "disabled_but_added_count": 0,
            },
        }
        return {
            "profile": {"name": name},
            "run_config": {"parent_expansion_enabled": True},
            "metrics": {
                "overall": base_record,
                "by_query_type": {
                    "figure_caption": {
                        "final_topK_stable_parent": {"hit": 1, "total": 1, "rate": 1.0}
                    },
                    "table_content": {
                        "final_topK_stable_parent": {
                            "hit": table_stable,
                            "total": 1,
                            "rate": float(table_stable),
                        }
                    },
                    "caption_level_table": {
                        "final_topK_stable_parent": {"hit": 0, "total": 1, "rate": 0.0}
                    },
                },
                "gate_checks": {},
            },
        }

    results = [result("baseline", 1, 0, 100.0), result("candidate", 2, 1, 150.0)]

    add_gate_checks(results)

    assert results[1]["metrics"]["gate_checks"]["candidate_gate_passed"] is True
    assert results[1]["metrics"]["gate_checks"]["p95_latency_ratio_vs_baseline"] == 1.5


def test_build_settings_pins_parent_child_index_and_extract_mode(monkeypatch, tmp_path) -> None:
    from src.synbio_rag.domain import config as config_mod

    monkeypatch.setattr(config_mod, "dotenv_values", lambda _path: {})
    monkeypatch.setenv("GENERATION_V2_USE_QWEN_SYNTHESIS", "true")
    args = Namespace(
        milvus_uri=str(tmp_path / "pc.db"),
        collection_name="pc_collection",
        bm25_cache=str(tmp_path / "bm25.json"),
        parent_index=str(tmp_path / "parent_index.jsonl"),
        child_chunks_jsonl=str(tmp_path / "child_chunks.jsonl"),
        parent_chunks_jsonl=str(tmp_path / "parent_chunks.jsonl"),
    )
    profile = SweepProfile(
        name="candidate",
        retrieval={"dense_limit": 80, "bm25_limit": 80, "search_limit": 80},
        parent={
            "parent_expansion_enabled": False,
            "parent_expansion_max_total": 16,
            "parent_expansion_preserve_seed_chunks": True,
            "parent_expansion_max_added": 4,
        },
        generation={"v2_max_support_summary": 6},
    )

    settings = build_settings(args, profile)

    assert settings.retrieval.milvus_uri == str(tmp_path / "pc.db")
    assert settings.retrieval.collection_name == "pc_collection"
    assert settings.query_rewrite.mode == "off"
    assert settings.generation.v2_use_qwen_synthesis is False
    assert settings.retrieval.dense_limit == 80
    assert settings.retrieval.parent_expansion_enabled is False
    assert settings.retrieval.parent_expansion_max_total == 16
    assert settings.retrieval.parent_expansion_preserve_seed_chunks is True
    assert settings.retrieval.parent_expansion_max_added == 4
    assert settings.generation.v2_max_support_summary == 6
