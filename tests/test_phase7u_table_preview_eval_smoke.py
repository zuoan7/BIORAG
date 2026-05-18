from __future__ import annotations

from pathlib import Path

from scripts.evaluation import phase7u_shadow_smoke as phase7u


ROOT = Path(__file__).resolve().parents[1]


def _build_fixture(tmp_path: Path) -> tuple[Path, Path]:
    fixture_path = tmp_path / "query_fixture.jsonl"
    results_dir = tmp_path / "results"
    summary = phase7u.build_query_fixture(
        output_path=fixture_path,
        summary_path=results_dir / "query_fixture_summary.json",
    )
    assert summary["pass"] is True
    return fixture_path, results_dir


def test_phase7u_query_fixture_contract(tmp_path):
    fixture_path, _results_dir = _build_fixture(tmp_path)
    rows = phase7u.load_jsonl(fixture_path)
    counts = {query_type: 0 for query_type in phase7u.REQUIRED_QUERY_TYPES}
    for row in rows:
        counts[row["query_type"]] = counts.get(row["query_type"], 0) + 1

    assert len(rows) == 13
    assert 10 <= len(rows) <= 20
    assert counts["table_lookup"] >= 1
    assert counts["row_lookup"] >= 1
    assert counts["metric_lookup"] >= 1
    assert counts["non_table_control"] >= 2
    assert all(phase7u.REQUIRED_QUERY_FIELDS <= set(row) for row in rows)


def test_phase7u_shadow_and_merge_runtime_guards(tmp_path):
    fixture_path, results_dir = _build_fixture(tmp_path)

    shadow = phase7u.run_shadow_smoke(fixture_path=fixture_path, output_dir=results_dir)
    merge = phase7u.run_merge_smoke(fixture_path=fixture_path, output_dir=results_dir)
    merge_records = phase7u.load_jsonl(results_dir / "merge_smoke_records.jsonl")

    assert shadow["pass"] is True
    assert shadow["table_query_candidate_count_positive"] == shadow["table_query_count"]
    assert shadow["expected_hit_at_20"] == shadow["table_query_count"]
    assert merge["pass"] is True
    assert merge["merged_table_query_count"] == merge["table_query_count"]
    assert merge["blocked_non_table_count"] == merge["non_table_control_count"]
    assert all(
        row["mode"] == "merge_blocked" and row["preview_output_count"] == 0
        for row in merge_records
        if row["query_type"] == "non_table_control"
    )


def test_phase7u_rollback_and_citation_guard(tmp_path):
    fixture_path, results_dir = _build_fixture(tmp_path)

    rollback = phase7u.run_rollback_smoke(fixture_path=fixture_path, output_dir=results_dir)
    citation = phase7u.run_citation_guard_smoke(fixture_path=fixture_path, output_dir=results_dir)
    citation_records = phase7u.load_jsonl(results_dir / "citation_guard_smoke_records.jsonl")

    assert rollback["pass"] is True
    assert rollback["provider_called"] is False
    assert citation["pass"] is True
    assert citation["formal_citation_count"] == 0
    assert citation_records
    assert all(row["drop_reason"] == "table_preview_formal_citation_blocked" for row in citation_records)
    assert all(row["candidate_eligible"] is False for row in citation_records)


def test_phase7u_validator_accepts_unavailable_local_reranker_skip(tmp_path):
    fixture_path, results_dir = _build_fixture(tmp_path)
    reports_dir = tmp_path / "reports"

    phase7u.run_shadow_smoke(fixture_path=fixture_path, output_dir=results_dir)
    phase7u.run_merge_smoke(fixture_path=fixture_path, output_dir=results_dir)
    phase7u.run_rollback_smoke(fixture_path=fixture_path, output_dir=results_dir)
    phase7u.run_citation_guard_smoke(fixture_path=fixture_path, output_dir=results_dir)
    rerank = phase7u.run_rerank_smoke(
        fixture_path=fixture_path,
        output_dir=results_dir,
        model_path=tmp_path / "missing-reranker",
    )

    validation = phase7u.validate_preview_eval_smoke(
        fixture_path=fixture_path,
        results_dir=results_dir,
        reports_dir=reports_dir,
    )

    assert rerank["status"] == "skipped"
    assert rerank["local_reranker_available"] is False
    assert validation["pass"] is True
    assert (reports_dir / "preview_eval_smoke_report.md").exists()


def test_phase7u_scripts_do_not_import_forbidden_retrieval_or_embedding_paths():
    script_paths = [
        ROOT / "scripts/evaluation/phase7u_build_query_fixture.py",
        ROOT / "scripts/evaluation/phase7u_shadow_smoke.py",
        ROOT / "scripts/evaluation/phase7u_merge_smoke.py",
        ROOT / "scripts/evaluation/phase7u_rerank_smoke.py",
        ROOT / "scripts/evaluation/phase7u_rollback_smoke.py",
        ROOT / "scripts/evaluation/phase7u_citation_guard_smoke.py",
        ROOT / "scripts/evaluation/phase7u_validate_preview_eval_smoke.py",
    ]
    forbidden_patterns = [
        "pymilvus",
        "MilvusClient",
        "BGEM3Embedder",
        "BM25Retriever",
        "HybridRetriever",
        "rank_bm25",
        "SentenceTransformer",
    ]

    for path in script_paths:
        source = path.read_text(encoding="utf-8")
        for pattern in forbidden_patterns:
            assert pattern not in source
