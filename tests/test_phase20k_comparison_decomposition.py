"""Phase 20K: Comparison decomposition fix tests."""
import pytest, re
from src.synbio_rag.infrastructure.vectorstores.hybrid import (
    _extract_comparison_subqueries, _build_query_plan,
    _mask_organism_abbrevs, _unmask_organism_abbrevs,
)
from src.synbio_rag.domain.schemas import QueryAnalysis, QueryIntent

class FC: comparison_query_weight = 1.0; comparison_subquery_weight = 0.7

def test_ent010_cn_two_branches():
    q = "比较 6′-SL 与 2′-FL 两类 HMO 在当前文库里的工程化合成路径思路。请分别说明它们依赖的关键前体和末端催化步骤。"
    subs = _extract_comparison_subqueries(q)
    assert len(subs) >= 2
    assert any("6′-SL" in s or "6-sialyl" in s.lower() for s in subs)
    assert any("2′-FL" in s or "2-fucosyl" in s.lower() for s in subs)

def test_ent083_ecoli_subtilis_branches():
    q = "比较文库中 E. coli 和 B. subtilis 作为 NeuAc 生产宿主的策略差异和产量表现。"
    subs = _extract_comparison_subqueries(q)
    assert len(subs) >= 2, f"Expected >=2 subs, got {len(subs)}: {subs}"
    assert any("E. coli" in s for s in subs)
    assert any("B. subtilis" in s for s in subs)

def test_organism_period_not_sentence_boundary():
    for organism in ["E. coli", "B. subtilis", "S. cerevisiae", "P. pastoris", "C. glutamicum", "L. lactis"]:
        masked = _mask_organism_abbrevs(f"test {organism} more")
        assert "." not in masked, f"{organism} period not masked"
        unmasked = _unmask_organism_abbrevs(masked)
        assert organism in unmasked, f"{organism} not recovered"

def test_mask_roundtrip():
    text = "文库中 E. coli 和 B. subtilis 以及 S. cerevisiae 的比较"
    m = _mask_organism_abbrevs(text)
    assert "E." not in m and "B." not in m and "S." not in m
    u = _unmask_organism_abbrevs(m)
    assert "E. coli" in u and "B. subtilis" in u and "S. cerevisiae" in u

def test_decomposition_query_used_for_comparison():
    analysis = QueryAnalysis(intent=QueryIntent.COMPARISON, requires_external_tools=False, search_limit=40, rerank_top_k=10)
    plan = _build_query_plan(
        "EN retrieval query about E coli and B subtilis", analysis, FC(),
        decomposition_query="比较 E. coli 和 B. subtilis 作为 NeuAc 宿主"
    )
    assert len(plan) >= 3, f"Expected >=3 variants (1 main + 2 subs), got {len(plan)}"
    kinds = [v["kind"] for v in plan]
    assert "subquery" in kinds

def test_no_sample_special_case():
    import inspect
    from src.synbio_rag.infrastructure.vectorstores import hybrid
    src = inspect.getsource(hybrid)
    for banned in ["ent_010", "ent_083", "doc_0009", "doc_0073", "doc_0119", "doc_0147"]:
        assert banned not in src, f"Banned string {banned} in source"

def test_non_comparison_no_subqueries_in_plan():
    analysis = QueryAnalysis(intent=QueryIntent.FACTOID, requires_external_tools=False, search_limit=40, rerank_top_k=10)
    plan = _build_query_plan("some factoid query with 与 and 和", analysis, FC())
    assert len(plan) == 1  # Only the original query, no subqueries
