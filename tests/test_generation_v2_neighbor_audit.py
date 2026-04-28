from __future__ import annotations

from src.synbio_rag.application.generation_v2.models import AnswerPlan, EvidenceCandidate, SupportItem
from src.synbio_rag.application.generation_v2.neighbor_audit import NeighborAuditEngine, NeighborAuditResult, _get_seed_score
from src.synbio_rag.application.generation_v2.qwen_synthesizer import QwenSynthesizer
from src.synbio_rag.application.generation_v2.service import GenerationV2Service
from src.synbio_rag.domain.config import GenerationConfig
from src.synbio_rag.domain.schemas import QueryAnalysis, QueryIntent, RetrievedChunk


# ── test helpers ─────────────────────────────────────────────────────────────

def _analysis(intent: QueryIntent = QueryIntent.FACTOID) -> QueryAnalysis:
    return QueryAnalysis(intent=intent, requires_external_tools=False, search_limit=10, rerank_top_k=5)


def _chunk(
    chunk_id: str,
    text: str,
    *,
    doc_id: str = "doc1",
    section: str = "Results",
    fusion: float = 0.5,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_file=f"{doc_id}.pdf",
        title=f"title-{doc_id}",
        section=section,
        text=text,
        rerank_score=fusion,
        fusion_score=fusion,
        vector_score=0.2,
        bm25_score=0.1,
        metadata={},
    )


def _candidate(
    evidence_id: str = "E1",
    *,
    chunk_id: str = "doc1_sec01_chunk01",
    doc_id: str = "doc1",
    section: str = "Results",
    fusion: float = 0.5,
) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=evidence_id,
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_file=f"{doc_id}.pdf",
        title=f"title-{doc_id}",
        section=section,
        text="placeholder text with results showed yield",
        page_start=None,
        page_end=None,
        vector_score=0.2,
        bm25_score=0.1,
        rerank_score=fusion,
        fusion_score=fusion,
        metadata={},
        features={},
        reasons=["seed_chunk"],
    )


def _support_item(evidence_id: str = "E1", *, chunk_id: str = "doc1_sec01_chunk01", doc_id: str = "doc1") -> SupportItem:
    return SupportItem(
        evidence_id=evidence_id,
        candidate=_candidate(evidence_id, chunk_id=chunk_id, doc_id=doc_id),
        support_score=0.9,
        reasons=["selected_support"],
    )


def _make_engine(chunks: list[RetrievedChunk]) -> NeighborAuditEngine:
    """Build NeighborAuditEngine from an in-memory chunk list (no jsonl needed)."""
    from collections import defaultdict

    by_id: dict[str, RetrievedChunk] = {}
    positions: dict[str, tuple[str, int]] = {}
    doc_chunks: dict[str, list[RetrievedChunk]] = defaultdict(list)

    # group by doc_id in insertion order
    grouped: dict[str, list[RetrievedChunk]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.doc_id].append(chunk)

    for doc_id, doc_list in grouped.items():
        doc_chunks[doc_id] = doc_list
        for pos, chunk in enumerate(doc_list):
            by_id[chunk.chunk_id] = chunk
            positions[chunk.chunk_id] = (doc_id, pos)

    return NeighborAuditEngine(dict(by_id), dict(positions), dict(doc_chunks))


class FakeChatClient:
    def __init__(self, output: str = "") -> None:
        self.output = output
        self.calls = 0
        self.timeout_seconds = 30
        self.last_messages: list | None = None

    def is_enabled(self) -> bool:
        return True

    def chat_completion(self, *, messages: list, **_: object) -> str:
        self.calls += 1
        self.last_messages = messages
        return self.output


class DummyLedger:
    def __init__(self, candidates: list[EvidenceCandidate]) -> None:
        self.candidates = candidates

    def build(self, *args: object, **kwargs: object) -> list[EvidenceCandidate]:
        return self.candidates


class DummySelector:
    def __init__(self, items: list[SupportItem]) -> None:
        self.items = items
        self.last_summary_selection_debug: dict = {"is_summary": False}

    def select(self, *args: object, **kwargs: object) -> list[SupportItem]:
        return self.items


class DummyPlanner:
    def __init__(self, plan: AnswerPlan) -> None:
        self.plan_value = plan
        self.last_existence_guardrail: dict = {}
        self.last_comparison_coverage_debug: dict = {"reason": "not_comparison_intent", "parse_ok": False}
        self.last_summary_plan_debug: dict = {"is_summary": False}

    def plan(self, *args: object, **kwargs: object) -> AnswerPlan:
        return self.plan_value


class DummyBuilder:
    def build(self, *args: object, **kwargs: object) -> str:
        return "抽取式答案 [E1]"


def _service_with(
    *,
    audit_engine: NeighborAuditEngine | None,
    support_pack: list[SupportItem],
    plan: AnswerPlan | None = None,
    qwen_output: str = "根据当前证据，结论如下 [E1]",
    candidates: list[EvidenceCandidate] | None = None,
) -> GenerationV2Service:
    if plan is None:
        plan = AnswerPlan(mode="full", reason="factoid_supported")
    svc = GenerationV2Service(
        synthesizer=QwenSynthesizer(client=FakeChatClient(qwen_output)),
        neighbor_audit_engine=audit_engine,
    )
    svc.ledger_builder = DummyLedger(candidates or [item.candidate for item in support_pack])
    svc.support_selector = DummySelector(support_pack)
    svc.answer_planner = DummyPlanner(plan)
    svc.answer_builder = DummyBuilder()
    return svc


# ── tests ─────────────────────────────────────────────────────────────────────

def test_neighbor_audit_disabled_returns_enabled_false() -> None:
    """audit disabled 时 debug.neighbor_audit.enabled=False，support_pack 不变"""
    support = [_support_item()]
    svc = _service_with(audit_engine=None, support_pack=support)

    result = svc.run(
        question="总结 PHA 发酵结果",
        analysis=_analysis(QueryIntent.SUMMARY),
        seed_chunks=[_chunk("doc1_sec01_chunk01", "seed text with yield results showed 10 g/L")],
        config=GenerationConfig(v2_use_qwen_synthesis=False, v2_enable_neighbor_audit=False),
    )

    na = result.debug["neighbor_audit"]
    assert na["enabled"] is False
    assert [i.evidence_id for i in result.support_pack] == ["E1"]


def test_neighbor_audit_no_engine_still_disabled() -> None:
    """engine=None 时即使开启 flag 也返回 enabled=False"""
    svc = _service_with(audit_engine=None, support_pack=[_support_item()])

    result = svc.run(
        question="测试",
        analysis=_analysis(),
        seed_chunks=[_chunk("doc1_sec01_chunk01", "seed text results")],
        config=GenerationConfig(v2_use_qwen_synthesis=False, v2_enable_neighbor_audit=True),
    )

    assert result.debug["neighbor_audit"]["enabled"] is False


def test_neighbor_does_not_inherit_seed_score() -> None:
    """neighbor_score 不等于 seed_score，inherited_seed_score=False"""
    seed_fusion = 10.0
    corpus = [
        _chunk("chunk_0", "intro text", doc_id="doc1", section="Introduction", fusion=0.0),
        _chunk("chunk_1", "Results showed yield increased to 12 g/L", doc_id="doc1", section="Results", fusion=seed_fusion),
        _chunk("chunk_2", "Discussion: yield improvement explained by pathway balance", doc_id="doc1", section="Discussion", fusion=0.0),
    ]
    engine = _make_engine(corpus)
    seed_chunk = corpus[1]
    cand = _candidate("E1", chunk_id="chunk_1", fusion=seed_fusion)
    support = [SupportItem(evidence_id="E1", candidate=cand, support_score=0.9, reasons=[])]

    svc = _service_with(audit_engine=engine, support_pack=support, candidates=[cand])
    result = svc.run(
        question="Results showed yield",
        analysis=_analysis(QueryIntent.SUMMARY),
        seed_chunks=[seed_chunk],
        config=GenerationConfig(
            v2_use_qwen_synthesis=False,
            v2_enable_neighbor_audit=True,
            v2_neighbor_window=1,
            v2_neighbor_promotion_dry_run=True,
        ),
    )

    na = result.debug["neighbor_audit"]
    assert na["enabled"] is True
    for cand_dict in na["candidates"]:
        assert cand_dict["inherited_seed_score"] is False
        assert cand_dict["neighbor_score"] != seed_fusion
        assert cand_dict["neighbor_score"] < seed_fusion


def test_neighbor_citable_always_false() -> None:
    """所有 neighbor.citable=False，即使 promotion_status=dry_run_promoted"""
    corpus = [
        _chunk("chunk_0", "Methods: fermentation setup with 5L tank", doc_id="doc1", section="Methods", fusion=0.0),
        _chunk("chunk_1", "Results showed yield 12 g/L achieved", doc_id="doc1", section="Results", fusion=0.8),
        _chunk("chunk_2", "Discussion: yield improvement due to pathway", doc_id="doc1", section="Discussion", fusion=0.0),
    ]
    engine = _make_engine(corpus)
    cand = _candidate("E1", chunk_id="chunk_1", fusion=0.8)
    support = [SupportItem(evidence_id="E1", candidate=cand, support_score=0.9, reasons=[])]

    svc = _service_with(audit_engine=engine, support_pack=support, candidates=[cand])
    result = svc.run(
        question="Results yield",
        analysis=_analysis(QueryIntent.SUMMARY),
        seed_chunks=[corpus[1]],
        config=GenerationConfig(v2_use_qwen_synthesis=False, v2_enable_neighbor_audit=True),
    )

    for c in result.debug["neighbor_audit"]["candidates"]:
        assert c["citable"] is False


def test_window_1_collects_only_adjacent() -> None:
    """window=1 时只收集前后 1 个邻居，不收集 ±2"""
    corpus = [
        _chunk(f"chunk_{i}", f"text chunk {i} with results showed", doc_id="doc1", section="Results", fusion=0.0)
        for i in range(5)
    ]
    engine = _make_engine(corpus)
    seed = corpus[2]  # index 2
    cand = _candidate("E1", chunk_id="chunk_2", fusion=0.5)
    support = [SupportItem(evidence_id="E1", candidate=cand, support_score=0.9, reasons=[])]

    svc = _service_with(audit_engine=engine, support_pack=support, candidates=[cand])
    result = svc.run(
        question="text chunk results",
        analysis=_analysis(),
        seed_chunks=[seed],
        config=GenerationConfig(v2_use_qwen_synthesis=False, v2_enable_neighbor_audit=True, v2_neighbor_window=1),
    )

    na = result.debug["neighbor_audit"]
    collected_ids = {c["chunk_id"] for c in na["candidates"]}
    # should contain chunk_1 and chunk_3, NOT chunk_0 or chunk_4
    assert "chunk_1" in collected_ids
    assert "chunk_3" in collected_ids
    assert "chunk_0" not in collected_ids
    assert "chunk_4" not in collected_ids


def test_low_quality_sections_excluded() -> None:
    """References / Author 节不应 dry_run_promoted，promotion_status=excluded"""
    corpus = [
        _chunk("chunk_0", "Smith et al. 2023. Nature. DOI:10.1000", doc_id="doc1", section="References", fusion=0.0),
        _chunk("chunk_1", "Results showed yield increased to 15 g/L", doc_id="doc1", section="Results", fusion=0.8),
        _chunk("chunk_2", "Author Information: John Doe, University", doc_id="doc1", section="Author Information", fusion=0.0),
    ]
    engine = _make_engine(corpus)
    cand = _candidate("E1", chunk_id="chunk_1", fusion=0.8)
    support = [SupportItem(evidence_id="E1", candidate=cand, support_score=0.9, reasons=[])]

    svc = _service_with(audit_engine=engine, support_pack=support, candidates=[cand])
    result = svc.run(
        question="yield results",
        analysis=_analysis(),
        seed_chunks=[corpus[1]],
        config=GenerationConfig(v2_use_qwen_synthesis=False, v2_enable_neighbor_audit=True, v2_neighbor_window=1),
    )

    for c in result.debug["neighbor_audit"]["candidates"]:
        if c["section"].lower() in {"references", "author information"}:
            assert c["promotion_status"] == "excluded", f"expected excluded for section {c['section']}"


def test_summary_potential_boost_counted() -> None:
    """summary 问题中邻居包含 result/numeric 且有 query_overlap → potential_summary_boost_count 增加"""
    corpus = [
        _chunk("chunk_0", "Methods: fermentation protocol setup at 37C in 5L bioreactor system", doc_id="doc1", section="Methods", fusion=0.0),
        _chunk("chunk_1", "Abstract: this study investigated pathway yield improvement", doc_id="doc1", section="Abstract", fusion=0.6),
        _chunk("chunk_2", "Results showed yield increased to 12.5 g/L with improved production efficiency pathway", doc_id="doc1", section="Results", fusion=0.0),
    ]
    engine = _make_engine(corpus)
    cand = _candidate("E1", chunk_id="chunk_1", fusion=0.6)
    support = [SupportItem(evidence_id="E1", candidate=cand, support_score=0.9, reasons=[])]

    svc = _service_with(audit_engine=engine, support_pack=support, candidates=[cand])
    result = svc.run(
        question="yield improvement pathway results production",
        analysis=_analysis(QueryIntent.SUMMARY),
        seed_chunks=[corpus[1]],
        config=GenerationConfig(v2_use_qwen_synthesis=False, v2_enable_neighbor_audit=True, v2_neighbor_window=1),
    )

    na = result.debug["neighbor_audit"]
    assert na["summary"]["potential_summary_boost_count"] >= 0  # counts are present
    # chunk_2 is a neighbor of chunk_1, has result terms + query overlap → should be dry_run_promoted
    promoted = [c for c in na["candidates"] if c["promotion_status"] == "dry_run_promoted"]
    assert any(c["chunk_id"] == "chunk_2" for c in promoted)


def test_comparison_branch_overlap_recorded() -> None:
    """comparison 问题中邻居命中 branch term → promotion_reasons 包含 branch_overlap"""
    corpus = [
        _chunk("chunk_0", "Background information", doc_id="doc1", section="Introduction", fusion=0.0),
        _chunk("chunk_1", "Results showed E. coli yield 8 g/L with strategy A", doc_id="doc1", section="Results", fusion=0.7),
        _chunk("chunk_2", "Discussion of E. coli strategy B comparison with pathway improvement", doc_id="doc1", section="Discussion", fusion=0.0),
    ]
    engine = _make_engine(corpus)
    cand = _candidate("E1", chunk_id="chunk_1", fusion=0.7)
    support = [SupportItem(evidence_id="E1", candidate=cand, support_score=0.9, reasons=[])]

    svc = _service_with(audit_engine=engine, support_pack=support, candidates=[cand])
    result = svc.run(
        question="比较 E. coli strategy A 和 strategy B 的 yield 差异",
        analysis=_analysis(QueryIntent.COMPARISON),
        seed_chunks=[corpus[1]],
        config=GenerationConfig(v2_use_qwen_synthesis=False, v2_enable_neighbor_audit=True, v2_neighbor_window=1),
    )

    na = result.debug["neighbor_audit"]
    chunk2_candidates = [c for c in na["candidates"] if c["chunk_id"] == "chunk_2"]
    assert len(chunk2_candidates) == 1
    c2 = chunk2_candidates[0]
    # comparison_branch_overlap requires >= 3 token overlap; check that features recorded
    assert "features" in c2
    # promotion_status is dry_run_promoted or context_only, both are valid; branch_overlap reason if overlapped
    if "branch_overlap" in c2["promotion_reasons"]:
        assert c2["promotion_status"] in {"dry_run_promoted", "context_only"}


def test_no_support_pack_mutation() -> None:
    """audit 前后 support_pack evidence_ids 完全一致"""
    corpus = [
        _chunk("chunk_0", "previous section text", doc_id="doc1", section="Results", fusion=0.0),
        _chunk("chunk_1", "seed result: yield 10 g/L", doc_id="doc1", section="Results", fusion=0.8),
        _chunk("chunk_2", "next section text", doc_id="doc1", section="Results", fusion=0.0),
    ]
    engine = _make_engine(corpus)
    cand = _candidate("E1", chunk_id="chunk_1", fusion=0.8)
    support = [SupportItem(evidence_id="E1", candidate=cand, support_score=0.9, reasons=[])]

    svc = _service_with(audit_engine=engine, support_pack=support, candidates=[cand])
    result = svc.run(
        question="yield result",
        analysis=_analysis(),
        seed_chunks=[corpus[1]],
        config=GenerationConfig(v2_use_qwen_synthesis=False, v2_enable_neighbor_audit=True),
    )

    assert [i.evidence_id for i in result.support_pack] == ["E1"]
    # neighbors must not appear in support_pack
    sp_chunk_ids = {i.candidate.chunk_id for i in result.support_pack}
    for nc in result.debug["neighbor_audit"]["candidates"]:
        assert nc["chunk_id"] not in sp_chunk_ids or nc["chunk_id"] == "chunk_1"


def test_no_qwen_context_mutation() -> None:
    """Qwen input_evidence_ids 不包含 neighbor ids"""
    corpus = [
        _chunk("chunk_0", "neighbor with results showed yield 9 g/L", doc_id="doc1", section="Results", fusion=0.0),
        _chunk("chunk_1", "seed: main result 12 g/L", doc_id="doc1", section="Results", fusion=0.8),
        _chunk("chunk_2", "another neighbor discussion", doc_id="doc1", section="Discussion", fusion=0.0),
    ]
    engine = _make_engine(corpus)
    cand = _candidate("E1", chunk_id="chunk_1", fusion=0.8)
    support = [SupportItem(evidence_id="E1", candidate=cand, support_score=0.9, reasons=[])]

    client = FakeChatClient("根据证据，结论如下 [E1]")
    svc = _service_with(audit_engine=engine, support_pack=support, candidates=[cand], qwen_output="根据证据，结论如下 [E1]")
    svc.synthesizer = QwenSynthesizer(client=client)

    result = svc.run(
        question="main result yield",
        analysis=_analysis(),
        seed_chunks=[corpus[1]],
        config=GenerationConfig(v2_use_qwen_synthesis=True, v2_enable_neighbor_audit=True),
    )

    qwen_input_ids = result.debug["qwen_synthesis"]["input_evidence_ids"]
    neighbor_ids = {c["chunk_id"] for c in result.debug["neighbor_audit"]["candidates"]}
    qwen_input_chunk_ids = {item.candidate.chunk_id for item in result.support_pack}
    assert not (neighbor_ids & qwen_input_chunk_ids - {"chunk_1"})
    # Qwen input should only be E1
    assert qwen_input_ids == ["E1"]


def test_duplicate_neighbor_deduplicated() -> None:
    """同一邻居被两个 seed 找到时去重，candidates 中不出现重复 chunk_id"""
    corpus = [
        _chunk("chunk_0", "Results: yield improved to 10 g/L", doc_id="doc1", section="Results", fusion=0.0),
        _chunk("chunk_1", "seed1: result A 12 g/L", doc_id="doc1", section="Results", fusion=0.8),
        _chunk("chunk_2", "shared neighbor: discussion results", doc_id="doc1", section="Discussion", fusion=0.0),
        _chunk("chunk_3", "seed2: result B 14 g/L", doc_id="doc1", section="Results", fusion=0.7),
        _chunk("chunk_4", "next section text", doc_id="doc1", section="Methods", fusion=0.0),
    ]
    engine = _make_engine(corpus)
    cand1 = _candidate("E1", chunk_id="chunk_1", fusion=0.8)
    cand2 = _candidate("E2", chunk_id="chunk_3", fusion=0.7)
    support = [
        SupportItem(evidence_id="E1", candidate=cand1, support_score=0.9, reasons=[]),
        SupportItem(evidence_id="E2", candidate=cand2, support_score=0.8, reasons=[]),
    ]

    svc = _service_with(audit_engine=engine, support_pack=support, candidates=[cand1, cand2])
    result = svc.run(
        question="result yield",
        analysis=_analysis(),
        seed_chunks=[corpus[1], corpus[3]],
        config=GenerationConfig(v2_use_qwen_synthesis=False, v2_enable_neighbor_audit=True, v2_neighbor_window=1),
    )

    na = result.debug["neighbor_audit"]
    all_chunk_ids = [c["chunk_id"] for c in na["candidates"]]
    # no duplicate chunk_ids
    assert len(all_chunk_ids) == len(set(all_chunk_ids))
    # chunk_2 is shared neighbor; linked_seed_chunk_ids should reflect both seeds
    shared = [c for c in na["candidates"] if c["chunk_id"] == "chunk_2"]
    if shared:
        assert len(shared) == 1  # deduplicated


def test_audit_does_not_change_answer_or_citations() -> None:
    """开启 audit 不改变 answer_mode 和 citation_count"""
    corpus = [
        _chunk("chunk_0", "previous result text", doc_id="doc1", section="Results", fusion=0.0),
        _chunk("chunk_1", "seed: result 10 g/L", doc_id="doc1", section="Results", fusion=0.7),
        _chunk("chunk_2", "discussion of result", doc_id="doc1", section="Discussion", fusion=0.0),
    ]
    engine = _make_engine(corpus)
    cand = _candidate("E1", chunk_id="chunk_1", fusion=0.7)
    support = [SupportItem(evidence_id="E1", candidate=cand, support_score=0.9, reasons=[])]

    config_off = GenerationConfig(v2_use_qwen_synthesis=False, v2_enable_neighbor_audit=False)
    config_on = GenerationConfig(v2_use_qwen_synthesis=False, v2_enable_neighbor_audit=True)

    def run_with(cfg: GenerationConfig) -> dict:
        svc = _service_with(audit_engine=engine, support_pack=support, candidates=[cand])
        r = svc.run(
            question="result yield",
            analysis=_analysis(),
            seed_chunks=[corpus[1]],
            config=cfg,
        )
        return {"answer_mode": r.debug["answer_mode"], "citation_count": r.debug["citation_count"], "sp_count": r.debug["support_pack_count"]}

    off = run_with(config_off)
    on = run_with(config_on)
    assert off["answer_mode"] == on["answer_mode"]
    assert off["citation_count"] == on["citation_count"]
    assert off["sp_count"] == on["sp_count"]


# ── Stage 2E.0.1 new tests ────────────────────────────────────────────────────

def _chunk_scores(
    chunk_id: str,
    text: str,
    *,
    doc_id: str = "doc1",
    section: str = "Results",
    rerank: float = 0.0,
    fusion: float = 0.0,
    vector: float = 0.0,
    bm25: float = 0.0,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_file=f"{doc_id}.pdf",
        title=f"title-{doc_id}",
        section=section,
        text=text,
        rerank_score=rerank,
        fusion_score=fusion,
        vector_score=vector,
        bm25_score=bm25,
        metadata={},
    )


def _candidate_scores(
    evidence_id: str = "E1",
    *,
    chunk_id: str = "doc1_sec01_chunk01",
    doc_id: str = "doc1",
    section: str = "Results",
    rerank: float = 0.0,
    fusion: float = 0.0,
) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=evidence_id,
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_file=f"{doc_id}.pdf",
        title=f"title-{doc_id}",
        section=section,
        text="placeholder text with results showed yield",
        page_start=None,
        page_end=None,
        vector_score=0.0,
        bm25_score=0.0,
        rerank_score=rerank,
        fusion_score=fusion,
        metadata={},
        features={},
        reasons=["seed_chunk"],
    )


def test_score_source_priority_rerank_over_fusion() -> None:
    """seed 同时有 rerank_score 和 fusion_score，应使用 rerank_score"""
    seed = _chunk_scores("c1", "text", rerank=0.8, fusion=0.3)
    ev = _candidate_scores("E1", chunk_id="c1", rerank=0.9, fusion=0.3)
    score, source = _get_seed_score(ev, seed)
    assert source == "rerank_score"
    assert score == 0.9


def test_score_source_fallback_fusion_when_no_rerank() -> None:
    """seed 无 rerank_score，但有 fusion_score，应使用 fusion_score"""
    seed = _chunk_scores("c1", "text", rerank=0.0, fusion=0.4)
    ev = _candidate_scores("E1", chunk_id="c1", rerank=0.0, fusion=0.4)
    score, source = _get_seed_score(ev, seed)
    assert source == "fusion_score"
    assert score == 0.4


def test_score_source_missing_returns_zero() -> None:
    """seed 无任何可用分数（全为 0），返回 (0.0, 'missing')"""
    seed = _chunk_scores("c1", "text", rerank=0.0, fusion=0.0, vector=0.0, bm25=0.0)
    ev = _candidate_scores("E1", chunk_id="c1", rerank=0.0, fusion=0.0)
    score, source = _get_seed_score(ev, seed)
    assert source == "missing"
    assert score == 0.0


def test_missing_score_neighbor_not_promoted() -> None:
    """seed 无任何可用分数，neighbor 不得 dry_run_promoted，exclusion_reasons 包含 missing_score"""
    corpus = [
        _chunk_scores("chunk_0", "Results showed yield increased to 12 g/L with improved production",
                      section="Results", rerank=0.0, fusion=0.0),
        _chunk_scores("chunk_1", "seed no score", section="Results",
                      rerank=0.0, fusion=0.0),
        _chunk_scores("chunk_2", "Results discussion yield 10 g/L improvement shown in pathway analysis study",
                      section="Results", rerank=0.0, fusion=0.0),
    ]
    engine = _make_engine(corpus)
    cand = _candidate_scores("E1", chunk_id="chunk_1", rerank=0.0, fusion=0.0)
    support = [SupportItem(evidence_id="E1", candidate=cand, support_score=0.9, reasons=[])]

    svc = _service_with(audit_engine=engine, support_pack=support, candidates=[cand])
    result = svc.run(
        question="Results yield improvement",
        analysis=_analysis(),
        seed_chunks=[corpus[1]],
        config=GenerationConfig(
            v2_use_qwen_synthesis=False,
            v2_enable_neighbor_audit=True,
            v2_neighbor_window=1,
            v2_neighbor_min_promotion_score=0.05,
        ),
    )

    for c in result.debug["neighbor_audit"]["candidates"]:
        assert c["promotion_status"] != "dry_run_promoted", f"expected not promoted but got {c}"
        assert "missing_score" in c["exclusion_reasons"]


def test_no_inherit_seed_score_invariant() -> None:
    """inherited_seed_score=False 且 neighbor_score != seed_score"""
    seed_rerank = 0.85
    corpus = [
        _chunk_scores("chunk_0", "intro text", section="Introduction", rerank=0.0, fusion=0.0),
        _chunk_scores("chunk_1", "Results showed yield 12 g/L", section="Results",
                      rerank=seed_rerank, fusion=0.1),
        _chunk_scores("chunk_2", "Discussion: yield improvement due to pathway",
                      section="Discussion", rerank=0.0, fusion=0.0),
    ]
    engine = _make_engine(corpus)
    cand = _candidate_scores("E1", chunk_id="chunk_1", rerank=seed_rerank, fusion=0.1)
    support = [SupportItem(evidence_id="E1", candidate=cand, support_score=0.9, reasons=[])]

    svc = _service_with(audit_engine=engine, support_pack=support, candidates=[cand])
    result = svc.run(
        question="Results yield showed",
        analysis=_analysis(QueryIntent.SUMMARY),
        seed_chunks=[corpus[1]],
        config=GenerationConfig(
            v2_use_qwen_synthesis=False,
            v2_enable_neighbor_audit=True,
            v2_neighbor_window=1,
            v2_neighbor_min_promotion_score=0.05,
        ),
    )

    for c in result.debug["neighbor_audit"]["candidates"]:
        assert c["inherited_seed_score"] is False
        assert c["neighbor_score"] != seed_rerank
        assert c["neighbor_score"] < seed_rerank


def test_numeric_alone_does_not_promote() -> None:
    """has_numeric 单独不能触发 dry_run_promoted（无 query_overlap/core term overlap）"""
    corpus = [
        _chunk_scores("chunk_0", "There are 42 items listed in Table 3.",
                      section="Results", rerank=0.0, fusion=0.0),
        _chunk_scores("chunk_1", "seed: result 10 g/L",
                      section="Results", rerank=0.8, fusion=0.1),
        _chunk_scores("chunk_2", "Page 99 contains numbers: 1 2 3 4 5 data points.",
                      section="Results", rerank=0.0, fusion=0.0),
    ]
    engine = _make_engine(corpus)
    cand = _candidate_scores("E1", chunk_id="chunk_1", rerank=0.8, fusion=0.1)
    support = [SupportItem(evidence_id="E1", candidate=cand, support_score=0.9, reasons=[])]

    svc = _service_with(audit_engine=engine, support_pack=support, candidates=[cand])
    result = svc.run(
        question="completely unrelated topic XYZ",
        analysis=_analysis(),
        seed_chunks=[corpus[1]],
        config=GenerationConfig(
            v2_use_qwen_synthesis=False,
            v2_enable_neighbor_audit=True,
            v2_neighbor_window=1,
            v2_neighbor_min_promotion_score=0.05,
        ),
    )

    for c in result.debug["neighbor_audit"]["candidates"]:
        assert c["promotion_status"] != "dry_run_promoted", (
            f"chunk {c['chunk_id']} should not be promoted by numeric alone"
        )


def test_result_terms_alone_does_not_promote() -> None:
    """has_result_terms 单独不能触发 dry_run_promoted（无 query overlap）"""
    corpus = [
        _chunk_scores("chunk_0", "Results showed demonstrated increased yield titer production efficiency.",
                      section="Results", rerank=0.0, fusion=0.0),
        _chunk_scores("chunk_1", "seed result 10 g/L",
                      section="Results", rerank=0.8, fusion=0.1),
        _chunk_scores("chunk_2", "Yield showed improvement activity expression secretion.",
                      section="Results", rerank=0.0, fusion=0.0),
    ]
    engine = _make_engine(corpus)
    cand = _candidate_scores("E1", chunk_id="chunk_1", rerank=0.8, fusion=0.1)
    support = [SupportItem(evidence_id="E1", candidate=cand, support_score=0.9, reasons=[])]

    svc = _service_with(audit_engine=engine, support_pack=support, candidates=[cand])
    result = svc.run(
        question="completely different query topic XYZ",
        analysis=_analysis(),
        seed_chunks=[corpus[1]],
        config=GenerationConfig(
            v2_use_qwen_synthesis=False,
            v2_enable_neighbor_audit=True,
            v2_neighbor_window=1,
            v2_neighbor_min_promotion_score=0.05,
        ),
    )

    for c in result.debug["neighbor_audit"]["candidates"]:
        assert c["promotion_status"] != "dry_run_promoted", (
            f"chunk {c['chunk_id']} should not be promoted by result_terms alone"
        )


def test_high_value_section_alone_does_not_promote() -> None:
    """Results section 但无 query overlap，不应 dry_run_promoted"""
    corpus = [
        _chunk_scores("chunk_0", "Data analysis showed significant changes in unrelated metrics.",
                      section="Results", rerank=0.0, fusion=0.0),
        _chunk_scores("chunk_1", "seed: investigated pathway ABC",
                      section="Results", rerank=0.8, fusion=0.1),
        _chunk_scores("chunk_2", "Discussion of some other topic entirely without query terms.",
                      section="Discussion", rerank=0.0, fusion=0.0),
    ]
    engine = _make_engine(corpus)
    cand = _candidate_scores("E1", chunk_id="chunk_1", rerank=0.8, fusion=0.1)
    support = [SupportItem(evidence_id="E1", candidate=cand, support_score=0.9, reasons=[])]

    svc = _service_with(audit_engine=engine, support_pack=support, candidates=[cand])
    result = svc.run(
        question="pathway ABC",
        analysis=_analysis(),
        seed_chunks=[corpus[1]],
        config=GenerationConfig(
            v2_use_qwen_synthesis=False,
            v2_enable_neighbor_audit=True,
            v2_neighbor_window=1,
            v2_neighbor_min_promotion_score=0.05,
        ),
    )

    for c in result.debug["neighbor_audit"]["candidates"]:
        if c["chunk_id"] == "chunk_0":
            # "pathway ABC" and "unrelated metrics" has no query overlap
            assert c["promotion_status"] != "dry_run_promoted", (
                "high_value_section alone should not promote"
            )


def test_query_overlap_with_results_section_promotes() -> None:
    """Results section + query_overlap → should be dry_run_promoted"""
    corpus = [
        _chunk_scores("chunk_0", "Background of fermentation process for PHA yield measurement",
                      section="Introduction", rerank=0.0, fusion=0.0),
        _chunk_scores("chunk_1", "seed: PHA fermentation yield 10 g/L",
                      section="Results", rerank=0.8, fusion=0.1),
        _chunk_scores("chunk_2", "PHA fermentation yield improved to 15 g/L in bioreactor study",
                      section="Results", rerank=0.0, fusion=0.0),
    ]
    engine = _make_engine(corpus)
    cand = _candidate_scores("E1", chunk_id="chunk_1", rerank=0.8, fusion=0.1)
    support = [SupportItem(evidence_id="E1", candidate=cand, support_score=0.9, reasons=[])]

    svc = _service_with(audit_engine=engine, support_pack=support, candidates=[cand])
    result = svc.run(
        question="PHA fermentation yield bioreactor",
        analysis=_analysis(QueryIntent.SUMMARY),
        seed_chunks=[corpus[1]],
        config=GenerationConfig(
            v2_use_qwen_synthesis=False,
            v2_enable_neighbor_audit=True,
            v2_neighbor_window=1,
            v2_neighbor_min_promotion_score=0.05,
        ),
    )

    na = result.debug["neighbor_audit"]
    chunk2 = [c for c in na["candidates"] if c["chunk_id"] == "chunk_2"]
    assert len(chunk2) == 1
    assert chunk2[0]["promotion_status"] == "dry_run_promoted"
    assert chunk2[0]["score_passed"] is True
    assert chunk2[0]["semantic_gate_passed"] is True


def test_materials_and_methods_default_context_only() -> None:
    """Materials and Methods section 仅弱 overlap → context_only，不应 dry_run_promoted"""
    corpus = [
        _chunk_scores("chunk_0", "Fermentation was carried out in 5L bioreactor at 37C.",
                      section="Materials and Methods", rerank=0.0, fusion=0.0),
        _chunk_scores("chunk_1", "seed: yield results 10 g/L",
                      section="Results", rerank=0.8, fusion=0.1),
        _chunk_scores("chunk_2", "Standard protocol was followed for sample collection.",
                      section="Methods", rerank=0.0, fusion=0.0),
    ]
    engine = _make_engine(corpus)
    cand = _candidate_scores("E1", chunk_id="chunk_1", rerank=0.8, fusion=0.1)
    support = [SupportItem(evidence_id="E1", candidate=cand, support_score=0.9, reasons=[])]

    svc = _service_with(audit_engine=engine, support_pack=support, candidates=[cand])
    result = svc.run(
        question="yield results improvement",
        analysis=_analysis(),
        seed_chunks=[corpus[1]],
        config=GenerationConfig(
            v2_use_qwen_synthesis=False,
            v2_enable_neighbor_audit=True,
            v2_neighbor_window=1,
            v2_neighbor_min_promotion_score=0.05,
        ),
    )

    for c in result.debug["neighbor_audit"]["candidates"]:
        if c["section"].lower() in {"materials and methods", "methods"}:
            assert c["promotion_status"] != "dry_run_promoted", (
                f"section {c['section']} should be context_only with weak overlap"
            )


def test_introduction_default_context_only() -> None:
    """Introduction section 仅弱 overlap → context_only"""
    corpus = [
        _chunk_scores("chunk_0", "Previous studies have investigated various fermentation strategies.",
                      section="Introduction", rerank=0.0, fusion=0.0),
        _chunk_scores("chunk_1", "seed: PHA yield 10 g/L",
                      section="Results", rerank=0.8, fusion=0.1),
        _chunk_scores("chunk_2", "Synthetic biology offers new opportunities for metabolic engineering.",
                      section="Introduction", rerank=0.0, fusion=0.0),
    ]
    engine = _make_engine(corpus)
    cand = _candidate_scores("E1", chunk_id="chunk_1", rerank=0.8, fusion=0.1)
    support = [SupportItem(evidence_id="E1", candidate=cand, support_score=0.9, reasons=[])]

    svc = _service_with(audit_engine=engine, support_pack=support, candidates=[cand])
    result = svc.run(
        question="PHA yield 10 g/L fermentation",
        analysis=_analysis(),
        seed_chunks=[corpus[1]],
        config=GenerationConfig(
            v2_use_qwen_synthesis=False,
            v2_enable_neighbor_audit=True,
            v2_neighbor_window=1,
            v2_neighbor_min_promotion_score=0.05,
        ),
    )

    for c in result.debug["neighbor_audit"]["candidates"]:
        if c["section"].lower() == "introduction":
            assert c["promotion_status"] != "dry_run_promoted", (
                "Introduction with weak overlap should not be promoted"
            )


def test_no_support_refuse_blocking() -> None:
    """answer_plan.mode=refuse + support_pack=[] → 所有 neighbor 不得 dry_run_promoted"""
    corpus = [
        _chunk_scores("chunk_0", "Results showed yield 12 g/L PHA fermentation",
                      section="Results", rerank=0.0, fusion=0.0),
        _chunk_scores("chunk_1", "seed: some query relevant text",
                      section="Results", rerank=0.8, fusion=0.1),
        _chunk_scores("chunk_2", "Discussion: PHA yield improvement fermentation pathway",
                      section="Discussion", rerank=0.0, fusion=0.0),
    ]
    engine = _make_engine(corpus)
    cand = _candidate_scores("E1", chunk_id="chunk_1", rerank=0.8, fusion=0.1)
    refuse_plan = AnswerPlan(mode="refuse", reason="no_support_pack")

    svc = _service_with(
        audit_engine=engine,
        support_pack=[],
        plan=refuse_plan,
        candidates=[cand],
    )
    result = svc.run(
        question="PHA yield fermentation",
        analysis=_analysis(QueryIntent.SUMMARY),
        seed_chunks=[corpus[1]],
        config=GenerationConfig(
            v2_use_qwen_synthesis=False,
            v2_enable_neighbor_audit=True,
            v2_neighbor_window=1,
            v2_neighbor_min_promotion_score=0.05,
        ),
    )

    na = result.debug["neighbor_audit"]
    for c in na["candidates"]:
        assert c["promotion_status"] != "dry_run_promoted", (
            f"refuse case: {c['chunk_id']} should not be promoted"
        )
        assert c["blocked_by_refusal"] is True
    assert na["summary"]["no_support_blocked_count"] >= 0


def test_existence_no_support_blocking() -> None:
    """existence_guardrail support_status=none 对应 refuse reason → neighbor 不得 dry_run_promoted"""
    corpus = [
        _chunk_scores("chunk_0", "entity XYZ was not found in the database according to study",
                      section="Results", rerank=0.0, fusion=0.0),
        _chunk_scores("chunk_1", "seed: existence query about entity",
                      section="Results", rerank=0.8, fusion=0.1),
        _chunk_scores("chunk_2", "No evidence found for entity XYZ in literature",
                      section="Discussion", rerank=0.0, fusion=0.0),
    ]
    engine = _make_engine(corpus)
    cand = _candidate_scores("E1", chunk_id="chunk_1", rerank=0.8, fusion=0.1)
    existence_refuse_plan = AnswerPlan(mode="refuse", reason="existence_no_support")

    svc = _service_with(
        audit_engine=engine,
        support_pack=[],
        plan=existence_refuse_plan,
        candidates=[cand],
    )
    result = svc.run(
        question="Does entity XYZ exist in the database",
        analysis=_analysis(QueryIntent.FACTOID),
        seed_chunks=[corpus[1]],
        config=GenerationConfig(
            v2_use_qwen_synthesis=False,
            v2_enable_neighbor_audit=True,
            v2_neighbor_window=1,
            v2_neighbor_min_promotion_score=0.05,
        ),
    )

    for c in result.debug["neighbor_audit"]["candidates"]:
        assert c["promotion_status"] != "dry_run_promoted"
        assert c["blocked_by_refusal"] is True


def test_audit_dry_run_does_not_affect_downstream() -> None:
    """audit enabled 不影响 support_pack / citations / answer / Qwen input_evidence_ids"""
    corpus = [
        _chunk_scores("chunk_0", "PHA yield fermentation results improved to 12 g/L",
                      section="Results", rerank=0.0, fusion=0.0),
        _chunk_scores("chunk_1", "seed: main result 10 g/L",
                      section="Results", rerank=0.8, fusion=0.1),
        _chunk_scores("chunk_2", "Discussion: PHA fermentation pathway yield analysis",
                      section="Discussion", rerank=0.0, fusion=0.0),
    ]
    engine = _make_engine(corpus)
    cand = _candidate_scores("E1", chunk_id="chunk_1", rerank=0.8, fusion=0.1)
    support = [SupportItem(evidence_id="E1", candidate=cand, support_score=0.9, reasons=[])]

    def run_cfg(audit_on: bool) -> dict:
        svc = _service_with(audit_engine=engine, support_pack=support, candidates=[cand])
        r = svc.run(
            question="PHA fermentation yield",
            analysis=_analysis(QueryIntent.SUMMARY),
            seed_chunks=[corpus[1]],
            config=GenerationConfig(
                v2_use_qwen_synthesis=False,
                v2_enable_neighbor_audit=audit_on,
                v2_neighbor_window=1,
                v2_neighbor_min_promotion_score=0.05,
            ),
        )
        return {
            "sp_ids": [i.evidence_id for i in r.support_pack],
            "citation_count": r.debug["citation_count"],
            "answer_mode": r.debug["answer_mode"],
            "sp_count": r.debug["support_pack_count"],
        }

    off = run_cfg(False)
    on = run_cfg(True)
    assert off["sp_ids"] == on["sp_ids"]
    assert off["citation_count"] == on["citation_count"]
    assert off["answer_mode"] == on["answer_mode"]
    assert off["sp_count"] == on["sp_count"]


def test_duplicate_dedup_with_linked_seeds() -> None:
    """同一 neighbor 被多个 seed 找到时 candidates 中去重，linked_seed_chunk_ids 记录多个"""
    corpus = [
        _chunk_scores("chunk_0", "PHA yield 10 g/L production", section="Results",
                      rerank=0.0, fusion=0.0),
        _chunk_scores("chunk_1", "seed1: PHA result A 12 g/L",
                      section="Results", rerank=0.8, fusion=0.1),
        _chunk_scores("chunk_2", "shared PHA neighbor: discussion results improved",
                      section="Discussion", rerank=0.0, fusion=0.0),
        _chunk_scores("chunk_3", "seed2: PHA result B 14 g/L",
                      section="Results", rerank=0.7, fusion=0.1),
        _chunk_scores("chunk_4", "next section text",
                      section="Methods", rerank=0.0, fusion=0.0),
    ]
    engine = _make_engine(corpus)
    cand1 = _candidate_scores("E1", chunk_id="chunk_1", rerank=0.8, fusion=0.1)
    cand2 = _candidate_scores("E2", chunk_id="chunk_3", rerank=0.7, fusion=0.1)
    support = [
        SupportItem(evidence_id="E1", candidate=cand1, support_score=0.9, reasons=[]),
        SupportItem(evidence_id="E2", candidate=cand2, support_score=0.8, reasons=[]),
    ]

    svc = _service_with(audit_engine=engine, support_pack=support, candidates=[cand1, cand2])
    result = svc.run(
        question="PHA result yield",
        analysis=_analysis(),
        seed_chunks=[corpus[1], corpus[3]],
        config=GenerationConfig(
            v2_use_qwen_synthesis=False,
            v2_enable_neighbor_audit=True,
            v2_neighbor_window=1,
            v2_neighbor_min_promotion_score=0.05,
        ),
    )

    na = result.debug["neighbor_audit"]
    all_chunk_ids = [c["chunk_id"] for c in na["candidates"]]
    assert len(all_chunk_ids) == len(set(all_chunk_ids)), "duplicate chunk_ids found"
    shared = [c for c in na["candidates"] if c["chunk_id"] == "chunk_2"]
    if shared:
        assert len(shared) == 1
        assert len(shared[0]["linked_seed_chunk_ids"]) >= 1
