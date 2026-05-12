"""Phase 21A-9I: Negative abstention fix tests."""
import inspect
import pytest

from src.synbio_rag.domain.config import RetrievalConfig
from src.synbio_rag.domain.router import QueryRouter
from src.synbio_rag.domain.schemas import QueryAnalysis, QueryIntent


# ── Router tests ──────────────────────────────────────────────────────

class TestNegativeRouteDetection:
    def setup_method(self):
        self.router = QueryRouter(RetrievalConfig())

    def test_cn_explicit_abstain_clause_detected(self):
        queries = [
            "文库中是否有关于CRISPR基因编辑治疗镰状细胞病的临床试验数据？如果没有，请明确说明。",
            "文库中是否有关于mRNA疫苗III期临床试验的数据？如果没有，请明确说明证据不足。",
            "文库中是否有关于多奈哌齐的药代动力学研究？如果没有，请说明证据不足。",
            "文库中是否有关于植物来源黄酮类化合物抗肿瘤活性的系统评价？如果没有，请明确说明。",
        ]
        for q in queries:
            a = self.router.analyze(q)
            assert a.intent == QueryIntent.NEGATIVE, f"Expected NEGATIVE for: {q[:60]}"
            assert a.negative_intent_detected
            assert a.abstain_clause_detected

    def test_en_explicit_abstain_clause_detected(self):
        queries = [
            "Does the library contain clinical trial data on CRISPR gene editing for sickle cell disease? If not, explicitly state that it does not.",
            "Does the library contain data on Phase III clinical trial results of mRNA vaccines? If not, explicitly state that evidence is insufficient.",
        ]
        for q in queries:
            a = self.router.analyze(q)
            assert a.intent == QueryIntent.NEGATIVE, f"Expected NEGATIVE for: {q[:60]}"
            assert a.negative_intent_detected
            assert a.abstain_clause_detected

    def test_plain_existence_question_not_negative(self):
        """'是否有研究报道X' without explicit abstain clause → not negative."""
        queries = [
            "是否有研究报道CRISPR在代谢工程中的应用？",
            "是否有文献报道酶活性数据？",
            "是否包含2'-FL的合成路径信息？",
        ]
        for q in queries:
            a = self.router.analyze(q)
            assert a.intent != QueryIntent.NEGATIVE, f"Should NOT be negative: {q[:60]}"

    def test_summary_comparison_not_misclassified_as_negative(self):
        """Summary/comparison queries must not be intercepted."""
        queries = [
            "比较E. coli和B. subtilis作为NeuAc生产宿主的策略差异和产量表现。",
            "总结文库中利用工程酵母生产2'-FL的策略。",
        ]
        for q in queries:
            a = self.router.analyze(q)
            assert a.intent != QueryIntent.NEGATIVE, f"Should NOT be negative: {q[:60]}"

    def test_factoid_not_misclassified(self):
        queries = [
            "根据文库，细菌摄取芳香化合物时涉及哪些膜转运家族？请只基于证据回答。",
            "是什么机制导致了启动子强度差异？",
            "有哪些关键基因参与了补救途径？",
        ]
        for q in queries:
            a = self.router.analyze(q)
            assert a.intent != QueryIntent.NEGATIVE, f"Should NOT be negative: {q[:60]}"

    def test_negative_analysis_has_trace_fields(self):
        a = self.router.analyze("是否有关于X的数据？如果没有，请明确说明。")
        assert a.negative_intent_detected
        assert a.abstain_clause_detected
        assert a.intent_before_negative_guard in ("factoid", "unknown", "")


class TestQueryIntentHasNegative:
    def test_negative_in_enum(self):
        assert QueryIntent("negative") == QueryIntent.NEGATIVE
        assert QueryIntent.NEGATIVE.value == "negative"

    def test_negative_not_equal_to_other_intents(self):
        assert QueryIntent.NEGATIVE != QueryIntent.FACTOID
        assert QueryIntent.NEGATIVE != QueryIntent.SUMMARY


# ── Support selector tests ────────────────────────────────────────────

class TestNegativeSupportSuppression:
    def test_negative_intent_returns_empty_support(self):
        from src.synbio_rag.application.generation_v2.support_selector import SupportPackSelector
        from src.synbio_rag.application.generation_v2.models import EvidenceCandidate
        from src.synbio_rag.domain.config import GenerationConfig

        selector = SupportPackSelector()
        analysis = QueryAnalysis(
            intent=QueryIntent.NEGATIVE,
            requires_external_tools=False,
            search_limit=10,
            rerank_top_k=10,
            notes="negative test",
            negative_intent_detected=True,
            abstain_clause_detected=True,
            intent_before_negative_guard="factoid",
        )
        candidates = [EvidenceCandidate(
            evidence_id="E1", doc_id="doc_001", chunk_id="chunk_001",
            section="Results", text="test text content long enough for checks",
            title="Test", source_file="test.pdf",
            page_start=1, page_end=2,
            rerank_score=1.0, vector_score=0.5, bm25_score=0.5,
            fusion_score=0.8, features={}, reasons=[], metadata={},
        )]
        config = GenerationConfig()
        result = selector.select("test question", analysis, candidates, config)
        assert len(result) == 0, f"Expected empty support for NEGATIVE, got {len(result)}"
        assert selector.last_selection_debug.get("negative_guard") is True

    def test_factoid_not_suppressed(self):
        from src.synbio_rag.application.generation_v2.support_selector import SupportPackSelector
        from src.synbio_rag.application.generation_v2.models import EvidenceCandidate
        from src.synbio_rag.domain.config import GenerationConfig

        selector = SupportPackSelector()
        analysis = QueryAnalysis(
            intent=QueryIntent.FACTOID,
            requires_external_tools=False,
            search_limit=10,
            rerank_top_k=10,
            notes="factoid test",
        )
        candidates = [EvidenceCandidate(
            evidence_id="E1", doc_id="doc_001", chunk_id="chunk_001",
            section="Results", text="test text content long enough for quality checks with sufficient words",
            title="Test", source_file="test.pdf",
            page_start=1, page_end=2,
            rerank_score=2.0, vector_score=0.5, bm25_score=0.5,
            fusion_score=1.0,
            features={"has_numeric": True, "has_result_terms": True,
                      "section_type": "Results", "text_length": 80},
            reasons=[], metadata={},
        )]
        config = GenerationConfig()
        result = selector.select("what is X?", analysis, candidates, config)
        assert len(result) >= 1


# ── No sample special case ────────────────────────────────────────────

class TestNoSampleSpecialCase:
    def test_router_no_sample_id_special_case(self):
        src = inspect.getsource(QueryRouter.analyze)
        for banned in ["ent_021", "ent_091", "ent_092", "ent_093", "ent_095"]:
            assert banned not in src, f"Banned {banned} in router"

    def test_no_expected_doc_ids_in_router(self):
        src = inspect.getsource(QueryRouter.analyze)
        assert "expected_doc" not in src
        assert "expected_source" not in src

    def test_negative_guard_in_selector_no_sample_ids(self):
        from src.synbio_rag.application.generation_v2 import support_selector as ss
        src = inspect.getsource(ss.SupportPackSelector.select)
        # But the negative guard code is in select() above the all_scored line
        # Just check no sample IDs in the module
        full_src = inspect.getsource(ss)
        for banned in ["ent_021", "ent_091", "ent_092", "ent_093", "ent_095"]:
            assert banned not in full_src, f"Banned {banned} in support_selector"
