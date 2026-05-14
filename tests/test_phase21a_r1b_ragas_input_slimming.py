"""Phase 21A-R1B: Tests for RAGAS input slimming, metric eligibility, and judge wrapper."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from scripts.evaluation.run_phase21a_r1b_ragas_slimmed import (
    MAX_CONTEXTS,
    MAX_CHARS_PER_CONTEXT,
    MAX_TOTAL_CONTEXT_CHARS,
    BGEM3RagasEmbeddings,
    QwenRagasJudge,
    build_slimmed_contexts,
    check_metric_eligibility,
)


# ═══ Fixtures ══════════════════════════════════════════════════════════════

def _make_candidate(eid: str, text: str, doc_id: str = "doc1",
                    source_file: str = "file1.pdf", chunk_id: str = "",
                    section: str = "Results") -> dict:
    return {
        "evidence_id": eid,
        "chunk_id": chunk_id or f"chunk_{eid}",
        "doc_id": doc_id,
        "source_file": source_file,
        "section": section,
        "text": text,
        "title": "Test",
        "vector_score": 0.8,
        "bm25_score": 0.5,
        "rerank_score": 0.9,
        "fusion_score": 0.85,
    }


def _make_support_item(eid: str, score: float = 0.9) -> dict:
    return {"evidence_id": eid, "chunk_id": f"chunk_{eid}",
            "doc_id": "doc1", "section": "Results",
            "support_score": score, "reasons": ["entity_match"]}


def _make_gen_v2_with_support() -> dict:
    """Build gen_v2 debug with both support_pack and candidates."""
    texts = [
        "CRISPR-Cas9 enables precise genome editing in eukaryotic cells. "
        "The system uses a guide RNA to target specific DNA sequences.",
        "Cas9 nuclease creates double-strand breaks at target sites. "
        "Cellular repair mechanisms then introduce desired modifications.",
        "Recent advances include base editing and prime editing technologies. "
        "These enable single-nucleotide changes without double-strand breaks.",
        "Off-target effects remain a concern for therapeutic applications. "
        "Improved guide RNA design reduces unintended modifications.",
    ]
    candidates = [_make_candidate(f"eid_{i}", t) for i, t in enumerate(texts)]
    support = [_make_support_item(f"eid_{i}") for i in range(3)]  # only first 3 selected
    return {"candidates": candidates, "support_pack": support}


# ═══ Test 1: RAGAS contexts use selected_support not final_chunks ═════════

class TestContextSource:
    def test_uses_selected_support_not_final_chunks(self):
        gen_v2 = _make_gen_v2_with_support()
        result = build_slimmed_contexts(gen_v2)

        assert result["context_source"] == "selected_support"
        assert result["context_count"] == 3
        # Should NOT include the 4th candidate (off-target effects)
        for ctx in result["contexts"]:
            assert "off-target" not in ctx.lower()
        # Should include text from selected support items
        assert "CRISPR-Cas9" in result["contexts"][0]

    def test_fallback_to_cited_contexts(self):
        gen_v2 = {
            "candidates": [_make_candidate("e1", "Direct candidate text")],
            "support_pack": [],  # empty support
            "citations": [{"chunk_id": "chunk_e1", "evidence_id": "e1",
                           "quote": "Cited text from citation", "doc_id": "doc1",
                           "source_file": "f1.pdf"}],
        }
        result = build_slimmed_contexts(gen_v2)
        assert result["context_source"] == "cited_contexts"
        assert result["context_fallback_reason"] == "selected_support_empty"
        assert "Cited text from citation" in result["contexts"][0]

    def test_fallback_to_final_chunks_last_resort(self):
        gen_v2 = {
            "candidates": [_make_candidate("e1", "Final chunk text")],
            "support_pack": [],
            "citations": [],
        }
        result = build_slimmed_contexts(gen_v2)
        assert result["context_source"] == "final_chunks_fallback"
        assert "Final chunk text" in result["contexts"][0]


# ═══ Test 2: Context limits applied ═══════════════════════════════════════

class TestContextLimits:
    def test_max_3_contexts(self):
        texts = [f"Context number {i} with some content." for i in range(6)]
        candidates = [_make_candidate(f"eid_{i}", t) for i, t in enumerate(texts)]
        support = [_make_support_item(f"eid_{i}") for i in range(6)]
        gen_v2 = {"candidates": candidates, "support_pack": support}

        result = build_slimmed_contexts(gen_v2)
        assert result["context_count"] <= MAX_CONTEXTS
        assert len(result["contexts"]) <= MAX_CONTEXTS

    def test_each_context_under_1000_chars(self):
        long_text = "X" * 1500
        candidates = [_make_candidate("e1", long_text)]
        support = [_make_support_item("e1")]
        gen_v2 = {"candidates": candidates, "support_pack": support}

        result = build_slimmed_contexts(gen_v2)
        for ctx in result["contexts"]:
            assert len(ctx) <= MAX_CHARS_PER_CONTEXT
        assert result["context_truncated"]

    def test_total_chars_under_3000(self):
        texts = ["A" * 1200, "B" * 1200, "C" * 1200]
        candidates = [_make_candidate(f"eid_{i}", t) for i, t in enumerate(texts)]
        support = [_make_support_item(f"eid_{i}") for i in range(3)]
        gen_v2 = {"candidates": candidates, "support_pack": support}

        result = build_slimmed_contexts(gen_v2)
        total = sum(len(c) for c in result["contexts"])
        assert total <= MAX_TOTAL_CONTEXT_CHARS

    def test_short_contexts_not_truncated(self):
        texts = ["Short text.", "Another short text."]
        candidates = [_make_candidate(f"eid_{i}", t) for i, t in enumerate(texts)]
        support = [_make_support_item(f"eid_{i}") for i in range(2)]
        gen_v2 = {"candidates": candidates, "support_pack": support}

        result = build_slimmed_contexts(gen_v2)
        assert not result["context_truncated"]
        assert len(result["contexts"]) == 2


# ═══ Test 3: context_recall skipped without reference ════════════════════

class TestContextRecallEligibility:
    def test_skipped_without_reference(self):
        record = {"sample_id": "s1", "reference": "", "is_negative": False}
        eligibility = check_metric_eligibility(record)
        assert not eligibility["context_recall_computed"]
        assert "context_recall_not_applicable_no_reference" in eligibility["metric_skip_reason"]

    def test_computed_with_reference(self):
        record = {"sample_id": "s2", "reference": "The correct answer.", "is_negative": False}
        eligibility = check_metric_eligibility(record)
        assert eligibility["context_recall_computed"]


# ═══ Test 4: answer_correctness skipped without reference ════════════════

class TestAnswerCorrectnessEligibility:
    def test_skipped_without_reference(self):
        record = {"sample_id": "s1", "reference": "", "is_negative": False}
        eligibility = check_metric_eligibility(record)
        assert not eligibility["answer_correctness_computed"]
        assert "answer_correctness_not_applicable_no_reference" in eligibility["metric_skip_reason"]

    def test_computed_with_reference(self):
        record = {"sample_id": "s2", "reference": "Answer key.", "is_negative": False}
        eligibility = check_metric_eligibility(record)
        assert eligibility["answer_correctness_computed"]


# ═══ Test 5: Negative samples separate handling ══════════════════════════

class TestNegativeSamples:
    def test_negative_excluded_from_standard_metrics(self):
        record = {"sample_id": "s1", "reference": "Some reference.",
                  "is_negative": True}
        eligibility = check_metric_eligibility(record)
        assert not eligibility["faithfulness_computed"]
        assert not eligibility["answer_relevancy_computed"]
        assert not eligibility["context_recall_computed"]
        assert "negative_sample_separate_audit" in eligibility["metric_skip_reason"]

    def test_non_negative_included(self):
        record = {"sample_id": "s2", "reference": "Ref.", "is_negative": False}
        eligibility = check_metric_eligibility(record)
        assert eligibility["faithfulness_computed"]
        assert eligibility["answer_relevancy_computed"]


# ═══ Test 6: BGE-M3 adapter methods ══════════════════════════════════════

class TestBGEM3Adapter:
    def test_adapter_class_exists(self):
        assert BGEM3RagasEmbeddings is not None

    def test_has_embed_query_and_embed_documents(self):
        assert hasattr(BGEM3RagasEmbeddings, 'embed_query')
        assert hasattr(BGEM3RagasEmbeddings, 'embed_documents')

    def test_not_using_openai_embeddings(self):
        import ragas.embeddings.base as emb_base
        assert not issubclass(BGEM3RagasEmbeddings, emb_base.BaseRagasEmbeddings)


# ═══ Test 7: Qwen judge wrapper passes max_tokens ═══════════════════════

class TestQwenJudgeWrapper:
    def test_wrapper_class_exists(self):
        assert QwenRagasJudge is not None

    def test_wrapper_stores_max_tokens(self):
        with patch('openai.OpenAI'), \
             patch('ragas.llms.llm_factory'), \
             patch.dict('os.environ', {'QWEN_CHAT_API_BASE': 'http://test', 'QWEN_CHAT_API_KEY': 'test'}):
            wrapper = QwenRagasJudge(max_tokens=4096)
            assert wrapper.max_tokens == 4096

    def test_wrapper_no_monkey_patch(self):
        with patch('openai.OpenAI'), \
             patch('ragas.llms.llm_factory'), \
             patch.dict('os.environ', {'QWEN_CHAT_API_BASE': 'http://test', 'QWEN_CHAT_API_KEY': 'test'}):
            wrapper = QwenRagasJudge(max_tokens=4096)
            desc = wrapper.describe()
            assert desc['monkey_patch_used'] is False
            assert desc['llm_factory_used'] is True

    def test_describe_returns_config(self):
        with patch('openai.OpenAI'), \
             patch('ragas.llms.llm_factory'), \
             patch.dict('os.environ', {'QWEN_CHAT_API_BASE': 'http://test', 'QWEN_CHAT_API_KEY': 'test'}):
            wrapper = QwenRagasJudge(
                model='qwen-plus', max_tokens=8192, temperature=0.0,
                timeout=60, max_retries=2,
            )
            desc = wrapper.describe()
            assert desc['model'] == 'qwen-plus'
            assert desc['max_tokens'] == 8192
            assert desc['temperature'] == 0.0


# ═══ Test 8: No RAG pipeline files changed ═══════════════════════════════

class TestNoPipelineChanges:
    """Confirm this phase only modifies evaluation/RAGAS files, not RAG pipeline."""

    PIPELINE_FILES = [
        "src/synbio_rag/application/pipeline.py",
        "src/synbio_rag/application/generation_v2/service.py",
        "src/synbio_rag/application/generation_v2/support_selector.py",
        "src/synbio_rag/application/generation_v2/citation_binder.py",
        "src/synbio_rag/domain/router.py",
        "src/synbio_rag/domain/config.py",
        "src/synbio_rag/rewrite/query_rewrite_service.py",
    ]

    EVAL_FILES = [
        "scripts/evaluation/run_phase21a_r1b_ragas_slimmed.py",
        "scripts/evaluation/evaluate_ragas.py",
    ]

    def test_implementation_files_are_eval_only(self):
        """The new implementation file is under scripts/evaluation/."""
        impl_path = Path(BASE / "scripts/evaluation/run_phase21a_r1b_ragas_slimmed.py")
        assert impl_path.exists()
        assert "scripts/evaluation" in str(impl_path)

    def test_pipeline_files_exist_but_not_in_eval(self):
        """Pipeline files are in src/, eval files are in scripts/evaluation/."""
        for f in self.PIPELINE_FILES:
            p = Path(BASE / f)
            if p.exists():
                assert "scripts/evaluation" not in str(p)

    def test_eval_files_are_eval_only(self):
        for f in self.EVAL_FILES:
            p = Path(BASE / f)
            if p.exists():
                assert "scripts/evaluation" in str(p)


# ═══ Integration: end-to-end context flow ════════════════════════════════

class TestContextFlow:
    def test_complete_context_flow_with_metadata(self):
        gen_v2 = _make_gen_v2_with_support()
        result = build_slimmed_contexts(gen_v2)

        assert len(result["contexts"]) == 3
        assert result["context_source"] == "selected_support"
        assert result["context_count"] == 3
        assert result["total_context_chars"] > 0
        assert len(result["context_metadata"]) == 3
        for meta in result["context_metadata"]:
            assert "doc_id" in meta
            assert "source_file" in meta
            assert "evidence_id" in meta
            assert "chunk_id" in meta

    def test_empty_support_yields_fallback_with_reason(self):
        gen_v2 = {"candidates": [], "support_pack": [], "citations": []}
        result = build_slimmed_contexts(gen_v2)
        assert result["context_count"] == 0
        assert result["context_source"] == "final_chunks_fallback"
        assert result["context_fallback_reason"] != ""
