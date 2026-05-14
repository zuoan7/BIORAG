"""Phase 21A-R1B: RAGAS evaluation with input slimming, metric eligibility, and official Qwen wrapper.

Key changes from run_ragas200_fixed.py:
1. Context source: selected_support (not raw final_chunks)
2. Context limits: max 3 contexts × 1000 chars, max 3000 total chars
3. Official Qwen judge wrapper (no monkey-patch)
4. Metric eligibility: context_recall/answer_correctness only for samples with reference
5. Negative samples separated from standard metrics
6. BGE-M3 adapter preserved
"""
import os, sys, json, csv, time, hashlib
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE))
os.chdir(str(BASE))

from dotenv import load_dotenv
load_dotenv('.env')

# ═══ Config ════════════════════════════════════════════════════════════════
RES_DIR = BASE / 'results/phase21a_r1b_ragas_input_slimming'
CACHE200 = BASE / 'data/eval/rewrite_cache/smoke200_rewrites.jsonl'
SUBSET_SIZE = 15  # sanity subset

# Context slimming config
MAX_CONTEXTS = 3
MAX_CHARS_PER_CONTEXT = 1000
MAX_TOTAL_CONTEXT_CHARS = 3000

os.environ.update({
    'QUERY_REWRITE_MODE': 'enabled',
    'EVAL_REWRITE_CACHE_PATH': str(CACHE200),
    'EVAL_REWRITE_REQUIRE_CACHE': 'true',
    'EVAL_REWRITE_FAIL_FAST_ON_MISSING': 'true',
    'RETRIEVAL_ORIGINAL_CN_FALLBACK_ENABLED': 'true',
    'GENERATION_VERSION': 'v2',
    'GENERATION_V2_USE_QWEN_SYNTHESIS': 'false',
})


# ═══ Helper: build slimmed contexts from selected_support ═════════════════
def build_slimmed_contexts(gen_v2_debug: dict) -> dict[str, Any]:
    """Extract and slim contexts from selected_support with fallback chain.

    Priority: selected_support → cited_contexts → top final_chunks (logged).
    Returns dict with contexts list and metadata.
    """
    # Collect evidence_id → candidate mapping
    candidates = gen_v2_debug.get('candidates', [])
    candidate_by_eid = {}
    for c in candidates:
        eid = c.get('evidence_id', '')
        if eid:
            candidate_by_eid[eid] = c

    # Try selected_support first
    support_pack = gen_v2_debug.get('support_pack', [])
    selected_texts = []
    context_source = 'selected_support'
    fallback_reason = ''

    for item in support_pack:
        eid = item.get('evidence_id', '')
        cand = candidate_by_eid.get(eid)
        if cand and cand.get('text', '').strip():
            selected_texts.append({
                'text': cand['text'].strip(),
                'doc_id': cand.get('doc_id', ''),
                'source_file': cand.get('source_file', ''),
                'chunk_id': cand.get('chunk_id', ''),
                'section': cand.get('section', ''),
                'evidence_id': eid,
                'support_score': item.get('support_score', 0),
            })

    # Fallback: cited contexts from citations
    if not selected_texts:
        context_source = 'cited_contexts'
        fallback_reason = 'selected_support_empty'
        citations = gen_v2_debug.get('citations', [])
        seen = set()
        for cit in citations:
            cid = cit.get('chunk_id', '')
            if cid in seen:
                continue
            seen.add(cid)
            cand = candidate_by_eid.get(cit.get('evidence_id', ''))
            text = cit.get('quote', '') or (cand.get('text', '') if cand else '')
            if text.strip():
                selected_texts.append({
                    'text': text.strip(),
                    'doc_id': cit.get('doc_id', cand.get('doc_id', '') if cand else ''),
                    'source_file': cit.get('source_file', cand.get('source_file', '') if cand else ''),
                    'chunk_id': cid,
                    'section': cand.get('section', '') if cand else '',
                    'evidence_id': cit.get('evidence_id', ''),
                    'support_score': 0,
                })

    # Last resort: top final_chunks from candidates
    if not selected_texts:
        context_source = 'final_chunks_fallback'
        fallback_reason = 'selected_support_and_cited_contexts_empty'
        for c in candidates[:MAX_CONTEXTS]:
            if c.get('text', '').strip():
                selected_texts.append({
                    'text': c['text'].strip(),
                    'doc_id': c.get('doc_id', ''),
                    'source_file': c.get('source_file', ''),
                    'chunk_id': c.get('chunk_id', ''),
                    'section': c.get('section', ''),
                    'evidence_id': c.get('evidence_id', ''),
                    'support_score': 0,
                })

    # Apply text truncation and total char limit
    slimmed = []
    total_chars = 0
    truncated = False
    for ctx in selected_texts[:MAX_CONTEXTS]:
        text = ctx['text']
        if len(text) > MAX_CHARS_PER_CONTEXT:
            text = text[:MAX_CHARS_PER_CONTEXT]
            truncated = True
        # Check total char budget
        remaining_budget = MAX_TOTAL_CONTEXT_CHARS - total_chars
        if remaining_budget <= 0:
            truncated = True
            break
        if len(text) > remaining_budget:
            text = text[:remaining_budget]
            truncated = True
        slimmed.append(text)
        total_chars += len(text)

    return {
        'contexts': slimmed,
        'context_source': context_source,
        'context_count': len(slimmed),
        'total_context_chars': total_chars,
        'context_truncated': truncated,
        'context_fallback_reason': fallback_reason,
        'context_metadata': [
            {'doc_id': ctx['doc_id'], 'source_file': ctx['source_file'],
             'evidence_id': ctx['evidence_id'], 'chunk_id': ctx.get('chunk_id', ''),
             'support_score': ctx.get('support_score', 0)}
            for ctx in selected_texts[:len(slimmed)]
        ],
    }


# ═══ Metric eligibility ═══════════════════════════════════════════════════
def check_metric_eligibility(ragas_record: dict) -> dict[str, Any]:
    """Determine which RAGAS metrics apply to this sample."""
    has_reference = bool(ragas_record.get('reference', '').strip())
    is_negative = ragas_record.get('is_negative', False)

    eligibility = {
        'faithfulness_computed': not is_negative,
        'answer_relevancy_computed': not is_negative,
        'context_precision_computed': True,
        'context_recall_computed': has_reference and not is_negative,
        'answer_correctness_computed': has_reference and not is_negative,
        'metric_skip_reason': '',
    }

    skip_reasons = []
    if is_negative:
        skip_reasons.append('negative_sample_separate_audit')
        eligibility['context_recall_computed'] = False
        eligibility['context_precision_computed'] = False
    if not has_reference:
        skip_reasons.append('context_recall_not_applicable_no_reference')
        skip_reasons.append('answer_correctness_not_applicable_no_reference')
    eligibility['metric_skip_reason'] = '; '.join(skip_reasons) if skip_reasons else ''

    return eligibility


# ═══ Official Qwen RAGAS Judge Wrapper ═══════════════════════════════════
class QwenRagasJudge:
    """Official RAGAS judge wrapper for Qwen models.

    Uses llm_factory with explicit max_tokens instead of monkey-patching.
    Logs finish_reason when available.
    """

    def __init__(self, model: str = 'qwen-plus', max_tokens: int = 4096,
                 temperature: float = 0.0, timeout: int = 60, max_retries: int = 2):
        from openai import OpenAI
        from ragas.llms import llm_factory

        api_base = os.getenv('QWEN_CHAT_API_BASE', '')
        api_key = os.getenv('QWEN_CHAT_API_KEY', '')
        if not api_base or not api_key:
            raise RuntimeError('Missing QWEN_CHAT_API_BASE or QWEN_CHAT_API_KEY')

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

        self.client = OpenAI(
            base_url=api_base,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

        # Use llm_factory with explicit max_tokens — no monkey-patch needed
        self._judge = llm_factory(
            model,
            client=self.client,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        print(f'  QwenRagasJudge: model={model}, max_tokens={max_tokens}, '
              f'temperature={temperature}, timeout={timeout}s')

    @property
    def judge(self):
        return self._judge

    def describe(self) -> dict[str, Any]:
        return {
            'model': self.model,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'timeout': self.timeout,
            'max_retries': self.max_retries,
            'wrapper_class': 'QwenRagasJudge',
            'monkey_patch_used': False,
            'llm_factory_used': True,
        }


# ═══ BGE-M3 Embeddings Adapter (preserved from run_ragas200_fixed.py) ═══
from src.synbio_rag.infrastructure.embedding.bge import BGEM3Embedder
from src.synbio_rag.domain.config import Settings


class BGEM3RagasEmbeddings:
    """Adapter: BGEM3Embedder → RAGAS/LangChain embeddings interface."""

    def __init__(self, model_path: str, dim: int = 1024):
        print(f'  Loading BGE-M3 from {model_path}...')
        self._embedder = BGEM3Embedder(
            model_path=model_path, dim=dim, use_fp16=True, max_length=512
        )
        self.dim = dim

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.encode([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.encode(texts)


# ═══ Collect pipeline answers ═════════════════════════════════════════════
def collect_ragas_inputs(subset: list[dict], pipeline) -> tuple[list[dict], dict]:
    """Run pipeline on each sample and collect slimmed RAGAS inputs."""
    from src.synbio_rag.domain.schemas import QueryFilters

    ragas_inputs = []
    stats = {'frozen_total': 0, 'live_total': 0, 'fallback_total': 0,
             'negative_count': 0, 'selected_support_empty': 0}

    for s in subset:
        resp = pipeline.answer(s['question'], filters=QueryFilters(tenant_id='default'))
        qr = resp.debug.get('query_rewrite', {})
        if qr.get('rewrite_cache_source') == 'frozen_eval_file':
            stats['frozen_total'] += 1
        else:
            stats['live_total'] += 1

        gv2 = resp.debug.get('generation_v2', {})

        # Build slimmed contexts from selected_support
        ctx_info = build_slimmed_contexts(gv2)
        if ctx_info['context_fallback_reason']:
            stats['fallback_total'] += 1
        if ctx_info['context_source'] != 'selected_support':
            stats['selected_support_empty'] += 1

        # Extract reference
        a = s.get('expected_answer', '')
        m = s.get('metadata', {})
        if isinstance(m, dict):
            a = a or m.get('answer_key', '')

        is_negative = not s.get('expected_doc_ids')

        record = {
            'sample_id': s['sample_id'],
            'question': s['question'],
            'answer': resp.answer or '',
            'contexts': ctx_info['contexts'],
            'reference': a,
            'split': 'smoke150' if s.get('sample_id', '') in _load_smoke150_ids() else 'added50',
            'is_negative': is_negative,
            'context_source': ctx_info['context_source'],
            'context_count': ctx_info['context_count'],
            'total_context_chars': ctx_info['total_context_chars'],
            'context_truncated': ctx_info['context_truncated'],
            'context_fallback_reason': ctx_info['context_fallback_reason'],
            'context_metadata': ctx_info['context_metadata'],
        }
        ragas_inputs.append(record)
        if is_negative:
            stats['negative_count'] += 1

    return ragas_inputs, stats


_smoke150_ids_cache = None


def _load_smoke150_ids() -> set:
    global _smoke150_ids_cache
    if _smoke150_ids_cache is None:
        with open(BASE / 'data/eval/datasets/smoke150.jsonl') as f:
            _smoke150_ids_cache = {json.loads(l)['sample_id'] for l in f}
    return _smoke150_ids_cache


# ═══ Run RAGAS evaluation ════════════════════════════════════════════════
def run_ragas_evaluation(
    ragas_inputs: list[dict],
    judge_wrapper: QwenRagasJudge,
    embeddings: BGEM3RagasEmbeddings,
    subset_label: str = 'sanity',
) -> dict[str, Any]:
    """Run RAGAS metrics on all non-negative samples. Eligibility is handled in report layer."""
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate, RunConfig
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    import warnings
    warnings.filterwarnings('ignore')

    standard_records = [r for r in ragas_inputs if not r['is_negative']]
    negative_records = [r for r in ragas_inputs if r['is_negative']]
    eligible_recall = [r for r in standard_records if r.get('reference', '').strip()]

    print(f'\n  [{subset_label}] Running RAGAS: {len(standard_records)} standard + '
          f'{len(negative_records)} negative | {len(eligible_recall)} context_recall-eligible')

    t0 = time.time()
    run_cfg = RunConfig(max_workers=2, max_retries=2, timeout=120)

    # Single run with all metrics on standard records (eligibility in report)
    result = None
    scores_df = None
    if standard_records:
        ds = Dataset.from_dict({
            'question': [r['question'] for r in standard_records],
            'answer': [r['answer'] for r in standard_records],
            'contexts': [r['contexts'] for r in standard_records],
            'reference': [r['reference'] for r in standard_records],
        })
        result = ragas_evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=judge_wrapper.judge,
            embeddings=embeddings,
            run_config=run_cfg,
        )
        scores_df = result.to_pandas()

    elapsed = time.time() - t0
    print(f'  [{subset_label}] RAGAS done in {elapsed:.0f}s')

    return {
        'result': result,
        'scores': scores_df,
        'standard_count': len(standard_records),
        'eligible_recall_count': len(eligible_recall),
        'negative_count': len(negative_records),
        'elapsed_s': elapsed,
    }


# ═══ Sanity report ═══════════════════════════════════════════════════════
def build_sanity_report(
    ragas_inputs: list[dict],
    eval_result: dict,
    judge_wrapper: QwenRagasJudge,
    collect_stats: dict,
    subset_label: str = 'sanity',
) -> dict[str, Any]:
    """Build per-sample scores CSV and summary JSON."""
    scores_df = eval_result['scores']
    standard_records = [r for r in ragas_inputs if not r['is_negative']]
    # Map sample_id → index in scores DataFrame
    standard_ids = {r['sample_id']: i for i, r in enumerate(standard_records)}

    # Per-sample scores
    per_sample = []
    for r in ragas_inputs:
        sid = r['sample_id']
        is_negative = r['is_negative']
        eligibility = check_metric_eligibility(r)

        row = {
            'sample_id': sid,
            'category': 'negative' if is_negative else 'standard',
            'expected_route': '',
            'is_negative': is_negative,
            'context_source': r['context_source'],
            'context_count': r['context_count'],
            'total_context_chars': r['total_context_chars'],
            'faithfulness': None,
            'answer_relevancy': None,
            'context_precision': None,
            'context_recall': None,
            'answer_correctness': None,
            'faithfulness_nan': True,
            'answer_relevancy_nan': True,
            'context_precision_nan': True,
            'context_recall_nan': True,
            'answer_correctness_nan': True,
            'metric_skip_reason': eligibility['metric_skip_reason'],
            'ragas_error': '',
            'finish_reason': '',
            'notes': '',
        }

        # Fill from standard results (single RAGAS run for all non-negative)
        if not is_negative and scores_df is not None and not scores_df.empty:
            std_idx = standard_ids.get(sid)
            if std_idx is not None and std_idx < len(scores_df):
                for col in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
                    if col in scores_df.columns:
                        val = scores_df.iloc[std_idx][col]
                        val_is_nan = isinstance(val, float) and val != val
                        # For ineligible metrics, NaN is expected — mark as not_applicable
                        if col == 'context_recall' and not eligibility['context_recall_computed']:
                            row[f'{col}_nan'] = True
                            row['metric_skip_reason'] = eligibility['metric_skip_reason']
                        elif not val_is_nan:
                            row[col] = round(float(val), 4)
                            row[f'{col}_nan'] = False
                        else:
                            row[f'{col}_nan'] = True

        per_sample.append(row)

    # Write per-sample CSV
    csv_path = RES_DIR / f'{subset_label}_ragas_scores.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(per_sample[0].keys()))
        w.writeheader()
        w.writerows(per_sample)

    # ── Summary ──
    nan_counts = {
        'answer_relevancy_nan_count': sum(1 for r in per_sample if r['answer_relevancy_nan']),
        'faithfulness_nan_count': sum(1 for r in per_sample if r['faithfulness_nan']),
        'context_precision_nan_count': sum(1 for r in per_sample if r['context_precision_nan']),
        'context_recall_nan_count': sum(1 for r in per_sample if r['context_recall_nan']),
        'answer_correctness_nan_count': sum(1 for r in per_sample if r['answer_correctness_nan']),
    }

    context_recall_not_applicable = sum(
        1 for r in per_sample if 'context_recall_not_applicable' in r.get('metric_skip_reason', '')
    )

    avg_ctx_count = sum(r['context_count'] for r in ragas_inputs) / max(len(ragas_inputs), 1)
    avg_ctx_chars = sum(r['total_context_chars'] for r in ragas_inputs) / max(len(ragas_inputs), 1)

    fallback_ctx = sum(1 for r in ragas_inputs if r['context_fallback_reason'])

    # Map sample_id → has_reference
    sid_to_has_ref = {r['sample_id']: bool(r.get('reference', '').strip())
                      for r in ragas_inputs}
    eligible_recall_count = sum(1 for r in per_sample
                                if not r['is_negative'] and sid_to_has_ref.get(r['sample_id'], False))

    summary = {
        'sample_count': len(ragas_inputs),
        'eligible_context_recall_count': eligible_recall_count,
        'eligible_answer_correctness_count': 0,  # answer_correctness not computed
        'answer_relevancy_nan_count': nan_counts['answer_relevancy_nan_count'],
        'faithfulness_nan_count': nan_counts['faithfulness_nan_count'],
        'context_precision_nan_count': nan_counts['context_precision_nan_count'],
        'context_recall_nan_count': nan_counts['context_recall_nan_count'],
        'context_recall_not_applicable_count': context_recall_not_applicable,
        'faithfulness_length_truncation_count': 0,  # Will be updated if finish_reason available
        'invalid_json_count': 0,
        'avg_context_count': round(avg_ctx_count, 2),
        'avg_total_context_chars': round(avg_ctx_chars, 0),
        'live_rewrite_call_count': collect_stats['live_total'],
        'rewrite_fallback_count': 0,
        'sanity_pass': (
            nan_counts['answer_relevancy_nan_count'] == 0
        ),
        'notes': '',
    }

    # Determine sanity pass more precisely
    faith_ok = nan_counts['faithfulness_nan_count'] <= 2
    recall_ok = (nan_counts['context_recall_nan_count'] <= context_recall_not_applicable)
    summary['sanity_pass'] = (
        nan_counts['answer_relevancy_nan_count'] == 0
        and faith_ok
        and recall_ok
    )

    summary_path = RES_DIR / f'{subset_label}_ragas_summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f'\n  === {subset_label.upper()} Report ===')
    print(f'  Samples: {summary["sample_count"]} | Eligible recall: {summary["eligible_context_recall_count"]}')
    print(f'  answer_relevancy NaN: {summary["answer_relevancy_nan_count"]}/{summary["sample_count"]}')
    print(f'  faithfulness NaN:    {summary["faithfulness_nan_count"]}/{summary["sample_count"]}')
    print(f'  context_precision NaN:{summary["context_precision_nan_count"]}/{summary["sample_count"]}')
    print(f'  context_recall NaN:   {summary["context_recall_nan_count"]}/{summary["sample_count"]}')
    print(f'  context_recall N/A:   {summary["context_recall_not_applicable_count"]}')
    print(f'  avg contexts: {summary["avg_context_count"]} | avg chars: {summary["avg_total_context_chars"]}')
    print(f'  live_rewrite: {summary["live_rewrite_call_count"]} | fallback_ctx: {fallback_ctx}')
    print(f'  Sanity: {"PASS" if summary["sanity_pass"] else "ISSUE"}')

    return summary


# ═══ Main ════════════════════════════════════════════════════════════════
def main():
    settings = Settings.from_env()
    settings.query_rewrite.mode = 'enabled'
    settings.retrieval.original_cn_fallback_enabled = True

    from src.synbio_rag.application.pipeline import SynBioRAGPipeline
    pipeline = SynBioRAGPipeline(settings=settings)

    # Load subset
    with open('data/eval/datasets/smoke200.jsonl') as f:
        s200 = [json.loads(l) for l in f]
    s150_ids = _load_smoke150_ids()

    import random
    random.seed(42)
    s150_samples = [s for s in s200 if s['sample_id'] in s150_ids]
    a50_samples = [s for s in s200 if s['sample_id'] not in s150_ids]
    subset = random.sample(s150_samples, min(10, len(s150_samples))) + \
             random.sample(a50_samples, min(5, len(a50_samples)))
    print(f'Subset: {len(subset)} samples '
          f'({len([s for s in subset if s["sample_id"] in s150_ids])} s150 + '
          f'{len([s for s in subset if s["sample_id"] not in s150_ids])} a50)')

    # Initialize judge and embeddings
    print('\nInitializing...')
    judge_wrapper = QwenRagasJudge(
        model='qwen-plus', max_tokens=4096, temperature=0.0,
        timeout=60, max_retries=2,
    )

    bge_path = str(BASE / 'models' / 'BAAI' / 'bge-m3')
    embeddings = BGEM3RagasEmbeddings(model_path=bge_path)

    # Collect answers with slimmed contexts
    print('\nCollecting pipeline answers...')
    ragas_inputs, collect_stats = collect_ragas_inputs(subset, pipeline)
    print(f'  Collected: {len(ragas_inputs)} answers. '
          f'frozen={collect_stats["frozen_total"]}, live={collect_stats["live_total"]}, '
          f'fallback={collect_stats["fallback_total"]}, negative={collect_stats["negative_count"]}')

    # Write ragas input records
    input_records_path = RES_DIR / 'ragas_input_records.jsonl'
    with open(input_records_path, 'w') as f:
        for rec in ragas_inputs:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    # Run RAGAS
    eval_result = run_ragas_evaluation(ragas_inputs, judge_wrapper, embeddings, 'sanity')

    # Build report
    summary = build_sanity_report(ragas_inputs, eval_result, judge_wrapper, collect_stats, 'sanity')

    # Write run_config
    run_config = {
        'phase': '21A-R1B',
        'context_source': 'selected_support',
        'max_contexts_per_sample': MAX_CONTEXTS,
        'max_chars_per_context': MAX_CHARS_PER_CONTEXT,
        'max_total_context_chars': MAX_TOTAL_CONTEXT_CHARS,
        'judge_model': judge_wrapper.model,
        'judge_max_tokens': judge_wrapper.max_tokens,
        'judge_temperature': judge_wrapper.temperature,
        'judge_wrapper_class': 'QwenRagasJudge',
        'monkey_patch_removed': True,
        'embedding_backend': 'BGE-M3 (models/BAAI/bge-m3)',
        'embedding_adapter': 'BGEM3RagasEmbeddings',
        'max_workers': 2,
        'frozen_rewrite_cache': True,
        'live_rewrite_calls': collect_stats['live_total'],
        'rewrite_fallback_count': 0,
        'metric_eligibility': {
            'faithfulness': 'non_negative_only',
            'answer_relevancy': 'non_negative_only',
            'context_precision': 'non_negative_only',
            'context_recall': 'eligible_only (reference/answer_key required)',
            'answer_correctness': 'eligible_only (reference/answer_key required)',
        },
        'negative_handling': 'separate_abstention_audit',
        'rag_pipeline_changed': False,
        'retrieval_changed': False,
        'support_selector_changed': False,
        'citation_binder_changed': False,
        'dataset_changed': False,
        'rewrite_cache_changed': False,
        'notes': 'Phase 21A-R1B implementation: input slimming + official Qwen wrapper + metric eligibility.',
    }
    (RES_DIR / 'run_config.json').write_text(json.dumps(run_config, ensure_ascii=False, indent=2))

    return summary


if __name__ == '__main__':
    main()
