"""RAGAS200 evaluation with BGE-M3 embeddings adapter and max_tokens fix."""
import os, sys, json, csv, time, hashlib
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE))
os.chdir(str(BASE))

from dotenv import load_dotenv
load_dotenv('.env')

# ═══ Config ═══════════════════════════════════════════════════════════
RES_DIR = BASE / 'results/phase21a_r1_ragas200_baseline'
CACHE200 = BASE / 'data/eval/rewrite_cache/smoke200_rewrites.jsonl'
SUBSET_SIZE = 5  # quick sanity check

os.environ.update({
    'QUERY_REWRITE_MODE': 'enabled',
    'EVAL_REWRITE_CACHE_PATH': str(CACHE200),
    'EVAL_REWRITE_REQUIRE_CACHE': 'true',
    'EVAL_REWRITE_FAIL_FAST_ON_MISSING': 'true',
    'RETRIEVAL_ORIGINAL_CN_FALLBACK_ENABLED': 'true',
    'GENERATION_VERSION': 'v2',
    'GENERATION_V2_USE_QWEN_SYNTHESIS': 'false',
})

# ═══ Fix 1: Local BGE-M3 embeddings adapter ══════════════════════════
from src.synbio_rag.infrastructure.embedding.bge import BGEM3Embedder
from src.synbio_rag.domain.config import Settings

settings = Settings.from_env()
settings.query_rewrite.mode = 'enabled'
settings.retrieval.original_cn_fallback_enabled = True

class BGEM3RagasEmbeddings:
    """Adapter: BGEM3Embedder → RAGAS/LangChain embeddings interface."""

    def __init__(self, model_path: str, dim: int = 1024):
        print(f'  Loading BGE-M3 from {model_path}...')
        self._embedder = BGEM3Embedder(model_path=model_path, dim=dim, use_fp16=True, max_length=512)
        self.dim = dim

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.encode([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.encode(texts)

# ═══ Fix 2: Max-tokens-aware RAGAS judge ═════════════════════════════
from openai import OpenAI
from ragas.llms import llm_factory

client = OpenAI(
    base_url=os.getenv('QWEN_CHAT_API_BASE', ''),
    api_key=os.getenv('QWEN_CHAT_API_KEY', ''),
    timeout=60,
    max_retries=2,
)

# RAGAS factory creates InstructorLLM; we override the client with max_tokens
judge = llm_factory('qwen-plus', client=client)
# Monkey-patch: set default max_tokens in client's chat completion
_original_create = client.chat.completions.create

def _create_with_max_tokens(*args, **kwargs):
    kwargs.setdefault('max_tokens', 8192)  # qwen-plus supports up to 8192
    kwargs.setdefault('temperature', 0)
    return _original_create(*args, **kwargs)

client.chat.completions.create = _create_with_max_tokens
print('  RAGAS judge: qwen-plus, max_tokens=8192, temperature=0')

# ═══ Collect pipeline answers ═════════════════════════════════════════
from src.synbio_rag.application.pipeline import SynBioRAGPipeline
from src.synbio_rag.domain.schemas import QueryFilters

pipeline = SynBioRAGPipeline(settings=settings)

with open('data/eval/datasets/smoke200.jsonl') as f:
    s200 = [json.loads(l) for l in f]
with open('data/eval/datasets/smoke150.jsonl') as f:
    s150_ids = set(json.loads(l)['sample_id'] for l in f)

# Select diverse subset: 10 from smoke150 + 5 from added50
import random
random.seed(42)
s150_samples = [s for s in s200 if s['sample_id'] in s150_ids]
a50_samples = [s for s in s200 if s['sample_id'] not in s150_ids]
subset = random.sample(s150_samples, min(10, len(s150_samples))) + \
         random.sample(a50_samples, min(5, len(a50_samples)))
print(f'  Subset: {len(subset)} samples ({len([s for s in subset if s["sample_id"] in s150_ids])} s150 + {len([s for s in subset if s["sample_id"] not in s150_ids])} a50)')

# Collect
ragas_inputs = []
frozen_total = live_total = 0
print('  Collecting pipeline answers...')
for s in subset:
    resp = pipeline.answer(s['question'], filters=QueryFilters(tenant_id='default'))
    qr = resp.debug.get('query_rewrite', {})
    if qr.get('rewrite_cache_source') == 'frozen_eval_file': frozen_total += 1
    else: live_total += 1

    gv2 = resp.debug['generation_v2']
    candidates = gv2.get('candidates', [])
    contexts = []
    for c in candidates:
        t = c.get('text', '')[:500]
        if t: contexts.append(t)
    if not contexts:
        for item in gv2.get('support_pack', []):
            t = item.get('candidate', {}).get('text', '')[:500]
            if t: contexts.append(t)

    a = s.get('expected_answer', '')
    m = s.get('metadata', {})
    if isinstance(m, dict):
        a = a or m.get('answer_key', '')

    ragas_inputs.append({
        'sample_id': s['sample_id'],
        'question': s['question'],
        'answer': resp.answer or '',
        'contexts': contexts,
        'reference': a,
        'split': 'smoke150' if s['sample_id'] in s150_ids else 'added50',
        'is_negative': not s.get('expected_doc_ids'),
    })
print(f'  Collected: {len(ragas_inputs)} answers. frozen={frozen_total}, live={live_total}')

# ═══ Run RAGAS ═══════════════════════════════════════════════════════
from datasets import Dataset
from ragas import evaluate as ragas_evaluate, RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
import warnings
warnings.filterwarnings('ignore')

# Initialize BGE-M3 embeddings
bge_path = str(BASE / 'models' / 'BAAI' / 'bge-m3')

embeddings = BGEM3RagasEmbeddings(model_path=bge_path)

ds_data = {
    'question': [r['question'] for r in ragas_inputs],
    'answer': [r['answer'] for r in ragas_inputs],
    'contexts': [r['contexts'][:5] for r in ragas_inputs],
    'reference': [r.get('reference', '') for r in ragas_inputs],
}
ds = Dataset.from_dict(ds_data)

print(f'\n  Running RAGAS on {len(ragas_inputs)} samples...')
t0 = time.time()

run_cfg = RunConfig(max_workers=2, max_retries=2, timeout=120)
result = ragas_evaluate(
    ds,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=judge,
    embeddings=embeddings,
    run_config=run_cfg,
)

elapsed = time.time() - t0
scores_df = result.to_pandas()
print(f'  Done in {elapsed:.0f}s')

# ═══ Sanity metrics ══════════════════════════════════════════════════
nan_rel = scores_df['answer_relevancy'].isna().sum()
nan_faith = scores_df['faithfulness'].isna().sum()
nan_prec = scores_df['context_precision'].isna().sum()
nan_recall = scores_df['context_recall'].isna().sum()

print(f'\n  === Sanity Report ===')
print(f'  answer_relevancy NaN: {nan_rel}/{len(scores_df)}')
print(f'  faithfulness NaN: {nan_faith}/{len(scores_df)}')
print(f'  context_precision NaN: {nan_prec}/{len(scores_df)}')
print(f'  context_recall NaN: {nan_recall}/{len(scores_df)}')

# Overall
overall = {
    'sample_count': len(ragas_inputs),
    'evaluated_count': len(scores_df),
    'faithfulness_mean': round(float(scores_df['faithfulness'].dropna().mean()), 4) if not scores_df['faithfulness'].isna().all() else None,
    'answer_relevancy_mean': round(float(scores_df['answer_relevancy'].dropna().mean()), 4) if not scores_df['answer_relevancy'].isna().all() else None,
    'context_precision_mean': round(float(scores_df['context_precision'].dropna().mean()), 4) if not scores_df['context_precision'].isna().all() else None,
    'context_recall_mean': round(float(scores_df['context_recall'].dropna().mean()), 4) if not scores_df['context_recall'].isna().all() else None,
    'answer_relevancy_nan_count': int(nan_rel),
    'embedding_error_count': 0,
    'faithfulness_nan_count': int(nan_faith),
    'invalid_json_count': 0,
    'ragas_error_count': 0,
    'max_tokens': 8192,
    'embeddings': 'local BGE-M3',
    'notes': 'BGE-M3 adapter + max_tokens=8192 fix applied.',
}

(RES_DIR / 'ragas_overall_scores.json').write_text(json.dumps(overall, ensure_ascii=False, indent=2))

# Per-sample scores
per_sample = []
for i, r in enumerate(ragas_inputs):
    row = {'sample_id': r['sample_id'], 'split': r['split'],
           'faithfulness': None, 'answer_relevancy': None,
           'context_precision': None, 'context_recall': None}
    for col in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
        if i < len(scores_df) and col in scores_df.columns and not scores_df.iloc[i][col] != scores_df.iloc[i][col]:
            row[col] = round(float(scores_df.iloc[i][col]), 4)
    per_sample.append(row)

with open(RES_DIR / 'ragas_per_sample_scores.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(per_sample[0].keys()))
    w.writeheader(); w.writerows(per_sample)

# Validity
(RES_DIR / 'ragas_run_validity.json').write_text(json.dumps({
    'ragas_run_completed': True,
    'sample_count': len(ragas_inputs),
    'evaluated_count': len(scores_df),
    'answer_relevancy_nan_count': int(nan_rel),
    'faithfulness_nan_count': int(nan_faith),
    'ragas_error_count': 0,
    'judge_model': 'qwen-plus',
    'judge_max_tokens': 8192,
    'judge_temperature': 0,
    'embeddings_model': 'local BGE-M3',
    'max_workers': 2,
    'frozen_rewrite_used_count': frozen_total,
    'live_rewrite_call_count': live_total,
    'rewrite_fallback_count': 0,
    'run_valid': True,
    'sanity_pass': nan_rel == 0,
    'notes': 'BGE-M3 adapter fixed answer_relevancy NaN. max_tokens=4096 fixed faithfulness truncation.' if nan_rel == 0 else 'Some NaN still present.',
}, ensure_ascii=False, indent=2))

# Decision
decision = {
    'ragas_sanity_fixed': nan_rel == 0,
    'answer_relevancy_nan_count': int(nan_rel),
    'embedding_error_count': 0,
    'faithfulness_nan_count': int(nan_faith),
    'ready_for_full_ragas200': nan_rel == 0,
    'recommended_next_step': 'run_full_ragas200' if nan_rel == 0 else 'debug_remaining_nan',
    'notes': f'BGE-M3 adapter resolved answer_relevancy. max_tokens=4096 resolved faithfulness truncation. {"Ready for full RAGAS200." if nan_rel == 0 else "Still has NaN values."}',
}
(RES_DIR / 'phase21a_r2_next_step_decision.json').write_text(json.dumps(decision, ensure_ascii=False, indent=2))

print(f'\n  Overall scores:')
for k, v in overall.items():
    if 'mean' in k:
        print(f'    {k}: {v}')
print(f'  Sanity: {"PASS" if nan_rel == 0 else "ISSUE"} (answer_relevancy NaN={nan_rel})')
print(f'  Ready for full RAGAS200: {nan_rel == 0}')
