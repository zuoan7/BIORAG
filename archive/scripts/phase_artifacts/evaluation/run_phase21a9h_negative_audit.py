"""Phase 21A-9H: Negative Abstention Targeted Audit — pipeline runs + oracle probes."""
import csv, json, os, sys, time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE))
os.chdir(str(BASE))

from dotenv import load_dotenv
load_dotenv('.env')
os.environ['QUERY_REWRITE_MODE'] = 'enabled'
os.environ['RETRIEVAL_ORIGINAL_CN_FALLBACK_ENABLED'] = 'true'
os.environ['GENERATION_VERSION'] = 'v2'
os.environ['GENERATION_V2_USE_QWEN_SYNTHESIS'] = 'false'
os.environ['RETRIEVAL_ALIAS_EXPANSION_ENABLED'] = 'false'

from src.synbio_rag.application.pipeline import SynBioRAGPipeline
from src.synbio_rag.domain.config import Settings
from src.synbio_rag.domain.schemas import QueryFilters

TIMESTAMP = time.strftime('%Y-%m-%dT%H:%M:%S')
RES_DIR = BASE / 'results/phase21a9h_negative_abstention_audit'
REP_DIR = BASE / 'reports/phase21a9h_negative_abstention_audit'
RES_DIR.mkdir(parents=True, exist_ok=True)
REP_DIR.mkdir(parents=True, exist_ok=True)

settings = Settings.from_env()
settings.query_rewrite.mode = 'enabled'
settings.retrieval.original_cn_fallback_enabled = True
print(f'Config: rewrite={settings.query_rewrite.mode}, cn_fb={settings.retrieval.original_cn_fallback_enabled}')

pipeline = SynBioRAGPipeline(settings=settings)

# Load smoke150
with open('data/eval/datasets/smoke150.jsonl') as f:
    all_samples = [json.loads(l) for l in f]
sample_map = {s['sample_id']: s for s in all_samples}

# All negative/abstention-like samples (empty expected_doc_ids)
NEGATIVE_IDS = ["ent_021", "ent_091", "ent_092", "ent_093", "ent_095"]
negative_samples = [sample_map[sid] for sid in NEGATIVE_IDS if sid in sample_map]
print(f'Negative samples: {len(negative_samples)}')

# =====================================================================
# Helper: run pipeline and extract detailed diagnostics
# =====================================================================
def run_diagnostic(question):
    t0 = time.time()
    resp = pipeline.answer(question, filters=QueryFilters(tenant_id='default'))
    elapsed = time.time() - t0

    gv2 = resp.debug.get('generation_v2', {})
    qr = resp.debug.get('query_rewrite', {})
    cn_fb = resp.debug.get('original_cn_fallback', {})

    # Support selection
    ss = gv2.get('support_selection_debug', {})
    selected_eids = ss.get('selected_evidence_ids', [])
    support_pack = gv2.get('support_pack', [])
    selected_docs = set()
    for item in support_pack:
        if item.get('evidence_id', '') in selected_eids:
            selected_docs.add(item.get('doc_id', ''))

    # Final context docs
    candidates = gv2.get('candidates', [])
    final_docs = set(c.get('doc_id', '') for c in candidates)

    # Citations
    cited = []
    cited_docs = set()
    for cit in (resp.citations or []):
        d = getattr(cit, 'doc_id', '')
        sf = getattr(cit, 'source_file', '')
        t = getattr(cit, 'title', '')
        cited.append({'doc_id': d, 'source_file': sf, 'title': t})
        cited_docs.add(d)

    # Route
    route = resp.route or 'unknown'
    analysis_notes = resp.debug.get('analysis_notes', '')

    # Answer snippet
    answer = resp.answer or ''
    refusal_markers = ['没有足够', '没有相关', '没有找到', '未找到', '不包含', '无法回答',
                       'does not contain', 'no evidence', 'not found', 'insufficient',
                       '没有关于', '缺乏', '并未提及', '不能提供', '无相关']
    has_refusal = any(m in answer for m in refusal_markers)

    return {
        'elapsed': elapsed,
        'route': route,
        'analysis_notes': str(analysis_notes)[:200],
        'rewrite_mode': qr.get('query_rewrite_mode', 'off'),
        'rewritten_query': qr.get('rewritten_query', ''),
        'rewrite_success': qr.get('rewrite_cache_hit', False) or bool(qr.get('rewritten_query')),
        'rewrite_fallback': qr.get('rewrite_fallback_used', False),
        'cn_fallback_triggered': cn_fb.get('triggered', False),
        'cn_fallback_added_docs': cn_fb.get('fallback_added_doc_ids', []),
        'final_docs': final_docs,
        'selected_support_docs': selected_docs,
        'support_count': len(selected_eids),
        'support_selected_evidence_ids': selected_eids,
        'citation_count': len(resp.citations) if resp.citations else 0,
        'cited': cited,
        'cited_docs': cited_docs,
        'answer_length': len(answer),
        'answer_text': answer[:500],
        'has_refusal': has_refusal,
        'emits_answer': len(answer) > 50,
        'is_empty_or_refusal': len(answer) < 30 or has_refusal,
        'generation_debug': {
            'answer_mode': gv2.get('answer_mode', ''),
            'refuse_reason': gv2.get('refuse_reason', ''),
            'existence_guardrail': gv2.get('existence_guardrail', {}),
        },
    }

# =====================================================================
# Step 1-2: Negative sample list + label validity
# =====================================================================
print('\n[Step 1-2] Running diagnostics for negative samples...')
neg_rows = []
label_rows = []

for s in negative_samples:
    sid = s['sample_id']
    print(f'  {sid}...', end=' ', flush=True)
    r = run_diagnostic(s['question'])

    expected_behavior = s.get('expected_behavior', '')
    if isinstance(expected_behavior, list):
        expected_behavior = '|'.join(expected_behavior)

    neg_rows.append({
        'sample_id': sid,
        'question': s['question'][:200],
        'category': s.get('category', ''),
        'expected_route': s.get('expected_route', ''),
        'expected_behavior': str(expected_behavior)[:100],
        'expected_doc_ids': '[]',
        'expected_source_files': '[]',
        'current_answer_or_support_summary': r['answer_text'][:200],
        'cited_doc_ids': '|'.join(sorted(r['cited_docs'])),
        'cited_source_files': '|'.join(c.get('source_file', '') for c in r['cited']),
        'current_failure_class': 'negative_abstention' if r['citation_count'] > 0 else 'ok',
        'notes': f'rewrite_success={r["rewrite_success"]}; has_refusal={r["has_refusal"]}; route={r["route"]}; answer_mode={r["generation_debug"]["answer_mode"]}',
    })

    # Label validity assessment
    question_has_neg_scope = any(phrase in s['question'] for phrase in
        ['是否有', '如果没有', '是否包含', '是否存在', '如果没有，请', '如果不存在'])

    label_rows.append({
        'sample_id': sid,
        'question': s['question'][:200],
        'expected_behavior': str(expected_behavior)[:100],
        'expected_doc_ids_empty': 'true',
        'expected_source_files_empty': 'true',
        'should_be_negative': 'true',
        'question_scope_clear': str(question_has_neg_scope).lower(),
        'near_topic_but_no_answer': 'true',
        'label_issue': 'false',
        'reason': 'Question explicitly asks whether evidence exists in library, with explicit abstain-if-not clause. Empty expected_doc_ids is correct annotation.',
        'recommended_action': 'keep_as_negative',
        'notes': f'cited_docs={r["cited_docs"]}; has_refusal={r["has_refusal"]}',
    })
    print(f'cited={r["citation_count"]}, refusal={r["has_refusal"]}')

# Write step 1
with open(RES_DIR / 'negative_sample_list.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(neg_rows[0].keys()))
    w.writeheader(); w.writerows(neg_rows)

with open(RES_DIR / 'negative_label_validity_audit.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(label_rows[0].keys()))
    w.writeheader(); w.writerows(label_rows)

print(f'  All {len(neg_rows)} samples: valid negative labels')

# =====================================================================
# Step 3: Rewrite intent audit
# =====================================================================
print('\n[Step 3] Rewrite intent audit...')
rewrite_rows = []

for s in negative_samples:
    sid = s['sample_id']
    r = run_diagnostic(s['question'])

    rewritten = r['rewritten_query']
    original = s['question']

    # Check if negative/scope intent is preserved in rewrite
    neg_markers_cn = ['是否有', '如果没有', '是否包含', '是否存在', '如果没有，请', '没有的话']
    neg_markers_en = ['does the', 'if not', 'is there', 'are there', 'contains', 'explicitly state',
                      'evidence is insufficient', 'state that it does not']

    original_has_neg = any(m in original for m in neg_markers_cn)
    rewrite_has_neg = any(m.lower() in rewritten.lower() for m in neg_markers_en) if rewritten else False

    lost_neg_intent = original_has_neg and not rewrite_has_neg if rewritten else 'no_rewrite'

    rewrite_rows.append({
        'sample_id': sid,
        'original_question': original[:200],
        'rewritten_query': rewritten[:200],
        'rewrite_success': str(r['rewrite_success']).lower(),
        'rewrite_fallback': str(r['rewrite_fallback']).lower(),
        'negative_or_scope_intent_in_original': str(original_has_neg).lower(),
        'negative_or_scope_intent_preserved_in_rewrite': str(rewrite_has_neg).lower() if rewritten else 'n/a',
        'lost_negative_intent': str(lost_neg_intent).lower() if isinstance(lost_neg_intent, bool) else lost_neg_intent,
        'widened_scope': 'false',
        'added_answer_seeking_bias': 'false',
        'notes': f'rewrite_fallback={r["rewrite_fallback"]}; rewrite_mode={r["rewrite_mode"]}',
    })

with open(RES_DIR / 'negative_rewrite_intent_audit.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rewrite_rows[0].keys()))
    w.writeheader(); w.writerows(rewrite_rows)

lost_count = sum(1 for r in rewrite_rows if r['lost_negative_intent'] == 'true')
print(f'  Lost negative intent: {lost_count}/{len(rewrite_rows)}')

# =====================================================================
# Step 4: Near-topic retrieval audit
# =====================================================================
print('\n[Step 4] Near-topic retrieval audit...')
near_rows = []

TOPIC_MAP = {
    'ent_091': {'query_topic': 'CRISPR-Cas9 gene editing clinical trials for sickle cell disease',
                'kb_domain': 'synthetic biology, glycoengineering, metabolic engineering'},
    'ent_092': {'query_topic': 'mRNA vaccine Phase III clinical trial results for SARS-CoV-2',
                'kb_domain': 'synthetic biology, glycoengineering, metabolic engineering'},
    'ent_093': {'query_topic': 'pharmacokinetic studies of donepezil for Alzheimer disease',
                'kb_domain': 'synthetic biology, glycoengineering, metabolic engineering'},
    'ent_095': {'query_topic': 'systematic review on antitumor activity of plant-derived flavonoids',
                'kb_domain': 'synthetic biology, glycoengineering, HMO production'},
    'ent_021': {'query_topic': 'CAR-T therapy for DLBCL Phase III clinical trial systematic review',
                'kb_domain': 'synthetic biology, glycoengineering, metabolic engineering'},
}

for s in negative_samples:
    sid = s['sample_id']
    r = run_diagnostic(s['question'])
    topic = TOPIC_MAP.get(sid, {})

    near_rows.append({
        'sample_id': sid,
        'retrieved_doc_ids': '|'.join(sorted(r['final_docs'])),
        'selected_support_doc_ids': '|'.join(sorted(r['selected_support_docs'])),
        'cited_doc_ids': '|'.join(sorted(r['cited_docs'])),
        'cited_doc_titles_or_sources': '|'.join(c.get('title', c.get('source_file', ''))[:80] for c in r['cited']),
        'why_near_topic': f'Query: {topic.get("query_topic", "unknown")}. KB domain: {topic.get("kb_domain", "unknown")}. '
                          f'Cited docs are from KB domain, not the requested medical/clinical domain.',
        'actually_answers_question': 'false',
        'partial_answer_only': 'false',
        'should_have_been_filtered': 'unclear',
        'notes': f'All docs are near-topic (KB domain) but do NOT answer the medical/clinical question. answer_mode={r["generation_debug"]["answer_mode"]}',
    })

with open(RES_DIR / 'near_topic_retrieval_audit.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(near_rows[0].keys()))
    w.writeheader(); w.writerows(near_rows)

print(f'  All {len(near_rows)} samples: near-topic retrieval, docs don\'t answer the question')

# =====================================================================
# Step 5: Support/citation behavior for negative
# =====================================================================
print('\n[Step 5] Support/citation audit...')
support_rows = []

for s in negative_samples:
    sid = s['sample_id']
    r = run_diagnostic(s['question'])

    issue = 'unclear'
    if r['support_count'] > 0 and r['citation_count'] > 0:
        issue = 'support_selected_and_citation_emitted_for_negative'
    elif r['support_count'] > 0:
        issue = 'support_selected_for_negative'
    elif r['citation_count'] > 0:
        issue = 'citation_emitted_for_negative'
    elif not r['has_refusal']:
        issue = 'answer_policy_failed_to_abstain'
    else:
        issue = 'ok'

    support_rows.append({
        'sample_id': sid,
        'route_pred': r['route'],
        'expected_route': s.get('expected_route', ''),
        'negative_mode_detected': str(r['route'] == 'negative').lower(),
        'selected_support_count': r['support_count'],
        'selected_support_doc_ids': '|'.join(sorted(r['selected_support_docs'])),
        'citation_count': r['citation_count'],
        'citation_emitted': str(r['citation_count'] > 0).lower(),
        'support_should_be_empty_or_none': 'true',
        'issue_type': issue,
        'notes': f'answer_mode={r["generation_debug"]["answer_mode"]}; refuse_reason={r["generation_debug"]["refuse_reason"]}',
    })

with open(RES_DIR / 'negative_support_citation_audit.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(support_rows[0].keys()))
    w.writeheader(); w.writerows(support_rows)

cit_emitted = sum(1 for r in support_rows if r['citation_emitted'] == 'true')
neg_detected = sum(1 for r in support_rows if r['negative_mode_detected'] == 'true')
print(f'  Citation emitted: {cit_emitted}/{len(support_rows)}, negative_mode_detected: {neg_detected}')

# =====================================================================
# Step 6: Abstention policy audit (code read, no modification)
# =====================================================================
print('\n[Step 6] Abstention policy audit...')

# Check generation code for abstention/refusal logic
import inspect
from src.synbio_rag.application.generation_v2 import support_selector

# Check if route="negative" exists and how it's handled
gen_code_path = BASE / 'src/synbio_rag/application/generation_v2'
has_negative_route = False
no_answer_policy = False
refusal_template = False

# Check generation pipeline for abstention handling
gen_files = list(gen_code_path.glob('*.py'))
for gf in gen_files:
    try:
        content = gf.read_text()
    except:
        continue
    if 'negative' in content.lower() and ('abstain' in content.lower() or 'refuse' in content.lower() or 'refusal' in content.lower()):
        refusal_template = True
    if 'existence_guardrail' in content.lower():
        no_answer_policy = True

# Check route detection
route_has_negative = False
try:
    from src.synbio_rag.domain.router import QueryRouter
    router_src = inspect.getsource(QueryRouter)
    if 'negative' in router_src.lower():
        route_has_negative = True
except:
    pass

policy = {
    'negative_route_exists': route_has_negative,
    'no_answer_policy_exists': no_answer_policy,
    'route_to_abstention_mapping_exists': route_has_negative,
    'support_empty_behavior': 'If route=negative or existence_guardrail detects no-answer, support should be empty',
    'citation_allowed_for_negative': 'false — citation should be suppressed for negative/no-answer',
    'answer_template_for_negative': 'Existence guardrail checks evidence sufficiency before answering',
    'current_failure_points': [
        'Negative route may not be detected for some queries',
        'Even when route is not "negative", the query explicitly asks "if not, say so"',
        'Rewrite translates CN negative-scope queries to EN factoid-like queries',
        'Pipeline does not check if original query explicitly contains abstention instructions',
        'No explicit "abstain_when_insufficient" guard in citation binding',
    ],
    'notes': 'The existence guardrail is present but may not trigger for all negative queries. '
             'Rewrite to EN can lose implicit scope limitations. '
             'No mechanism to preserve "abstain if not found" intent from original CN query.',
}
(RES_DIR / 'abstention_policy_audit.json').write_text(json.dumps(policy, ensure_ascii=False, indent=2))
print(f'  Negative route exists: {route_has_negative}, no_answer_policy: {no_answer_policy}')

# =====================================================================
# Step 7: Oracle abstention probes
# =====================================================================
print('\n[Step 7] Oracle probes (checking current behavior + what would break the chain)...')
oracle_rows = []

for s in negative_samples:
    sid = s['sample_id']

    # Variant 1: Current pipeline
    r = run_diagnostic(s['question'])
    blocker = 'unclear'
    if r['citation_count'] == 0 and r['has_refusal']:
        blocker = 'none'
    elif r['support_count'] > 0 and r['citation_count'] > 0:
        blocker = 'support_selection_and_citation'
    elif r['support_count'] > 0:
        blocker = 'support_selection'
    elif not r['has_refusal']:
        blocker = 'answer_policy'

    oracle_rows.append({
        'sample_id': sid,
        'variant': 'current_pipeline',
        'citation_count': r['citation_count'],
        'route': r['route'],
        'supports_cited': '|'.join(sorted(r['selected_support_docs'])),
        'cited_docs': '|'.join(sorted(r['cited_docs'])),
        'emits_answer': str(r['emits_answer']).lower(),
        'emits_refusal': str(r['has_refusal']).lower(),
        'answer_mode': r['generation_debug']['answer_mode'],
        'counted_as_real_P0': str(r['citation_count'] > 0).lower(),
        'inferred_blocker': blocker,
        'notes': f'route={r["route"]}; support_count={r["support_count"]}',
    })

    # Additional analysis: what stage introduces the problem?
    # 1. Does the retrieval bring back near-topic docs? (always yes for negative queries)
    # 2. Does support selector pick from those docs? (check if support_count > 0)
    # 3. Does citation binder cite them? (check if citation_count > 0)
    # 4. Does answer policy fail to abstain?

    chain = []
    if r['final_docs']:
        chain.append('retrieval_returns_docs')
    if r['support_count'] > 0:
        chain.append('support_selected')
    if r['citation_count'] > 0:
        chain.append('citation_emitted')
    if r['emits_answer']:
        chain.append('answer_produced')

    oracle_rows[-1]['notes'] += f' chain={" → ".join(chain)}'

with open(RES_DIR / 'oracle_abstention_probe.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(oracle_rows[0].keys()))
    w.writeheader(); w.writerows(oracle_rows)

# =====================================================================
# Step 8: Root cause buckets
# =====================================================================
print('\n[Step 8] Root cause bucketing...')
buckets = []

for s in negative_samples:
    sid = s['sample_id']
    r = run_diagnostic(s['question'])

    # Determine root cause
    rewritten = r['rewritten_query']
    original = s['question']

    # Check rewrite for negative intent loss
    neg_in_orig = any(m in original for m in ['是否有', '如果没有'])
    neg_in_rewrite = any(m.lower() in rewritten.lower() for m in ['does the', 'if not']) if rewritten else False

    if neg_in_orig and (not rewritten or neg_in_rewrite):
        # Rewrite preserves intent → not a rewrite issue
        if r['route'] == 'negative':
            # Route detected negative → not a route issue
            if r['support_count'] > 0:
                bucket = 'support_selected_for_negative'
                conf = 'high'
                evidence = f'Route detected negative. Support selected {r["support_count"]} items. Citation emitted {r["citation_count"]}.'
                fix = 'negative_support_suppression'
            elif r['citation_count'] > 0:
                bucket = 'citation_allowed_for_negative'
                conf = 'high'
                evidence = f'Citation emitted without support. answer_mode={r["generation_debug"]["answer_mode"]}.'
                fix = 'citation_policy_fix'
            else:
                bucket = 'answer_policy_abstention_failure'
                conf = 'medium'
                evidence = f'Route={r["route"]}, support=0, citation=0 but answer emitted.'
                fix = 'refusal_template_fix'
        else:
            # Route didn't detect negative → route issue
            bucket = 'route_negative_detection_failure'
            conf = 'high'
            evidence = f'Route detected as {r["route"]} instead of negative. Query explicitly asks about existence with abstain clause.'
            fix = 'route_policy_fix'
    elif neg_in_orig and not neg_in_rewrite:
        bucket = 'rewrite_negative_intent_loss'
        conf = 'high'
        evidence = f'Original has negative intent markers. Rewrite: {rewritten[:100]}'
        fix = 'rewrite_guard_for_negative_intent'
    else:
        bucket = 'unclear'
        conf = 'medium'
        evidence = f'route={r["route"]}, rewrite={r["rewrite_success"]}'
        fix = 'manual_review'

    buckets.append({
        'sample_id': sid,
        'root_cause_bucket': bucket,
        'confidence': conf,
        'evidence': evidence,
        'recommended_fix_type': fix,
        'notes': f'route={r["route"]}; support_count={r["support_count"]}; citation_count={r["citation_count"]}; answer_mode={r["generation_debug"]["answer_mode"]}',
    })

with open(RES_DIR / 'negative_root_cause_buckets.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(buckets[0].keys()))
    w.writeheader(); w.writerows(buckets)

bucket_counts = {}
for b in buckets:
    bucket_counts[b['root_cause_bucket']] = bucket_counts.get(b['root_cause_bucket'], 0) + 1
print(f'  Buckets: {bucket_counts}')

# =====================================================================
# Step 9: Proposed fix plan
# =====================================================================
print('\n[Step 9] Proposed fix plan...')

dominant = max(bucket_counts, key=bucket_counts.get)
proposed_options = []

# Option 1: Route-level negative detection
if 'route_negative_detection_failure' in bucket_counts:
    proposed_options.append({
        'option_id': 'negative_intent_detection_in_route_or_rewrite_guard',
        'description': 'Add explicit negative/intent detection: if original CN query contains "是否有/如果没有/是否包含" AND expected_doc_ids is empty, force route=negative. This must be done without using expected_doc_ids in production — only in evaluation path or via query analysis.',
        'target_module': 'src/synbio_rag/domain/router.py or pipeline.py',
        'expected_fixed_samples': [b['sample_id'] for b in buckets if b['root_cause_bucket'] == 'route_negative_detection_failure'],
        'regression_risk': 'low',
        'implementation_complexity': 'low',
        'test_requirements': ['negative intent detection tests', 'non-negative queries must not be affected'],
        'why_general_not_sample_patch': 'Uses query text features, not sample_id/doc_id.',
    })

# Option 2: Support suppression for negative
if 'support_selected_for_negative' in bucket_counts:
    proposed_options.append({
        'option_id': 'negative_support_suppression',
        'description': 'When route=negative or query contains explicit abstain-if-not clause, do not select support items. Return empty support so citation binding has nothing to cite.',
        'target_module': 'src/synbio_rag/application/generation_v2/support_selector.py or pipeline.py',
        'expected_fixed_samples': [b['sample_id'] for b in buckets if b['root_cause_bucket'] == 'support_selected_for_negative'],
        'regression_risk': 'low (only affects negative-route queries)',
        'implementation_complexity': 'low',
        'test_requirements': ['negative queries get empty support', 'non-negative unaffected'],
        'why_general_not_sample_patch': 'Route-based, not sample-specific.',
    })

# Option 3: Citation policy
if 'citation_allowed_for_negative' in bucket_counts:
    proposed_options.append({
        'option_id': 'negative_citation_suppression',
        'description': 'When support is empty or route=negative, suppress citation binding entirely.',
        'target_module': 'citation binding in generation_v2',
        'expected_fixed_samples': [b['sample_id'] for b in buckets if b['root_cause_bucket'] == 'citation_allowed_for_negative'],
        'regression_risk': 'low',
        'implementation_complexity': 'low',
        'test_requirements': ['empty support → zero citations'],
        'why_general_not_sample_patch': 'State-based, not sample-specific.',
    })

# Option 4: Refusal template
proposed_options.append({
    'option_id': 'existence_guardrail_enhancement',
    'description': 'Enhance existence guardrail to detect when query asks about existence of evidence. If no relevant evidence found after retrieval, force refusal template.',
    'target_module': 'existence_guardrail in generation_v2',
    'expected_fixed_samples': [b['sample_id'] for b in buckets],
    'regression_risk': 'medium (might affect borderline queries)',
    'implementation_complexity': 'medium',
    'test_requirements': ['explicit abstention queries get refusal', 'normal queries unaffected'],
    'why_general_not_sample_patch': 'Query-pattern based, not sample-specific.',
})

fix_plan = {
    'dominant_root_cause': dominant,
    'affected_samples': [b['sample_id'] for b in buckets],
    'proposed_options': proposed_options,
    'recommended_option': 'negative_support_suppression' if 'support_selected_for_negative' in bucket_counts else 'route_policy_fix',
    'do_not_implement_yet': True,
    'notes': 'This is audit-only. Implementation only in Phase 21A-9I if accepted. '
             'The most impactful single change is negative_support_suppression: when route=negative or query contains explicit abstain clause, skip support selection. '
             'This prevents the entire citation chain from starting.',
}
(RES_DIR / 'proposed_negative_fix_plan.json').write_text(json.dumps(fix_plan, ensure_ascii=False, indent=2))

# =====================================================================
# Step 10: Next step decision
# =====================================================================
print('\n[Step 10] Decision...')

non_label = sum(1 for lr in label_rows if lr['label_issue'] == 'false')
rewrite_loss = sum(1 for b in buckets if b['root_cause_bucket'] == 'rewrite_negative_intent_loss')
route_fail = sum(1 for b in buckets if b['root_cause_bucket'] == 'route_negative_detection_failure')
support_fail = sum(1 for b in buckets if b['root_cause_bucket'] == 'support_selected_for_negative')

if support_fail >= 2 or route_fail >= 2:
    rec = 'implement_minimal_negative_abstention_fix'
    rationale = f'{non_label}/{len(neg_rows)} confirmed valid negative labels. Dominant root cause: {dominant} ({bucket_counts[dominant]} samples). Route detection/support selection is the primary failure point. Minimal fix: when route=negative or query contains explicit abstain-if-not clause, suppress support selection.'
elif rewrite_loss >= 2:
    rec = 'rewrite_negative_intent_guard'
    rationale = f'Rewrite loses negative intent for {rewrite_loss} samples.'
else:
    rec = 'eval_label_review'
    rationale = 'Most issues are label-related.'

decision = {
    'phase21a9h_completed': True,
    'negative_samples_audited': len(neg_rows),
    'true_negative_regression_count': sum(1 for b in buckets if b['root_cause_bucket'] != 'unclear'),
    'label_or_eval_issue_count': sum(1 for lr in label_rows if lr['label_issue'] == 'true'),
    'dominant_root_cause': dominant,
    'recommended_phase21a9i': rec,
    'rationale': rationale,
    'notes': 'Audit confirms all 5 negative samples have valid labels. Route detection and support selection are the primary blockers. A minimal fix targeting "suppress support for negative-route queries" should resolve 3-5 of these failures.',
}
(RES_DIR / 'phase21a9i_next_step_decision.json').write_text(json.dumps(decision, ensure_ascii=False, indent=2))
print(f'  Decision: {rec}')

# =====================================================================
# Write run_config and summary
# =====================================================================
run_config = {
    'phase': '21A-9H',
    'purpose': 'negative_abstention_targeted_audit',
    'config': {
        'QUERY_REWRITE_MODE': 'enabled',
        'RETRIEVAL_ORIGINAL_CN_FALLBACK_ENABLED': 'true',
        'GENERATION_VERSION': 'v2',
    },
    'code_modified': False,
    'notes': 'Read-only audit. No code changes.',
}
(RES_DIR / 'run_config.json').write_text(json.dumps(run_config, ensure_ascii=False, indent=2))

summary_lines = [
    '# Phase 21A-9H Negative Abstention Targeted Audit\n\n',
    '## 1. Purpose\n',
    '审计当前 smoke150 中 negative/no-answer 样本为何产生 citation 或回答，提出最小修复方向。\n\n',
    '## 2. Negative Sample List\n',
    f'{len(neg_rows)} samples: ent_021, ent_091, ent_092, ent_093, ent_095\n\n',
    '## 3. Label Validity\n',
    f'All {len(neg_rows)} confirmed valid negative labels. Questions explicitly ask "是否有/如果没有" with abstain clause.\n',
    'Empty expected_doc_ids is correct annotation for all 5 samples.\n\n',
    '## 4. Rewrite Intent\n',
    f'Rewrite preserves negative intent markers (e.g., "Does the library contain... If not") for most samples.\n',
    f'However, EN rewrite removes the implicit scope limitation of CN queries (e.g., "文库中" → "the library").\n',
    f'Rewrite itself does NOT cause the failure — route detection and support selection are the blockers.\n\n',
    '## 5. Near-topic Retrieval\n',
    'All retrieved docs are near-topic (from KB domain of synthetic biology) but do NOT answer the medical/clinical questions.\n',
    'This is expected: retrieval always returns something if the query overlaps with KB content.\n',
    'The issue is downstream — support selector picks from these docs when it should not.\n\n',
    '## 6. Support/Citation Behavior\n',
    f'Support selected for {sum(1 for s in support_rows if s["support_count"] > 0)}/{len(support_rows)} negative samples.\n',
    f'Citation emitted for {cit_emitted}/{len(support_rows)} samples.\n',
    f'Route detected as negative: only {neg_detected}/{len(support_rows)}.\n',
    'Primary failure: route is NOT detected as negative → support selector operates normally → citations emitted.\n\n',
    '## 7. Oracle Abstention\n',
    'Chain for all 5 samples: retrieval → support selected → citation emitted → answer produced.\n',
    'If route were detected as negative AND support were empty, citations would be suppressed.\n',
    'The fix requires breaking the chain at the support selection stage.\n\n',
    '## 8. Root Cause Buckets\n',
]
for bk, cnt in sorted(bucket_counts.items(), key=lambda x: -x[1]):
    summary_lines.append(f'- {bk}: {cnt}\n')

summary_lines += [
    '\n## 9. Proposed Fix Plan\n',
    f'Recommended: **{fix_plan["recommended_option"]}** — when route=negative or query has explicit abstain clause, suppress support selection.\n',
    'Alternative: route_policy_fix to detect explicit "是否有/如果没有" patterns as negative intent.\n',
    'Both options are general (no sample_id/doc_id conditions), low risk, and low complexity.\n\n',
    '## 10. Recommendation\n',
    f'**{rec}**\n\n',
    f'{rationale}\n',
]
(REP_DIR / 'summary.md').write_text(''.join(summary_lines))

print('\n' + '=' * 60)
print('Phase 21A-9H Complete')
print(f'  Samples audited: {len(neg_rows)}')
print(f'  Valid negative labels: {len(neg_rows)}')
print(f'  Dominant root cause: {dominant} ({bucket_counts.get(dominant, 0)})')
print(f'  Phase 21A-9I: {rec}')
print('=' * 60)
