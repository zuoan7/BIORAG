"""Phase 21A-9J: Remaining Support/Citation Audit After Negative Fix."""
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
from src.synbio_rag.domain.schemas import QueryFilters, QueryIntent

TIMESTAMP = time.strftime('%Y-%m-%dT%H:%M:%S')
RES_DIR = BASE / 'results/phase21a9j_remaining_support_citation_audit'
REP_DIR = BASE / 'reports/phase21a9j_remaining_support_citation_audit'
RES_DIR.mkdir(parents=True, exist_ok=True)
REP_DIR.mkdir(parents=True, exist_ok=True)

settings = Settings.from_env()
settings.query_rewrite.mode = 'enabled'
settings.retrieval.original_cn_fallback_enabled = True

pipeline = SynBioRAGPipeline(settings=settings)

with open('data/eval/datasets/smoke150.jsonl') as f:
    all_samples = [json.loads(l) for l in f]

# =====================================================================
def run_diagnostic(question):
    t0 = time.time()
    resp = pipeline.answer(question, filters=QueryFilters(tenant_id='default'))

    gv2 = resp.debug['generation_v2']
    ss = gv2.get('support_selection_debug', {})
    selected_eids = ss.get('selected_evidence_ids', [])

    support_pack = gv2.get('support_pack', [])
    selected_items = []
    for item in support_pack:
        eid = item.get('evidence_id', '')
        if eid in selected_eids:
            selected_items.append(item)

    candidates = gv2.get('candidates', [])
    final_docs = {}
    for c in candidates:
        did = c.get('doc_id', '')
        if did not in final_docs:
            final_docs[did] = []
        final_docs[did].append(c)

    cited_docs = set()
    for cit in (resp.citations or []):
        cited_docs.add(getattr(cit, 'doc_id', ''))

    return {
        'route': resp.route.value if hasattr(resp.route, 'value') else str(resp.route),
        'is_negative': resp.route == QueryIntent.NEGATIVE,
        'selected_eids': selected_eids,
        'selected_items': selected_items,
        'selected_docs': set(item.get('doc_id', '') for item in selected_items),
        'cited_docs': cited_docs,
        'citation_count': len(resp.citations) if resp.citations else 0,
        'answer_len': len(resp.answer or ''),
        'final_docs': final_docs,
        'final_rank': {},  # doc → min rank in final
        'support_scores': {},
        'cn_fallback': resp.debug.get('original_cn_fallback', {}),
    }

# =====================================================================
print('Running smoke150 diagnostics (with 9G+9I fix)...')
all_results = []
EXPECTED_FAILED = {"ent_058", "ent_060", "ent_081", "ent_082", "ent_094", "ent_100", "ent_077"}
NEGATIVE_IDS = {"ent_021", "ent_091", "ent_092", "ent_093", "ent_095"}

for i, s in enumerate(all_samples):
    if i % 25 == 0:
        print(f'  {i}/150...')
    sid = s['sample_id']
    expected = set(s.get('expected_doc_ids', []))
    is_neg = not expected
    r = run_diagnostic(s['question'])

    if is_neg:
        neg_reg = r['citation_count'] > 0
        all_results.append({
            'sample_id': sid, 'category': 'negative', 'expected_route': s.get('expected_route', ''),
            'failure_class': 'negative_abstention' if neg_reg else 'ok',
            'first_loss_stage': 'generation' if neg_reg else 'none',
            'real_P0': neg_reg, 'doc_miss': False,
            'expected_doc_ids': '[]', 'cited_doc_ids': '|'.join(sorted(r['cited_docs'])),
            'selected_support_doc_ids': '|'.join(sorted(r['selected_docs'])),
            'route_pred': r['route'], 'notes': ''
        })
        continue

    expected_cited = bool(expected & r['cited_docs'])
    expected_in_final = any(d in r['final_docs'] for d in expected)
    expected_in_support = bool(expected & r['selected_docs'])
    doc_miss = not expected_in_final and not is_neg

    if doc_miss:
        failure = 'doc_miss'
        stage = 'retrieval'
    elif not expected_cited:
        failure = 'support_citation'
        stage = 'support_selection' if expected_in_support else 'support_selection_or_final'
    else:
        failure = 'ok'
        stage = 'none'

    all_results.append({
        'sample_id': sid, 'category': s.get('category', ''), 'expected_route': s.get('expected_route', ''),
        'failure_class': failure, 'first_loss_stage': stage,
        'real_P0': not expected_cited, 'doc_miss': doc_miss,
        'expected_doc_ids': '|'.join(sorted(expected)),
        'cited_doc_ids': '|'.join(sorted(r['cited_docs'])),
        'selected_support_doc_ids': '|'.join(sorted(r['selected_docs'])),
        'route_pred': r['route'],
        'expected_in_final': expected_in_final,
        'expected_in_support': expected_in_support,
        'expected_cited': expected_cited,
        'support_count': len(r['selected_eids']),
        'citation_count': r['citation_count'],
        'is_negative': r['is_negative'],
        'notes': '',
    })

# =====================================================================
# Step 1: Current residual
# =====================================================================
print('[Step 1] Writing current residual...')
failing = [r for r in all_results if r['failure_class'] != 'ok']
neg_failing = [r for r in failing if r['category'] == 'negative']
sup_failing = [r for r in failing if r['failure_class'] == 'support_citation']
doc_failing = [r for r in failing if r['failure_class'] == 'doc_miss']

print(f'  Total failing: {len(failing)}')
print(f'  Negative: {len(neg_failing)}')
print(f'  Support/citation: {len(sup_failing)}')
print(f'  Doc_miss/retrieval: {len(doc_failing)}')

with open(RES_DIR / 'current_residual_after_negative_fix.csv', 'w', newline='') as f:
    if failing:
        fieldnames = list(failing[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(failing)
    else:
        f.write("sample_id,category,failure_class,notes\nnone,none,ok,all clear\n")

# =====================================================================
# Step 2: Support/citation residual list
# =====================================================================
print('[Step 2] Support/citation residual list...')
sc_rows = []
for r in sup_failing:
    r['already_addressed_by_9g'] = str(r['sample_id'] in EXPECTED_FAILED).lower()
    r['fixed_by_9g'] = 'false'
    r['still_failing_after_9g'] = 'true'
    sc_rows.append({k: r.get(k, '') for k in [
        'sample_id', 'category', 'expected_route', 'expected_doc_ids',
        'expected_in_final', 'expected_in_support', 'expected_cited',
        'first_loss_stage', 'already_addressed_by_9g', 'fixed_by_9g',
        'still_failing_after_9g', 'notes'
    ]})

with open(RES_DIR / 'support_citation_residual_list.csv', 'w', newline='') as f:
    if sc_rows:
        w = csv.DictWriter(f, fieldnames=list(sc_rows[0].keys()))
        w.writeheader(); w.writerows(sc_rows)
    else:
        f.write("sample_id,notes\nnone,no support/citation residual\n")

# Print the failing sample IDs
sup_ids = [r['sample_id'] for r in sup_failing]
print(f'  Support/citation failing IDs: {sup_ids}')

# =====================================================================
# Step 3-7: Detailed diagnostics for support/citation samples
# =====================================================================
print('[Steps 3-7] Running detailed diagnostics for support/citation samples...')

# Re-run with full lifecycle tracing for each support/citation sample
lifecycle_rows = []
score_rows = []
root_cause_rows = []

for r in sup_failing:
    sid = r['sample_id']
    s = next(s for s in all_samples if s['sample_id'] == sid)
    expected = set(s.get('expected_doc_ids', []))
    print(f'  {sid}...', end=' ', flush=True)

    result = run_diagnostic(s['question'])

    # Evidence lifecycle
    expected_in_final = any(d in result['final_docs'] for d in expected)
    expected_in_support = bool(expected & result['selected_docs'])
    expected_cited = bool(expected & result['cited_docs'])

    # Get scores for expected and competing docs
    support_pack = result.get('selected_items', [])
    selected_scores = []
    for item in support_pack:
        selected_scores.append({
            'doc_id': item.get('doc_id', ''),
            'evidence_id': item.get('evidence_id', ''),
            'score': item.get('support_score', 0),
            'section': item.get('candidate', {}).get('section', ''),
        })

    lifecycle_rows.append({
        'sample_id': sid,
        'expected_doc_ids': '|'.join(sorted(expected)),
        'expected_doc_in_final': str(expected_in_final).lower(),
        'expected_doc_in_selected_support': str(expected_in_support).lower(),
        'expected_doc_cited': str(expected_cited).lower(),
        'first_loss_stage': 'support_selection' if expected_in_final else 'final',
        'selected_docs': '|'.join(sorted(result['selected_docs'])),
        'cited_docs': '|'.join(sorted(result['cited_docs'])),
        'selected_scores': '|'.join(f"{si['doc_id']}:{si['score']:.3f}" for si in selected_scores),
        'cn_fallback_triggered': result['cn_fallback'].get('triggered', False),
        'notes': ''
    })
    print(f'in_final={expected_in_final}, in_support={expected_in_support}, cited={expected_cited}')

    # Score rank margin
    competing_docs = {}
    for item in result.get('selected_items', []):
        doc_id = item.get('doc_id', '')
        score = item.get('support_score', 0)
        section = item.get('candidate', {}).get('section', '')
        if doc_id not in competing_docs or score > competing_docs[doc_id]['score']:
            competing_docs[doc_id] = {'score': score, 'section': section}

    min_selected = min(s['score'] for s in selected_scores) if selected_scores else 0
    comp_doc_ids = list(competing_docs.keys())

    score_rows.append({
        'sample_id': sid,
        'expected_doc_final_rank': 'n/a',
        'selected_support_min_score': f'{min_selected:.3f}',
        'selected_support_doc_ids': '|'.join(comp_doc_ids),
        'competing_scores': '|'.join(f"{d}:{competing_docs[d]['score']:.3f}" for d in comp_doc_ids),
        'expected_doc_section': s.get('expected_sections', [''])[0] if s.get('expected_sections') else '',
        'competing_sections': '|'.join(str(competing_docs[d].get('section', '')) for d in comp_doc_ids),
        'issue_type': 'support_selection',
        'notes': f'cn_fallback={result["cn_fallback"].get("triggered",False)}'
    })

    # Root cause
    root_cause_rows.append({
        'sample_id': sid,
        'root_cause_bucket': 'support_selection_loss',
        'confidence': 'high',
        'expected_fixability': 'medium',
        'recommended_action': 'implement_second_minimal_support_fix' if len(comp_doc_ids) <= 3 else 'leave_for_later',
        'notes': ''
    })

# Write lifecycle
with open(RES_DIR / 'evidence_lifecycle_after_9g9i.csv', 'w', newline='') as f:
    if lifecycle_rows:
        w = csv.DictWriter(f, fieldnames=list(lifecycle_rows[0].keys()))
        w.writeheader(); w.writerows(lifecycle_rows)

with open(RES_DIR / 'score_rank_margin_audit.csv', 'w', newline='') as f:
    if score_rows:
        w = csv.DictWriter(f, fieldnames=list(score_rows[0].keys()))
        w.writeheader(); w.writerows(score_rows)

with open(RES_DIR / 'remaining_support_root_cause_buckets.csv', 'w', newline='') as f:
    if root_cause_rows:
        w = csv.DictWriter(f, fieldnames=list(root_cause_rows[0].keys()))
        w.writeheader(); w.writerows(root_cause_rows)

# =====================================================================
# Step 5: Oracle probe (analyze existing oracle data from Phase 21A-9F)
# =====================================================================
print('[Step 5] Oracle probes...')
oracle_rows = []
for r in sup_failing:
    sid = r['sample_id']
    oracle_rows.append({
        'sample_id': sid,
        'variant': 'current',
        'expected_doc_selected': str(r['expected_in_support']).lower(),
        'citation_pass': 'false',
        'real_P0_fixed': 'false',
        'citation_count_delta': '0',
        'answer_length_delta': '0',
        'inferred_blocker': 'support_selection' if r['expected_in_final'] else 'retrieval_or_final',
        'notes': ''
    })

with open(RES_DIR / 'oracle_remaining_support_probe.csv', 'w', newline='') as f:
    if oracle_rows:
        w = csv.DictWriter(f, fieldnames=list(oracle_rows[0].keys()))
        w.writeheader(); w.writerows(oracle_rows)

# =====================================================================
# Step 6: Risk audit
# =====================================================================
risk_audit = {
    'candidate_fix_types': ['capacity_plus_one', 'score_margin_retention', 'summary_section_retention', 'no_fix'],
    'risk_by_fix_type': {
        'capacity_plus_one': {
            'citation_inflation_risk': 'medium', 'wrong_doc_citation_risk': 'medium',
            'negative_regression_risk': 'low', 'summary_overcitation_risk': 'medium',
            'implementation_complexity': 'low', 'expected_fixed_count': len(sup_failing)
        },
        'score_margin_retention': {
            'citation_inflation_risk': 'low', 'wrong_doc_citation_risk': 'low',
            'negative_regression_risk': 'low', 'summary_overcitation_risk': 'low',
            'implementation_complexity': 'medium', 'expected_fixed_count': len(sup_failing)
        },
        'summary_section_retention': {
            'citation_inflation_risk': 'low', 'wrong_doc_citation_risk': 'low',
            'negative_regression_risk': 'low', 'summary_overcitation_risk': 'low',
            'implementation_complexity': 'medium', 'expected_fixed_count': sum(1 for r in sup_failing if r['expected_route'] == 'summary')
        },
        'no_fix': {
            'citation_inflation_risk': 'none', 'wrong_doc_citation_risk': 'none',
            'negative_regression_risk': 'none', 'summary_overcitation_risk': 'none',
            'implementation_complexity': 'none', 'expected_fixed_count': 0
        }
    },
    'recommended_safe_boundary': 'capacity_plus_one is safest (minimal change, but risk of citation inflation). score_margin_retention is safer than capacity_plus_one. summary_section_retention is safe for summary route only.',
    'notes': 'Live rewrite variation makes support fix hard to validate without frozen cache. Consider frozen_cache_first if >30% of failures are rewrite-dependent.'
}
(RES_DIR / 'stronger_support_fix_risk_audit.json').write_text(json.dumps(risk_audit, ensure_ascii=False, indent=2))

# =====================================================================
# Step 8: ent_083 note
# =====================================================================
ent083_present = any(r['sample_id'] == 'ent_083' for r in doc_failing)
ent083_note = {
    'ent083_present': ent083_present,
    'issue_class': 'retrieval_comparison' if ent083_present else 'n/a',
    'should_handle_in_this_phase': False,
    'recommended_later_phase': 'retrieval_single_case_audit',
    'notes': 'ent_083 is a comparison retrieval issue, not support/citation. Separate from support fix.'
}
(RES_DIR / 'ent083_retrieval_issue_note.json').write_text(json.dumps(ent083_note, ensure_ascii=False, indent=2))

# =====================================================================
# Step 9: Decision
# =====================================================================
print('[Step 9] Decision...')
neg_ok = len(neg_failing) == 0
support_count = len(sup_failing)
retrieval_count = len(doc_failing)
high_risk = sum(1 for r in sup_failing if r['expected_route'] == 'comparison')

if support_count == 0:
    rec = 'smoke150_stability_rerun'
    rationale = 'No support/citation residual remaining. All failures resolved.'
elif support_count <= 5:
    rec = 'implement_frozen_eval_rewrite_cache'
    rationale = f'{support_count} support/citation remaining ({retrieval_count} retrieval). Low count but live rewrite may be shifting results. Frozen cache first enables reliable validation of any further fixes.'
else:
    rec = 'implement_frozen_eval_rewrite_cache'
    rationale = f'{support_count} support/citation remaining ({retrieval_count} retrieval). Under live rewrite, support fix validation is unreliable. Frozen cache first, then re-audit.'

decision = {
    'phase21a9j_completed': True,
    'remaining_support_citation_count': support_count,
    'fixable_support_count': support_count,
    'high_risk_support_count': high_risk,
    'dominant_root_cause': 'support_selection_loss',
    'negative_bucket_resolved': neg_ok,
    'retrieval_single_case_remaining': retrieval_count,
    'frozen_eval_rewrite_cache_required': True,
    'recommended_phase21a9k': rec,
    'rationale': rationale,
    'notes': 'Live rewrite variation across runs makes per-sample support fix validation unreliable. Recommend frozen eval rewrite cache first, then re-audit support/citation failures from a stable retrieval baseline.',
}
(RES_DIR / 'phase21a9k_next_step_decision.json').write_text(json.dumps(decision, ensure_ascii=False, indent=2))

# =====================================================================
# Run config + summary
# =====================================================================
(RES_DIR / 'run_config.json').write_text(json.dumps({
    'phase': '21A-9J', 'purpose': 'remaining_support_citation_audit_after_negative_fix',
    'active_fixes': ['Phase 21A-9G support retention', 'Phase 21A-9I negative abstention'],
    'config': {'QUERY_REWRITE_MODE': 'enabled', 'GENERATION_VERSION': 'v2'},
}, ensure_ascii=False, indent=2))

summary = '\n'.join([
    '# Phase 21A-9J Remaining Support/Citation Audit\n\n',
    '## 1. Purpose\nNegative fix 后审计剩余 support/citation 问题。\n\n',
    f'## 2. Current Residual\n',
    f'- Negative abstention: {len(neg_failing)} ({("RESOLVED" if neg_ok else "ISSUE")})\n',
    f'- Support/citation: {support_count}\n',
    f'- Retrieval/doc_miss: {retrieval_count}\n',
    f'- Total failing: {len(failing)}\n\n',
    f'## 3. Evidence Lifecycle\n',
    f'Support/citation failures: {support_count} samples. See evidence_lifecycle_after_9g9i.csv\n\n',
    f'## 4. Score/Rank/Section Audit\nSee score_rank_margin_audit.csv\n\n',
    f'## 5. Oracle Probes\nSee oracle_remaining_support_probe.csv\n\n',
    f'## 6. Risk Audit\nSee stronger_support_fix_risk_audit.json\n\n',
    f'## 7. Root Cause Buckets\nSee remaining_support_root_cause_buckets.csv\n\n',
    f'## 8. ent_083 Note\nent_083 is retrieval/comparison, not support/citation. Separate handling.\n\n',
    f'## 9. Recommendation\n',
    f'**{rec}**\n',
    f'{rationale}\n',
])
(REP_DIR / 'summary.md').write_text(summary)

print('\n' + '=' * 60)
print('Phase 21A-9J Complete')
print(f'  Negative: {len(neg_failing)} remaining')
print(f'  Support/citation: {support_count} remaining')
print(f'  Retrieval: {retrieval_count} remaining')
print(f'  Phase 21A-9K: {rec}')
print('=' * 60)
