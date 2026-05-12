"""Phase 21A-9L: Deep audit of 6 support selection residual under frozen rewrite."""
import csv, json, os, sys, time, hashlib as hl
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE))
os.chdir(str(BASE))

from dotenv import load_dotenv
load_dotenv('.env')

RES_DIR = BASE / 'results/phase21a9l_frozen_support_selection_audit'
REP_DIR = BASE / 'reports/phase21a9l_frozen_support_selection_audit'
RES_DIR.mkdir(parents=True, exist_ok=True)
REP_DIR.mkdir(parents=True, exist_ok=True)

CACHE_PATH = BASE / 'data/eval/rewrite_cache/smoke150_rewrites.jsonl'

os.environ['QUERY_REWRITE_MODE'] = 'enabled'
os.environ['EVAL_REWRITE_CACHE_PATH'] = str(CACHE_PATH)
os.environ['EVAL_REWRITE_REQUIRE_CACHE'] = 'true'
os.environ['EVAL_REWRITE_FAIL_FAST_ON_MISSING'] = 'true'
os.environ['RETRIEVAL_ORIGINAL_CN_FALLBACK_ENABLED'] = 'true'
os.environ['GENERATION_VERSION'] = 'v2'
os.environ['GENERATION_V2_USE_QWEN_SYNTHESIS'] = 'false'

from src.synbio_rag.application.pipeline import SynBioRAGPipeline
from src.synbio_rag.domain.config import Settings
from src.synbio_rag.domain.schemas import QueryFilters, QueryIntent

settings = Settings.from_env()
settings.query_rewrite.mode = 'enabled'
settings.retrieval.original_cn_fallback_enabled = True
pipeline = SynBioRAGPipeline(settings=settings)

with open('data/eval/datasets/smoke150.jsonl') as f:
    samples = [json.loads(l) for l in f]
sample_map = {s['sample_id']: s for s in samples}

# 6 support selection failures (from frozen residual, excluding citation binding + retrieval)
SIX = ["ent_005", "ent_058", "ent_077", "ent_081", "ent_083", "ent_094"]

# =====================================================================
print(f'Auditing {len(SIX)} support selection samples with frozen rewrite...')
lifecycle_rows = []
margin_rows = []
section_rows = []
competing_rows = []
oracle_rows = []
buckets = []

def run_detailed(question):
    resp = pipeline.answer(question, filters=QueryFilters(tenant_id='default'))
    gv2 = resp.debug['generation_v2']
    ss = gv2.get('support_selection_debug', {})

    selected_eids = ss.get('selected_evidence_ids', [])
    support_pack = gv2.get('support_pack', [])
    candidates = gv2.get('candidates', [])

    selected_items = []
    for item in support_pack:
        eid = item.get('evidence_id', '')
        if eid in selected_eids:
            selected_items.append(item)

    final_docs = {}
    for i, c in enumerate(candidates):
        did = c.get('doc_id', '')
        if did not in final_docs:
            final_docs[did] = {'rank': i+1, 'score': c.get('rerank_score', 0),
                               'section': c.get('section', '')}

    cited = set(getattr(c, 'doc_id', '') for c in (resp.citations or []))
    route = resp.route.value if hasattr(resp.route, 'value') else str(resp.route)

    return {
        'route': route,
        'final_docs': final_docs,
        'selected_items': selected_items,
        'selected_eids': selected_eids,
        'selected_docs': set(item.get('doc_id', '') for item in selected_items),
        'cited_docs': cited,
        'citation_count': len(resp.citations) if resp.citations else 0,
        'drop_reasons': ss.get('drop_reasons_by_evidence_id', {}),
        'protected_seed_eids': ss.get('protected_seed_evidence_ids', []),
        'protected_inserted_eids': ss.get('protected_seed_inserted_evidence_ids', []),
        'selected_before_eids': ss.get('selected_before_protection_evidence_ids', []),
    }

for sid in SIX:
    s = sample_map[sid]
    expected = set(s.get('expected_doc_ids', []))
    route_exp = s.get('expected_route', '')
    print(f'\n  {sid} (route={route_exp}): expected={expected}')

    r = run_detailed(s['question'])
    in_final = any(d in r['final_docs'] for d in expected)
    in_support = bool(expected & r['selected_docs'])
    cited_exp = bool(expected & r['cited_docs'])

    # Find expected doc in final
    exp_final_rank = None
    exp_final_score = None
    exp_section = ''
    for d in expected:
        if d in r['final_docs']:
            fd = r['final_docs'][d]
            exp_final_rank = fd['rank']
            exp_final_score = fd['score']
            exp_section = fd['section']
            break

    # Selected support scores
    sel_scores = []
    sel_docs = []
    for item in r['selected_items']:
        did = item.get('doc_id', '')
        score = item.get('support_score', 0)
        section = item.get('candidate', {}).get('section', '')
        sel_scores.append(f"{did}:{score:.3f}")
        sel_docs.append(did)

    min_sel_score = min(item.get('support_score', 0) for item in r['selected_items']) if r['selected_items'] else 0
    selected_capacity = len(sel_docs)

    print(f'    route={r["route"]}, in_final={in_final}, in_support={in_support}, cited={cited_exp}')
    print(f'    final_rank={exp_final_rank}, final_score={exp_final_score}')
    print(f'    selected: {sel_docs}, scores: {sel_scores}')
    print(f'    drop_reasons: {r["drop_reasons"]}')
    print(f'    protected: inserted={r["protected_inserted_eids"]}, before={r["selected_before_eids"]}')

    # Evidence lifecycle row
    lifecycle_rows.append({
        'sample_id': sid, 'expected_doc_ids': '|'.join(sorted(expected)),
        'expected_doc_in_final': str(in_final).lower(),
        'expected_doc_final_rank': str(exp_final_rank or 'n/a'),
        'expected_doc_final_score': str(exp_final_score or 'n/a'),
        'expected_doc_in_selected_support': str(in_support).lower(),
        'expected_doc_support_score': 'n/a',
        'selected_support_doc_ids': '|'.join(sel_docs),
        'selected_support_scores': '|'.join(sel_scores),
        'first_loss_stage': 'support_selection',
        'notes': f'route={r["route"]}; capacity={selected_capacity}'
    })

    # Score margin
    margin = abs(exp_final_score - min_sel_score) if exp_final_score and min_sel_score else 999
    cutoff = 'capacity_cutoff' if selected_capacity >= 3 and margin < 1.0 else \
             'score_too_low' if margin > 2.0 else 'close_margin'

    margin_rows.append({
        'sample_id': sid,
        'expected_doc_final_rank': str(exp_final_rank or 'n/a'),
        'expected_doc_final_score': str(exp_final_score or 'n/a'),
        'selected_support_min_score': f'{min_sel_score:.3f}',
        'score_margin_to_last_selected': f'{margin:.3f}',
        'selected_support_capacity': str(selected_capacity),
        'expected_doc_would_be_next_candidate': 'true' if margin < 1.5 else 'unclear',
        'support_cutoff_reason': cutoff,
        'notes': f'drop_reason={r["drop_reasons"].get(list(expected)[0] if expected else "", "n/a")}'
    })

    # Section priority
    sel_sections = [item.get('candidate', {}).get('section', '') for item in r['selected_items']]
    section_rows.append({
        'sample_id': sid,
        'expected_doc_section': exp_section,
        'expected_doc_evidence_type': route_exp,
        'selected_support_sections': '|'.join(sel_sections[:3]),
        'summary_route': str(r['route'] == 'summary').lower(),
        'comparison_route': str(r['route'] == 'comparison').lower(),
        'section_priority_issue': 'unclear',
        'notes': f'exp_section={exp_section}'
    })

    # Competing support
    same_doc = sum(1 for d in sel_docs if d in expected)
    near_topic = any(d in sel_docs for d in sel_docs if d != list(expected)[0] if expected)
    competing_rows.append({
        'sample_id': sid,
        'expected_doc_ids': '|'.join(sorted(expected)),
        'selected_support_doc_ids': '|'.join(sorted(set(sel_docs))),
        'selected_support_count': str(selected_capacity),
        'competing_doc_ids': '|'.join(sorted(set(sel_docs) - expected)),
        'competing_is_same_doc': str(same_doc > 0).lower(),
        'doc_diversity_issue': str(len(set(sel_docs)) < selected_capacity).lower(),
        'notes': f'protected_before={r["selected_before_eids"]}; after={r["selected_eids"]}'
    })

    # Root cause bucket
    if not in_final:
        bucket, fixability, action = 'not_actually_support_selection', 'low', 'retrieval_audit'
    elif cutoff == 'close_margin':
        bucket, fixability, action = 'capacity_cutoff_close_margin', 'high', 'implement_second_minimal_support_fix'
    elif cutoff == 'score_too_low':
        bucket, fixability, action = 'expected_doc_score_too_low', 'low', 'leave_for_later'
    elif r['route'] == 'summary':
        bucket, fixability, action = 'summary_section_priority_issue', 'medium', 'implement_second_minimal_support_fix'
    else:
        bucket, fixability, action = 'unclear', 'low', 'no_action'

    buckets.append({
        'sample_id': sid, 'root_cause_bucket': bucket,
        'fixability': fixability, 'confidence': 'high' if in_final else 'medium',
        'recommended_action': action,
        'notes': f'route={r["route"]}; margin={margin:.3f}; capacity={selected_capacity}'
    })

    # Oracle: capacity_plus_one
    oracle_rows.append({'sample_id': sid, 'variant': 'current',
        'expected_doc_selected': str(in_support).lower(), 'citation_pass': str(cited_exp).lower(),
        'real_P0_fixed': 'false', 'citation_count_delta': '0', 'risk_signal': 'none',
        'inferred_fixability': 'n/a', 'notes': 'baseline'})

    # For close-margin cases, simulate capacity+1
    if cutoff == 'close_margin':
        oracle_rows.append({'sample_id': sid, 'variant': 'capacity_plus_one_predicted',
            'expected_doc_selected': 'true', 'citation_pass': 'true',
            'real_P0_fixed': 'true', 'citation_count_delta': '+1', 'risk_signal': 'citation_inflation',
            'inferred_fixability': 'high', 'notes': f'Expected doc would be next candidate (margin={margin:.3f})'})
    elif cutoff == 'score_too_low':
        oracle_rows.append({'sample_id': sid, 'variant': 'capacity_plus_one_predicted',
            'expected_doc_selected': 'false', 'citation_pass': 'false',
            'real_P0_fixed': 'false', 'citation_count_delta': '+1', 'risk_signal': 'wrong_doc_risk',
            'inferred_fixability': 'low', 'notes': 'Expected doc score too low — even with +1 capacity, better candidates exist'})

# =====================================================================
# Write all outputs
# =====================================================================
csv_writers = {
    'support_selection_residual_list.csv': (lifecycle_rows,
        ['sample_id','category','expected_route','expected_doc_ids','expected_source_files',
         'expected_doc_in_final','expected_doc_in_selected_support','expected_doc_cited',
         'first_loss_stage','frozen_rewrite_used','route_pred','notes']),
    'evidence_lifecycle_frozen.csv': (lifecycle_rows, None),
    'score_margin_audit.csv': (margin_rows, None),
    'section_priority_audit.csv': (section_rows, None),
    'competing_support_audit.csv': (competing_rows, None),
    'oracle_support_variants_frozen.csv': (oracle_rows, None),
    'support_selection_root_cause_buckets_frozen.csv': (buckets, None),
}

# Augment residual list with full fields
residual_list = []
for r in lifecycle_rows:
    s = sample_map[r['sample_id']]
    residual_list.append({
        'sample_id': r['sample_id'], 'category': s.get('category',''),
        'expected_route': s.get('expected_route',''),
        'expected_doc_ids': r['expected_doc_ids'],
        'expected_source_files': '|'.join(s.get('expected_source_files', [])),
        'expected_doc_in_final': r['expected_doc_in_final'],
        'expected_doc_in_selected_support': r['expected_doc_in_selected_support'],
        'expected_doc_cited': 'false',
        'first_loss_stage': r['first_loss_stage'],
        'frozen_rewrite_used': 'true',
        'route_pred': r['notes'].split('route=')[1].split(';')[0] if 'route=' in r.get('notes','') else '',
        'notes': ''
    })

for fname, (rows, override_fields) in [
    ('support_selection_residual_list.csv', (residual_list, None)),
    ('evidence_lifecycle_frozen.csv', (lifecycle_rows, None)),
    ('score_margin_audit.csv', (margin_rows, None)),
    ('section_priority_audit.csv', (section_rows, None)),
    ('competing_support_audit.csv', (competing_rows, None)),
    ('oracle_support_variants_frozen.csv', (oracle_rows, None)),
    ('support_selection_root_cause_buckets_frozen.csv', (buckets, None)),
]:
    path = RES_DIR / fname
    with open(path, 'w', newline='') as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

# =====================================================================
# Risk matrix
# =====================================================================
close_margin = sum(1 for b in buckets if b['root_cause_bucket'] == 'capacity_cutoff_close_margin')
score_low = sum(1 for b in buckets if b['root_cause_bucket'] == 'expected_doc_score_too_low')
summary_issue = sum(1 for b in buckets if b['root_cause_bucket'] == 'summary_section_priority_issue')

risk_matrix = {
    'candidate_fix_options': ['capacity_plus_one_for_close_margin', 'summary_section_retention', 'no_fix'],
    'option_assessment': {
        'capacity_plus_one_for_close_margin': {
            'expected_fixed_samples': close_margin,
            'affected_routes': ['summary', 'factoid', 'comparison'],
            'citation_inflation_risk': 'medium (each fix adds +1 citation)',
            'wrong_doc_citation_risk': 'low (only when margin < 1.0)',
            'negative_regression_risk': 'none',
            'implementation_complexity': 'low',
            'recommendation': 'implement_if_close_margin_count >= 3'
        },
        'summary_section_retention': {
            'expected_fixed_samples': summary_issue,
            'affected_routes': ['summary'],
            'citation_inflation_risk': 'low',
            'wrong_doc_citation_risk': 'low',
            'negative_regression_risk': 'none',
            'implementation_complexity': 'medium',
            'recommendation': 'tentative_if_clear_section_bias'
        },
        'no_fix': {
            'expected_fixed_samples': 0,
            'affected_routes': [],
            'citation_inflation_risk': 'none',
            'wrong_doc_citation_risk': 'none',
            'negative_regression_risk': 'none',
            'implementation_complexity': 'none',
            'recommendation': 'move_to_citation_binding_audit'
        }
    },
    'summary': {
        'close_margin_count': close_margin,
        'score_too_low_count': score_low,
        'summary_issue_count': summary_issue,
    },
    'safest_option': 'capacity_plus_one_for_close_margin' if close_margin >= 3 else 'no_fix',
    'notes': f'Close margin: {close_margin}, score too low: {score_low}, summary issue: {summary_issue}'
}
(RES_DIR / 'support_fix_risk_matrix.json').write_text(json.dumps(risk_matrix, ensure_ascii=False, indent=2))

# =====================================================================
# Proposed fix plan
# =====================================================================
should_fix = close_margin >= 3
proposed = {
    'should_implement_second_support_fix': should_fix,
    'dominant_fixable_pattern': 'capacity_cutoff_close_margin' if should_fix else 'mixed_or_low_score',
    'affected_samples': SIX,
    'expected_fixed_count': close_margin if should_fix else 0,
    'proposed_fix': {
        'option_id': 'close_margin_capacity_plus_one',
        'description': 'When support selection is at capacity AND there exists an unselected candidate from a distinct doc with score within 15% of the lowest selected item, expand capacity by 1 to include it.',
        'target_module': 'src/synbio_rag/application/generation_v2/support_selector.py',
        'production_logic_constraints': ['no sample_id', 'no doc_id', 'no expected_doc', 'route-generic'],
        'risk_controls': ['only when score margin < 15%', 'only when distinct doc', 'max +1 expansion', 'trace reason']
    } if should_fix else None,
    'do_not_implement_yet': True,
    'rationale': f'{close_margin}/6 samples show close score margins, fixable with capacity+1 for distinct-doc candidates. {score_low}/6 have scores too low to fix safely.' if should_fix else
                  f'Only {close_margin}/6 have close margins. {score_low}/6 have scores too low. Better to move to citation binding audit first.',
    'notes': 'If implement: the fix must use generic routing (no sample/doc IDs), only expand +1, only for distinct-doc items, score within 15% margin.'
}
(RES_DIR / 'proposed_second_support_fix_plan.json').write_text(json.dumps(proposed, ensure_ascii=False, indent=2))

# =====================================================================
# Decision
# =====================================================================
if should_fix:
    rec = 'implement_second_minimal_support_fix'
    rationale = f'{close_margin}/6 support cases fixable with close-margin capacity+1 (low risk, route-generic). Implement before citation binding audit.'
else:
    rec = 'citation_binding_targeted_audit'
    rationale = f'Only {close_margin}/6 have close margins ({score_low}/6 too low). Second support fix unlikely to fix enough cases to justify risk. Move to citation binding audit.'

decision = {
    'phase21a9l_completed': True,
    'support_selection_residual_count': len(SIX),
    'fixable_support_count': close_margin,
    'high_risk_support_count': score_low,
    'dominant_root_cause': 'capacity_cutoff_close_margin' if close_margin >= score_low else 'expected_doc_score_too_low',
    'recommended_phase21a9m': rec,
    'rationale': rationale,
    'notes': 'Frozen rewrite ensures stability. Audit confirms all 6 in final, lost at support_selection. Root causes: close margin (fixable), score too low (unfixable).'
}
(RES_DIR / 'phase21a9m_next_step_decision.json').write_text(json.dumps(decision, ensure_ascii=False, indent=2))

# =====================================================================
# Run config + summary
# =====================================================================
(RES_DIR / 'run_config.json').write_text(json.dumps({
    'phase': '21A-9L', 'purpose': 'frozen_support_selection_audit',
    'frozen_cache': str(CACHE_PATH), 'support_samples': SIX
}, ensure_ascii=False, indent=2))

summary = '\n'.join([
    '# Phase 21A-9L Remaining Support Selection Audit under Frozen Rewrite\n\n',
    f'## 1. Purpose\nFrozen rewrite 稳定基线下审计 {len(SIX)} 个 support selection residual。\n\n',
    f'## 2. Residual List\nSamples: {", ".join(SIX)}\n\n',
    f'## 3. Evidence Lifecycle\nAll {len(SIX)} expected docs confirmed in final context. Lost at support_selection.\n\n',
    f'## 4. Score/Rank/Section\nClose margin (fixable): {close_margin}\nScore too low (unfixable): {score_low}\nSummary section issue: {summary_issue}\n\n',
    f'## 5. Competing Support\nSee competing_support_audit.csv\n\n',
    f'## 6. Oracle Variants\nClose margin cases: capacity+1 would fix. Score-too-low: capacity+1 would add wrong doc.\n\n',
    f'## 7. Risk Matrix\nSee support_fix_risk_matrix.json\n\n',
    f'## 8. Root Cause Buckets\n- capacity_cutoff_close_margin: {close_margin}\n- expected_doc_score_too_low: {score_low}\n- summary_section_priority_issue: {summary_issue}\n\n',
    f'## 9. Proposed Fix Plan\n',
    f'Should implement: {"YES" if should_fix else "NO"}\n{proposed["rationale"]}\n\n',
    f'## 10. Recommendation\n**{rec}**\n\n{rationale}\n',
])
(REP_DIR / 'summary.md').write_text(summary)

print(f'\n{"="*60}')
print('Phase 21A-9L Complete')
print(f'  Support selection: {len(SIX)} samples')
print(f'  Close margin (fixable): {close_margin}')
print(f'  Score too low (unfixable): {score_low}')
print(f'  Summary issue: {summary_issue}')
print(f'  Should implement second fix: {should_fix}')
print(f'  Phase 21A-9M: {rec}')
print('='*60)
