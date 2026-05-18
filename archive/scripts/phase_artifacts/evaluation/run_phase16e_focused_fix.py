#!/usr/bin/env python3
"""Phase 16E focused fix validation."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.synbio_rag.application.pipeline import SynBioRAGPipeline
from src.synbio_rag.domain.config import Settings
from src.synbio_rag.domain.schemas import QueryFilters

data = json.loads((ROOT / "data/eval/datasets/enterprise_ragas_smoke100.json").read_text())
by_id = {s.get("id",""): s for s in data if isinstance(s, dict)}
focused = [
    "ent_013","ent_040","ent_066","ent_077","ent_074","ent_086",
    "ent_005","ent_011","ent_055","ent_060","ent_100",
    "ent_020","ent_037",
]

settings = Settings.from_env()
settings.generation.version = "v2"
settings.generation.v2_use_qwen_synthesis = False
settings.generation.v2_enable_comparison_coverage = False
settings.generation.v2_enable_neighbor_audit = False
settings.generation.v2_enable_neighbor_promotion = False
settings.retrieval.parent_expansion_enabled = True

pipeline = SynBioRAGPipeline(settings)
results = []

for sid in focused:
    s = by_id.get(sid)
    if not s:
        continue
    resp = pipeline.answer(s.get("question",""), filters=QueryFilters())
    gv2 = (resp.debug or {}).get("generation_v2", {})
    sp = gv2.get("support_pack", []) or []
    mode = gv2.get("answer_mode", "?")
    cb = gv2.get("support_selection_debug", {}).get("citation_binding", {})
    ordered = cb.get("ordered_evidence_ids", [])
    uncited = cb.get("uncited_selected_support_evidence_ids", [])
    expected = s.get("doc_ids", [])
    cited_docs = [item.get("doc_id","") for item in sp if item.get("evidence_id","") in set(ordered)]
    exp_in_cited = any(d in cited_docs for d in expected)
    results.append({
        "sid": sid, "support": len(sp), "ordered": len(ordered),
        "uncited": len(uncited), "mode": mode,
        "exp_in_cited": exp_in_cited,
        "ordered_eids": "|".join(ordered),
        "uncited_eids": "|".join(uncited),
    })

with open("/tmp/p16e_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Done. Results written to /tmp/p16e_results.json")
for r in results:
    print(f"{r['sid']}: support={r['support']} ordered={r['ordered']} uncited={r['uncited']} mode={r['mode']} exp_in_cited={r['exp_in_cited']}")
