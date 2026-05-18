#!/usr/bin/env python3
"""Build Round8 diagnostic regression subsets without touching the original full sets."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "eval"
FALSE_ABSTENTION_PATH = DATA_DIR / "round8_false_abstention_regression.json"
COMPARISON_PATH = DATA_DIR / "round8_comparison_regression.json"

FALSE_ABSTENTION_SINGLE_DOC_CORE = [
    "ent_002",
    "ent_022",
    "ent_028",
    "ent_034",
    "ent_035",
    "ent_039",
    "ent_043",
    "ent_048",
    "ent_058",
    "ent_061",
    "ent_064",
]
MULTIDOC_PARTIAL_COMPARISON = [
    "ent_084",
    "ent_085",
    "ent_087",
    "ent_088",
    "ent_090",
]
RETRIEVAL_FAILURE_GUARDRAIL = [
    "ent_100",
]
ROUTE_AMBIGUITY = [
    "ent_035",
    "ent_039",
    "ent_061",
    "ent_022",
    "ent_048",
]

COMPARISON_CORE = [
    "ent_007",
    "ent_011",
    "ent_084",
    "ent_085",
    "ent_087",
    "ent_088",
    "ent_089",
    "ent_090",
]
COMPARISON_RETRIEVAL_STRESS = [
    "ent_010",
    "ent_020",
    "ent_083",
    "ent_086",
]
COMPARISON_PARTIAL_BRANCH = [
    "ent_084",
    "ent_085",
    "ent_087",
    "ent_088",
    "ent_089",
    "ent_090",
]

ACCEPTED_ROUTES = {
    "ent_035": ["factoid", "comparison"],
    "ent_039": ["factoid", "comparison"],
    "ent_061": ["factoid", "comparison"],
    "ent_022": ["summary", "comparison"],
    "ent_048": ["summary", "comparison"],
}

COMPARISON_BRANCH_LABELS = {
    "ent_007": [
        "NanR 天然调控 E. coli 唾液酸代谢",
        "NanR 被改造成 Neu5Ac 传感器",
    ],
    "ent_010": [
        "6′-SL 工程化合成路径",
        "2′-FL 工程化合成路径",
    ],
    "ent_011": [
        "染色体整合策略",
        "WcfB 改善末端转移策略",
        "salvage 路径补料生产策略",
    ],
    "ent_020": [
        "优化甲醇诱导时机和能量利用",
        "增加 AOX1 启动子调控表达盒拷贝数",
    ],
    "ent_083": [
        "E. coli 作为 NeuAc 生产宿主",
        "B. subtilis 作为 NeuAc 生产宿主",
    ],
    "ent_084": [
        "HAC1 overexpression",
        "OCH1 deletion",
    ],
    "ent_085": [
        "天然矿化调控",
        "骨桥蛋白磷酸化",
    ],
    "ent_086": [
        "2′-FL 途径工程策略一",
        "2′-FL 途径工程策略二",
        "2′-FL 途径工程策略三",
    ],
    "ent_087": [
        "Mxr1 甲醇代谢调控",
        "Mxr1 氨基酸碳源利用",
    ],
    "ent_088": [
        "GH1 糖苷酶半理性设计",
        "α-L-transfucosidase 设计目标",
    ],
    "ent_089": [
        "Epx1 前导序列",
        "突变 α 因子前导序列",
    ],
    "ent_090": [
        "2′-FL 改善骨质疏松",
        "HMO 体外发酵调节肠道菌群",
    ],
}

OUTPUTS = {
    "round8_false_abstention_single_doc_core.json": FALSE_ABSTENTION_SINGLE_DOC_CORE,
    "round8_multidoc_partial_comparison.json": MULTIDOC_PARTIAL_COMPARISON,
    "round8_retrieval_failure_guardrail.json": RETRIEVAL_FAILURE_GUARDRAIL,
    "round8_route_ambiguity.json": ROUTE_AMBIGUITY,
    "round8_comparison_core_regression.json": COMPARISON_CORE,
    "round8_comparison_retrieval_stress.json": COMPARISON_RETRIEVAL_STRESS,
    "round8_comparison_partial_branch.json": COMPARISON_PARTIAL_BRANCH,
}


def _load(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a list dataset")
    return data


def _normalize_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _sync_doc_fields(sample: dict[str, Any]) -> None:
    expected_doc_ids = _normalize_list(sample.get("expected_doc_ids") or sample.get("doc_ids"))
    expected_source_files = _normalize_list(sample.get("expected_source_files") or sample.get("source_files"))
    if sample.get("expected_route") == "comparison":
        sample["doc_ids"] = expected_doc_ids
        sample["source_files"] = expected_source_files


def _build_comparison_branches(sample: dict[str, Any]) -> list[dict[str, Any]]:
    sample_id = str(sample["id"])
    labels = COMPARISON_BRANCH_LABELS.get(sample_id)
    expected_doc_ids = _normalize_list(sample.get("expected_doc_ids") or sample.get("doc_ids"))
    expected_source_files = _normalize_list(sample.get("expected_source_files") or sample.get("source_files"))
    if not labels:
        labels = [f"branch_{idx + 1}" for idx in range(max(len(expected_doc_ids), len(expected_source_files), 1))]
    branch_count = max(len(labels), len(expected_doc_ids), len(expected_source_files))
    branches: list[dict[str, Any]] = []
    for idx in range(branch_count):
        branches.append(
            {
                "label": labels[idx] if idx < len(labels) else f"branch_{idx + 1}",
                "expected_doc_ids": [expected_doc_ids[idx]] if idx < len(expected_doc_ids) else [],
                "expected_source_files": [expected_source_files[idx]] if idx < len(expected_source_files) else [],
                "required": True,
            }
        )
    return branches


def _enrich(sample: dict[str, Any]) -> dict[str, Any]:
    enriched = deepcopy(sample)
    sample_id = str(enriched["id"])

    if sample_id in ACCEPTED_ROUTES:
        enriched["accepted_routes"] = ACCEPTED_ROUTES[sample_id]

    _sync_doc_fields(enriched)

    if enriched.get("expected_route") == "comparison":
        branches = _build_comparison_branches(enriched)
        enriched["comparison_branches"] = branches
        enriched["expected_min_doc_coverage"] = len(branches)
        enriched["allow_partial_if_doc_coverage"] = 1
        enriched["allow_partial_answer"] = True
        enriched["expected_answer_mode"] = "full_or_partial_compare"

    return enriched


def _select(samples: dict[str, dict[str, Any]], ids: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    missing = [sample_id for sample_id in ids if sample_id not in samples]
    selected = [_enrich(samples[sample_id]) for sample_id in ids if sample_id in samples]
    return selected, missing


def _duplicates(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    dupes: list[str] = []
    for sample_id in ids:
        if sample_id in seen and sample_id not in dupes:
            dupes.append(sample_id)
        seen.add(sample_id)
    return dupes


def main() -> int:
    false_abstention = _load(FALSE_ABSTENTION_PATH)
    comparison = _load(COMPARISON_PATH)
    merged = {sample["id"]: sample for sample in false_abstention + comparison}

    summary: dict[str, Any] = {"outputs": {}}
    for filename, ids in OUTPUTS.items():
        selected, missing = _select(merged, ids)
        output_path = DATA_DIR / filename
        output_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary["outputs"][filename] = {
            "sample_count": len(selected),
            "ids": [sample["id"] for sample in selected],
            "missing_ids": missing,
            "duplicate_ids": _duplicates(ids),
        }

    comparison_ids = sorted({sample_id for ids in OUTPUTS.values() for sample_id in ids if sample_id.startswith("ent_0")})
    comparison_samples = [
        _enrich(merged[sample_id])
        for sample_id in sorted(set(COMPARISON_CORE + COMPARISON_RETRIEVAL_STRESS + COMPARISON_PARTIAL_BRANCH))
    ]
    summary["checks"] = {
        "comparison_samples_all_have_branches": {
            sample["id"]: bool(sample.get("comparison_branches"))
            for sample in comparison_samples
        },
        "route_ambiguity_samples_have_accepted_routes": {
            sample_id: bool(_enrich(merged[sample_id]).get("accepted_routes"))
            for sample_id in ROUTE_AMBIGUITY
        },
        "multi_doc_comparison_doc_fields_synced": {
            sample["id"]: (
                _normalize_list(sample.get("doc_ids")) == _normalize_list(sample.get("expected_doc_ids"))
                and _normalize_list(sample.get("source_files")) == _normalize_list(sample.get("expected_source_files"))
            )
            for sample in comparison_samples
        },
        "all_declared_output_ids": comparison_ids,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
