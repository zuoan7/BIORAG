#!/usr/bin/env python3
"""
主要功能:
- 一键执行 evaluate_ragas.py 生成当前候选报告。
- 若存在基线报告，则自动对比检索指标和生成指标的前后变化。
- 可选把当前候选报告登记成新的 baseline，便于后续回归。

启动方法:
- 基本用法:
  source /home/gmy/miniconda3/bin/activate bge
  python scripts/evaluation/run_ragas_regression.py --dataset data/eval/datasets/ragas_eval_sample.jsonl

- 首次建立 baseline:
  python scripts/evaluation/run_ragas_regression.py \
    --dataset data/eval/datasets/ragas_eval_sample.jsonl \
    --init-baseline
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REGRESSION_ROOT = ROOT / "reports" / "evaluation" / "regression"


def build_run_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def normalize_label(value: str | None) -> str:
    if not value:
        return "default"
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
    return safe.strip("_") or "default"


def build_default_report_path(category: str, label: str, prefix: str) -> Path:
    return REGRESSION_ROOT / category / label / f"{prefix}_{build_run_stamp()}.json"


def canonicalize_dataset_ref(value: str) -> str:
    target = Path(value)
    if not target.is_absolute():
        target = ROOT / target
    try:
        return str(target.resolve())
    except FileNotFoundError:
        return str(target)


def report_matches_dataset(report_path: Path, dataset: str) -> bool:
    try:
        report = load_json(report_path)
    except Exception:
        return False

    report_dataset = report.get("dataset")
    if not report_dataset:
        return False
    return canonicalize_dataset_ref(str(report_dataset)) == canonicalize_dataset_ref(dataset)


def resolve_latest_report(
    path_or_dir: str,
    default_category: str | None = None,
    dataset: str | None = None,
) -> Path:
    target = Path(path_or_dir)
    if not target.is_absolute():
        target = ROOT / target
    if target.is_file():
        return target

    search_root = target
    if default_category and not search_root.exists():
        search_root = REGRESSION_ROOT / default_category
    if not search_root.exists():
        raise FileNotFoundError(f"路径不存在: {search_root}")

    candidates = sorted(search_root.rglob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"目录下没有可用报告: {search_root}")
    if dataset:
        dataset_candidates = [item for item in candidates if report_matches_dataset(item, dataset)]
        if dataset_candidates:
            return dataset_candidates[0]
        raise FileNotFoundError(
            f"目录下没有与数据集匹配的报告: {search_root} (dataset={canonicalize_dataset_ref(dataset)})"
        )
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一键执行并对比 RAGAS 回归评测")
    parser.add_argument(
        "--dataset",
        default="data/eval/datasets/enterprise_ragas_eval_v1.json",
        help="评测集路径，默认: data/eval/datasets/enterprise_ragas_eval_v1.json",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="RAG API 地址，默认: http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--judge-model",
        default="qwen-plus",
        help="ragas 裁判模型，默认: qwen-plus",
    )
    parser.add_argument(
        "--baseline",
        default="reports/evaluation/regression/baselines",
        help="基线报告文件或目录；默认自动选择 reports/evaluation/regression/baselines 下最新记录",
    )
    parser.add_argument(
        "--candidate",
        default="",
        help="当前候选报告路径；未指定时自动写入 reports/evaluation/regression/candidates/<label>/",
    )
    parser.add_argument(
        "--compare-output",
        default="",
        help="对比结果输出路径；未指定时自动写入 reports/evaluation/regression/compares/<label>/",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="HTTP 请求超时秒数，默认 180",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="最多评测多少条，0 表示全部",
    )
    parser.add_argument(
        "--skip-reference-metrics",
        action="store_true",
        help="透传给 evaluate_ragas.py",
    )
    parser.add_argument(
        "--skip-generation-metrics",
        action="store_true",
        help="透传给 evaluate_ragas.py，只输出检索对比",
    )
    parser.add_argument(
        "--include-debug",
        action="store_true",
        help="透传给 evaluate_ragas.py",
    )
    parser.add_argument(
        "--init-baseline",
        action="store_true",
        help="将当前候选报告保存为 baseline 并结束",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="完成对比后，追加生成一份新的 baseline 记录",
    )
    parser.add_argument(
        "--label",
        default="default",
        help="本次回归标签，用于分类保存 candidate / compare / baseline 记录",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def diff_section(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    all_keys = sorted(set(current) | set(baseline))
    deltas: dict[str, Any] = {}
    for key in all_keys:
        current_value = current.get(key)
        baseline_value = baseline.get(key)
        if isinstance(current_value, (int, float)) and isinstance(baseline_value, (int, float)):
            deltas[key] = {
                "baseline": baseline_value,
                "current": current_value,
                "delta": round(current_value - baseline_value, 4),
            }
        else:
            deltas[key] = {
                "baseline": baseline_value,
                "current": current_value,
            }
    return deltas


def run_evaluation(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "scripts/evaluation/evaluate_ragas.py",
        "--dataset",
        args.dataset,
        "--base-url",
        args.base_url,
        "--judge-model",
        args.judge_model,
        "--output",
        args.candidate,
        "--timeout",
        str(args.timeout),
    ]
    if args.max_samples > 0:
        command.extend(["--max-samples", str(args.max_samples)])
    if args.skip_reference_metrics:
        command.append("--skip-reference-metrics")
    if args.skip_generation_metrics:
        command.append("--skip-generation-metrics")
    if args.include_debug:
        command.append("--include-debug")
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    args = parse_args()
    label = normalize_label(args.label)
    candidate_path = Path(args.candidate) if args.candidate else build_default_report_path(
        "candidates", label, "ragas_candidate"
    )
    compare_output = Path(args.compare_output) if args.compare_output else build_default_report_path(
        "compares", label, "ragas_compare"
    )
    if not candidate_path.is_absolute():
        candidate_path = ROOT / candidate_path
    if not compare_output.is_absolute():
        compare_output = ROOT / compare_output
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    compare_output.parent.mkdir(parents=True, exist_ok=True)

    args.candidate = str(candidate_path)
    run_evaluation(args)
    if args.init_baseline:
        baseline_path = build_default_report_path("baselines", label, "ragas_baseline")
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(candidate_path, baseline_path)
        print(f"baseline 已初始化: {baseline_path}")
        return 0

    baseline_path = resolve_latest_report(args.baseline, default_category="baselines", dataset=args.dataset)
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"基线报告不存在: {baseline_path}。首次运行请加 --init-baseline。"
        )

    baseline = load_json(baseline_path)
    current = load_json(candidate_path)
    cur_abstention = current.get("abstention_analysis", {})
    base_abstention = baseline.get("abstention_analysis", {})

    cur_non_abstain_gen = cur_abstention.get("non_abstain_generation", {})
    base_non_abstain_gen = base_abstention.get("non_abstain_generation", {})

    compare_report = {
        "label": label,
        "dataset": args.dataset,
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "sample_count": current.get("sample_count"),
        "retrieval": diff_section(
            current.get("retrieval", {}).get("summary", {}),
            baseline.get("retrieval", {}).get("summary", {}),
        ),
        "generation": diff_section(
            current.get("generation", {}).get("summary", {}),
            baseline.get("generation", {}).get("summary", {}),
        ),
        "non_abstain_generation": diff_section(cur_non_abstain_gen, base_non_abstain_gen) if (cur_non_abstain_gen or base_non_abstain_gen) else {},
        "abstention": {
            "abstention_accuracy": {
                "baseline": base_abstention.get("abstention_accuracy"),
                "current": cur_abstention.get("abstention_accuracy"),
                "delta": round(
                    (cur_abstention.get("abstention_accuracy") or 0) - (base_abstention.get("abstention_accuracy") or 0), 4
                ) if cur_abstention.get("abstention_accuracy") is not None else None,
            },
            "correct_abstentions": {
                "baseline": f"{base_abstention.get('correct_abstentions', '?')}/{base_abstention.get('abstain_count', '?')}",
                "current": f"{cur_abstention.get('correct_abstentions', '?')}/{cur_abstention.get('abstain_count', '?')}",
            },
        } if (cur_abstention or base_abstention) else {},
        "enterprise_overall_status": {
            "baseline": baseline.get("enterprise_gates", {}).get("overall_status"),
            "current": current.get("enterprise_gates", {}).get("overall_status"),
        },
        "enterprise_gate_status": diff_section(
            {
                key: value.get("actual")
                for key, value in current.get("enterprise_gates", {}).get("gates", {}).items()
            },
            {
                key: value.get("actual")
                for key, value in baseline.get("enterprise_gates", {}).get("gates", {}).items()
            },
        ),
        "diagnostics": diff_section(
            current.get("diagnostics", {}).get("counts", {}),
            baseline.get("diagnostics", {}).get("counts", {}),
        ),
    }
    compare_output.write_text(json.dumps(compare_report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[retrieval_delta]")
    print(json.dumps(compare_report["retrieval"], ensure_ascii=False, indent=2))
    print("\n[generation_delta]")
    print(json.dumps(compare_report["generation"], ensure_ascii=False, indent=2))
    if compare_report.get("non_abstain_generation"):
        print("\n[non_abstain_generation_delta]")
        print(json.dumps(compare_report["non_abstain_generation"], ensure_ascii=False, indent=2))
    if compare_report.get("abstention"):
        print("\n[abstention_delta]")
        print(json.dumps(compare_report["abstention"], ensure_ascii=False, indent=2))
    print("\n[enterprise_gate_delta]")
    print(json.dumps(compare_report["enterprise_gate_status"], ensure_ascii=False, indent=2))
    print("\n[diagnostics_delta]")
    print(json.dumps(compare_report["diagnostics"], ensure_ascii=False, indent=2))
    print(f"\n对比报告已写入: {compare_output}")

    if args.update_baseline:
        new_baseline_path = build_default_report_path("baselines", label, "ragas_baseline")
        new_baseline_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(candidate_path, new_baseline_path)
        print(f"baseline 已追加: {new_baseline_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
