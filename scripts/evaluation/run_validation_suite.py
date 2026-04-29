#!/usr/bin/env python3
"""
主要功能:
- 固定执行两套验证集: smoke20 和 full100。
- 日常开发默认先跑 smoke20，门槛通过后再自动跑 full100。
- 复用 run_ragas_regression.py 的归档和 baseline 对比能力。

启动方法:
- 默认先 smoke 再 full:
  python scripts/evaluation/run_validation_suite.py --label round5

- 只跑最小验证集:
  python scripts/evaluation/run_validation_suite.py --stage smoke --label round5

- 只跑全量验证:
  python scripts/evaluation/run_validation_suite.py --stage full --label round5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REGRESSION_ROOT = ROOT / "reports" / "evaluation" / "regression"
DEFAULT_SMOKE_DATASET = "data/eval/datasets/enterprise_ragas_smoke20.json"
DEFAULT_FULL_DATASET = "data/eval/datasets/enterprise_ragas_eval_v1.json"


def normalize_label(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
    return safe.strip("_") or "default"


def build_report_path(category: str, label: str, stage: str, prefix: str, stamp: str) -> Path:
    return REGRESSION_ROOT / category / label / f"{prefix}_{stage}_{stamp}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="执行 smoke20/full100 双阶段企业回归验证")
    parser.add_argument(
        "--stage",
        choices=["smoke", "full", "gate"],
        default="gate",
        help="验证阶段: smoke=只跑 20 题, full=只跑 100 题, gate=先 smoke 通过再跑 full",
    )
    parser.add_argument(
        "--label",
        default="default",
        help="本轮验证标签，会自动追加 _smoke / _full 作为归档目录",
    )
    parser.add_argument(
        "--smoke-dataset",
        default=DEFAULT_SMOKE_DATASET,
        help=f"最小验证集路径，默认: {DEFAULT_SMOKE_DATASET}",
    )
    parser.add_argument(
        "--full-dataset",
        default=DEFAULT_FULL_DATASET,
        help=f"全量验证集路径，默认: {DEFAULT_FULL_DATASET}",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:9000",
        help="RAG API 地址，默认: http://127.0.0.1:9000",
    )
    parser.add_argument(
        "--judge-model",
        default="qwen-plus",
        help="ragas 裁判模型，默认: qwen-plus",
    )
    parser.add_argument(
        "--baseline",
        default="reports/evaluation/regression/baselines",
        help="baseline 报告文件或目录；会自动按数据集匹配最新记录",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="HTTP 请求超时秒数，默认 180",
    )
    parser.add_argument(
        "--skip-reference-metrics",
        action="store_true",
        help="透传给 run_ragas_regression.py",
    )
    parser.add_argument(
        "--skip-generation-metrics",
        action="store_true",
        help="透传给 run_ragas_regression.py，只输出检索对比",
    )
    parser.add_argument(
        "--include-debug",
        action="store_true",
        help="透传给 run_ragas_regression.py",
    )
    return parser.parse_args()


def run_stage(args: argparse.Namespace, dataset: str, stage: str) -> Path:
    base_label = normalize_label(args.label)
    stage_label = f"{base_label}_{stage}"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate_path = build_report_path("candidates", stage_label, stage, "ragas_candidate", stamp)
    compare_path = build_report_path("compares", stage_label, stage, "ragas_compare", stamp)
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    compare_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "scripts/evaluation/run_ragas_regression.py",
        "--dataset",
        dataset,
        "--base-url",
        args.base_url,
        "--judge-model",
        args.judge_model,
        "--baseline",
        args.baseline,
        "--candidate",
        str(candidate_path),
        "--compare-output",
        str(compare_path),
        "--timeout",
        str(args.timeout),
        "--label",
        stage_label,
    ]
    if args.skip_reference_metrics:
        command.append("--skip-reference-metrics")
    if args.skip_generation_metrics:
        command.append("--skip-generation-metrics")
    if args.include_debug:
        command.append("--include-debug")

    subprocess.run(command, cwd=ROOT, check=True)
    return candidate_path


def read_overall_status(report_path: Path) -> str:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return str(report.get("enterprise_gates", {}).get("overall_status") or "")


def main() -> int:
    args = parse_args()

    if args.stage == "smoke":
        run_stage(args, args.smoke_dataset, "smoke")
        return 0

    if args.stage == "full":
        run_stage(args, args.full_dataset, "full")
        return 0

    smoke_report = run_stage(args, args.smoke_dataset, "smoke")
    smoke_status = read_overall_status(smoke_report)
    print(f"\n[smoke_gate] overall_status={smoke_status} report={smoke_report}")
    if smoke_status != "pass":
        print("smoke20 未通过，停止执行 full100。")
        return 1

    full_report = run_stage(args, args.full_dataset, "full")
    print(f"\n[full_gate] 已完成 full100 验证 report={full_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
