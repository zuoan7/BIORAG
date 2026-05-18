#!/usr/bin/env python3
"""Run smoke100 RAGAS regression pipeline end to end."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS = "context_recall,context_precision,faithfulness,answer_relevancy,factual_correctness"
PREFERRED_DATASET = ROOT / "data/eval/datasets/enterprise_ragas_eval_v1.json"
FALLBACK_DATASET = ROOT / "data/eval/datasets/enterprise_ragas_smoke100.json"
DEFAULT_DATASET = PREFERRED_DATASET if PREFERRED_DATASET.exists() else FALLBACK_DATASET
PHASE9_CANONICAL_DATASET = FALLBACK_DATASET
PHASE9_CANONICAL_DATASET_SHA256 = "1e413d826dad87ad324dfa6cf9d2a6fe4897d6a5a55cacad54455aeaf1e4230e"
PHASE9_EXPECTED_SAMPLE_IDS = [f"ent_{i:03d}" for i in range(1, 101)]
PHASE9_REFERENCE_BASE_URL = "http://127.0.0.1:9000"

STABLE_ENV_OVERRIDES = {
    "GENERATION_VERSION": "v2",
    "GENERATION_V2_PROFILE": "stable",
    "GENERATION_V2_USE_QWEN_SYNTHESIS": "false",
    "GENERATION_V2_ENABLE_COMPARISON_COVERAGE": "false",
    "GENERATION_V2_ENABLE_NEIGHBOR_AUDIT": "false",
    "GENERATION_V2_ENABLE_NEIGHBOR_PROMOTION": "false",
    "GENERATION_V2_INCLUDE_NEIGHBOR_CONTEXT_IN_QWEN": "false",
}

PHASE9_ENV_OVERRIDES = {
    "GENERATION_VERSION": "v2",
    "GENERATION_V2_PROFILE": "stable",
    "GENERATION_V2_USE_QWEN_SYNTHESIS": "true",
    "GENERATION_V2_ENABLE_COMPARISON_COVERAGE": "true",
    "GENERATION_V2_ENABLE_NEIGHBOR_AUDIT": "false",
    "GENERATION_V2_ENABLE_NEIGHBOR_PROMOTION": "false",
    "GENERATION_V2_INCLUDE_NEIGHBOR_CONTEXT_IN_QWEN": "false",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run smoke100 build -> RAGAS -> merge -> review -> baseline report")
    p.add_argument("--preset", default="stable", choices=["stable", "phase9_accepted"],
                   help="Evaluation preset. phase9_accepted is strict and never falls back to another dataset.")
    p.add_argument("--base-url", default="http://127.0.0.1:9000", help="RAG API base URL")
    p.add_argument("--dataset", default="",
                   help="Dataset path. Default depends on --preset.")
    p.add_argument("--output-root", default=str(ROOT / "results/ragas"))
    p.add_argument("--timestamp", default="", help="Override timestamp suffix")
    p.add_argument("--context-source", default="final_chunks", choices=["final_chunks", "cited_chunks"])
    p.add_argument("--metrics", default=DEFAULT_METRICS)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--max-samples", type=int, default=0, help="0 = all samples")
    p.add_argument("--max-context-chars", type=int, default=12000)
    p.add_argument("--judge-model", default="")
    p.add_argument("--judge-api-base", default="")
    p.add_argument("--judge-api-key", default="")
    p.add_argument("--embedding-provider", default="local_bge", choices=["local_bge", "openai"])
    p.add_argument("--skip-metrics", default="")
    p.add_argument("--noise-ledger", default=str(ROOT / "results/ragas/smoke100_20260504_214135/phase10b_evaluation_noise_ledger.csv"))
    p.add_argument("--baseline-dir", default=str(ROOT / "results/ragas/smoke100_20260504_214135"))
    p.add_argument("--start-server", action="store_true",
                   help="Start uvicorn with preset env overrides, then stop it after the run")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--startup-timeout", type=float, default=120.0)
    p.add_argument("--no-stable-env", "--no-preset-env", action="store_true",
                   help="Do not apply preset env overrides to child processes")
    return p.parse_args()


def _env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if not args.no_stable_env:
        env.update(_preset_env_overrides(args.preset))
    return env


def _preset_env_overrides(preset: str) -> dict[str, str]:
    if preset == "phase9_accepted":
        return dict(PHASE9_ENV_OVERRIDES)
    return dict(STABLE_ENV_OVERRIDES)


def _default_dataset_for_preset(preset: str) -> Path:
    if preset == "phase9_accepted":
        return PHASE9_CANONICAL_DATASET
    return DEFAULT_DATASET


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_dataset_items(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("samples", "items", "data"):
            value = data.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unsupported dataset format: {path}")


def _dataset_metadata(path: Path) -> dict[str, Any]:
    items = _load_dataset_items(path)
    ids = [str(item.get("id") or item.get("sample_id") or "") for item in items]
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "sample_count": len(items),
        "sample_ids": ids,
        "ids_are_ent_001_to_ent_100": ids == PHASE9_EXPECTED_SAMPLE_IDS,
    }


def _resolve_dataset(args: argparse.Namespace) -> tuple[Path, bool, str]:
    requested = (Path(args.dataset) if args.dataset else _default_dataset_for_preset(args.preset)).resolve()
    comparable = args.preset == "phase9_accepted"
    reason = "preset phase9_accepted requested"

    if requested.exists():
        return requested, comparable, reason

    if args.preset == "phase9_accepted":
        raise FileNotFoundError(
            f"Phase 9 accepted preset requires canonical dataset and will not fallback: {requested}"
        )

    if requested == PREFERRED_DATASET and FALLBACK_DATASET.exists():
        return FALLBACK_DATASET.resolve(), False, (
            f"non-comparable: requested dataset missing ({requested}); "
            f"fell back to {FALLBACK_DATASET}"
        )

    raise FileNotFoundError(f"Dataset not found: {requested}")


def _comparability(args: argparse.Namespace, dataset_meta: dict[str, Any]) -> dict[str, Any]:
    env_overrides = {} if args.no_stable_env else _preset_env_overrides(args.preset)
    reasons: list[str] = []
    if args.preset != "phase9_accepted":
        reasons.append(f"preset is {args.preset}, not phase9_accepted")
    if Path(dataset_meta["path"]).resolve() != PHASE9_CANONICAL_DATASET.resolve():
        reasons.append("dataset path does not match Phase 9 canonical smoke100 dataset")
    if dataset_meta["sha256"] != PHASE9_CANONICAL_DATASET_SHA256:
        reasons.append("dataset sha256 does not match Phase 9 canonical smoke100 dataset")
    if dataset_meta["sample_count"] != 100:
        reasons.append("dataset sample_count is not 100")
    if not dataset_meta["ids_are_ent_001_to_ent_100"]:
        reasons.append("dataset IDs are not ent_001..ent_100")
    for key, expected in PHASE9_ENV_OVERRIDES.items():
        actual = env_overrides.get(key)
        if actual != expected:
            reasons.append(f"{key}={actual!r}, expected {expected!r}")
    return {
        "baseline_name": "Phase 9 accepted baseline",
        "baseline_result_dir": str(ROOT / "results/ragas/smoke100_20260504_214135"),
        "phase9_canonical_dataset": str(PHASE9_CANONICAL_DATASET),
        "phase9_canonical_dataset_sha256": PHASE9_CANONICAL_DATASET_SHA256,
        "phase9_reference_base_url": PHASE9_REFERENCE_BASE_URL,
        "base_url_matches_phase9_reference": args.base_url.rstrip("/") == PHASE9_REFERENCE_BASE_URL,
        "comparable_to_phase9": not reasons,
        "non_comparable_reasons": reasons,
    }


def _run(cmd: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> None:
    printable = " ".join(cmd)
    print(f"[run] {printable}")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {printable}\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        log.write(proc.stdout or "")
        log.write(f"\n[exit_code] {proc.returncode}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {printable}")


def _health_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/healthz"


def _wait_health(base_url: str, timeout: float) -> None:
    deadline = time.time() + timeout
    last_error = ""
    while time.time() < deadline:
        try:
            with urlopen(_health_url(base_url), timeout=5) as resp:
                if 200 <= resp.status < 300:
                    return
        except URLError as exc:
            last_error = str(exc)
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        time.sleep(1)
    raise RuntimeError(f"RAG API health check failed at {_health_url(base_url)}: {last_error}")


def _start_server(args: argparse.Namespace, env: dict[str, str], log_path: Path) -> subprocess.Popen[str]:
    cmd = [
        sys.executable, "-m", "uvicorn", "app.main:app",
        "--host", args.host,
        "--port", str(args.port),
    ]
    print(f"[server] starting {' '.join(cmd)}")
    log = log_path.open("a", encoding="utf-8")
    log.write(f"\n$ {' '.join(cmd)}\n")
    log.flush()
    proc: subprocess.Popen[str] = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        _wait_health(args.base_url, args.startup_timeout)
    except Exception:
        _stop_server(proc)
        log.close()
        raise
    return proc


def _stop_server(proc: subprocess.Popen[str] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    print("[server] stopping")
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=20)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


def _write_manifest(path: Path, args: argparse.Namespace, commands: list[list[str]],
                    env_overrides: dict[str, str],
                    dataset_meta: dict[str, Any],
                    comparability: dict[str, Any]) -> None:
    manifest: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "preset": args.preset,
            "base_url": args.base_url,
            "dataset": args.dataset,
            "dataset_sha256": dataset_meta.get("sha256"),
            "dataset_sample_count": dataset_meta.get("sample_count"),
            "dataset_ids_are_ent_001_to_ent_100": dataset_meta.get("ids_are_ent_001_to_ent_100"),
            "context_source": args.context_source,
            "metrics": args.metrics,
            "max_samples": args.max_samples,
            "start_server": args.start_server,
            "baseline_dir": args.baseline_dir,
            "noise_ledger": args.noise_ledger,
        },
        "preset_env_overrides": env_overrides,
        "stable_env_overrides": env_overrides,
        "dataset": dataset_meta,
        "comparability": comparability,
        "commands": commands,
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _print_run_header(args: argparse.Namespace, dataset_meta: dict[str, Any],
                      comparability: dict[str, Any], timestamp: str,
                      env_overrides: dict[str, str]) -> None:
    print("[smoke100_pipeline] Configuration")
    print(f"  preset: {args.preset}")
    print(f"  dataset path: {dataset_meta['path']}")
    print(f"  dataset sha256: {dataset_meta['sha256']}")
    print(f"  dataset sample_count: {dataset_meta['sample_count']}")
    print(f"  dataset ids ent_001..ent_100: {dataset_meta['ids_are_ent_001_to_ent_100']}")
    print(f"  generation profile: {env_overrides.get('GENERATION_V2_PROFILE')}")
    print(f"  qwen synthesis: {env_overrides.get('GENERATION_V2_USE_QWEN_SYNTHESIS')}")
    print(f"  comparison coverage: {env_overrides.get('GENERATION_V2_ENABLE_COMPARISON_COVERAGE')}")
    print(f"  neighbor audit: {env_overrides.get('GENERATION_V2_ENABLE_NEIGHBOR_AUDIT')}")
    print(f"  neighbor promotion: {env_overrides.get('GENERATION_V2_ENABLE_NEIGHBOR_PROMOTION')}")
    print(f"  neighbor context in qwen: {env_overrides.get('GENERATION_V2_INCLUDE_NEIGHBOR_CONTEXT_IN_QWEN')}")
    print(f"  base-url: {args.base_url}")
    print(f"  timestamp: {timestamp}")
    print(f"  comparable_to_phase9: {comparability['comparable_to_phase9']}")
    for reason in comparability.get("non_comparable_reasons") or []:
        print(f"  non-comparable reason: {reason}")


def main() -> int:
    args = parse_args()
    try:
        dataset_source_path, dataset_comparable, dataset_reason = _resolve_dataset(args)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke100_pipeline] {exc}")
        return 1
    args.dataset = str(dataset_source_path)

    ts = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"smoke100_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = _env(args)
    env_overrides = {} if args.no_stable_env else _preset_env_overrides(args.preset)
    dataset_meta = _dataset_metadata(dataset_source_path)
    comparability = _comparability(args, dataset_meta)
    if not dataset_comparable:
        comparability["comparable_to_phase9"] = False
        comparability.setdefault("non_comparable_reasons", []).append(dataset_reason)
    _print_run_header(args, dataset_meta, comparability, ts, env_overrides)

    log_path = output_dir / "smoke100_pipeline.log"
    commands: list[list[str]] = []
    server_proc: subprocess.Popen[str] | None = None

    try:
        if args.start_server:
            server_proc = _start_server(args, env, log_path)
        else:
            _wait_health(args.base_url, min(args.startup_timeout, 30))

        dataset_path = output_dir / f"smoke100_ragas_dataset_{args.context_source}.jsonl"

        build_cmd = [
            sys.executable, "scripts/evaluation/build_ragas_dataset.py",
            "--dataset", args.dataset,
            "--output-dir", str(output_dir),
            "--base-url", args.base_url,
            "--context-source", args.context_source,
            "--timeout", str(args.timeout),
            "--max-context-chars", str(args.max_context_chars),
            "--include-debug",
        ]
        if args.max_samples > 0:
            build_cmd += ["--max-samples", str(args.max_samples)]
        commands.append(build_cmd)
        _run(build_cmd, ROOT, env, log_path)

        ragas_cmd = [
            sys.executable, "scripts/evaluation/run_ragas_smoke100.py",
            "--input", str(dataset_path),
            "--output-dir", str(output_dir),
            "--metrics", args.metrics,
            "--timeout", str(args.timeout),
            "--embedding-provider", args.embedding_provider,
        ]
        if args.max_samples > 0:
            ragas_cmd += ["--max-samples", str(args.max_samples)]
        if args.judge_model:
            ragas_cmd += ["--judge-model", args.judge_model]
        if args.judge_api_base:
            ragas_cmd += ["--judge-api-base", args.judge_api_base]
        if args.judge_api_key:
            ragas_cmd += ["--judge-api-key", args.judge_api_key]
        if args.skip_metrics:
            ragas_cmd += ["--skip-metrics", args.skip_metrics]
        commands.append(ragas_cmd)
        _run(ragas_cmd, ROOT, env, log_path)

        merge_cmd = [
            sys.executable, "scripts/evaluation/merge_ragas_with_eval_metrics.py",
            "--ragas", str(output_dir / "ragas_scores.jsonl"),
            "--output-dir", str(output_dir),
        ]
        commands.append(merge_cmd)
        _run(merge_cmd, ROOT, env, log_path)

        review_cmd = [
            sys.executable, "scripts/evaluation/generate_review_candidates.py",
            "--input", str(output_dir / "ragas_eval_joined.jsonl"),
            "--output-dir", str(output_dir),
        ]
        commands.append(review_cmd)
        _run(review_cmd, ROOT, env, log_path)

        manifest_path = output_dir / "smoke100_pipeline_manifest.json"
        _write_manifest(manifest_path, args, commands, env_overrides, dataset_meta, comparability)

        report_cmd = [
            sys.executable, "scripts/evaluation/generate_baseline_regression_report.py",
            "--result-dir", str(output_dir),
            "--baseline-dir", args.baseline_dir,
            "--noise-ledger", args.noise_ledger,
            "--output-dir", str(output_dir),
        ]
        commands.append(report_cmd)
        _run(report_cmd, ROOT, env, log_path)
        _write_manifest(manifest_path, args, commands, env_overrides, dataset_meta, comparability)

    except Exception as exc:  # noqa: BLE001
        failure = {
            "timestamp": datetime.now().isoformat(),
            "status": "failed",
            "error": str(exc),
            "output_dir": str(output_dir),
        }
        (output_dir / "smoke100_pipeline_failed.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[smoke100_pipeline] FAILED: {exc}")
        return 1
    finally:
        _stop_server(server_proc)

    print(f"[smoke100_pipeline] DONE: {output_dir}")
    print(f"[smoke100_pipeline] Report: {output_dir / 'baseline_regression_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
