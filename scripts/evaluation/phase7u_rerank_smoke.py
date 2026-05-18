from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.phase7u_shadow_smoke import (
    DEFAULT_RERANKER_MODEL_PATH,
    FIXTURE_PATH,
    RESULTS_DIR,
    run_rerank_smoke,
)


def _path_arg(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase7U local rerank smoke.")
    parser.add_argument("--fixture-path", type=_path_arg, default=FIXTURE_PATH)
    parser.add_argument("--output-dir", type=_path_arg, default=RESULTS_DIR)
    parser.add_argument("--model-path", type=_path_arg, default=DEFAULT_RERANKER_MODEL_PATH)
    parser.add_argument("--max-queries", type=int, default=3)
    args = parser.parse_args()
    summary = run_rerank_smoke(
        fixture_path=args.fixture_path,
        output_dir=args.output_dir,
        model_path=args.model_path,
        max_queries=args.max_queries,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
