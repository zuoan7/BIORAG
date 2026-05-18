from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.phase7u_shadow_smoke import FIXTURE_PATH, RESULTS_DIR, build_query_fixture


def _path_arg(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Phase7U preview eval smoke query fixture.")
    parser.add_argument("--output-path", type=_path_arg, default=FIXTURE_PATH)
    parser.add_argument(
        "--summary-path",
        type=_path_arg,
        default=RESULTS_DIR / "query_fixture_summary.json",
    )
    args = parser.parse_args()
    summary = build_query_fixture(output_path=args.output_path, summary_path=args.summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
