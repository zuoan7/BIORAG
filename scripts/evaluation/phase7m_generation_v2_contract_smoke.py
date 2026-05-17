from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.evaluation.phase7m_contract_regression_check import (
    run_citation_guard_smoke,
    run_generation_v2_contract_smoke,
)


def main() -> int:
    generation = run_generation_v2_contract_smoke()
    citation = run_citation_guard_smoke()
    summary = {
        "generation_v2_contract": generation,
        "citation_guard": citation,
        "pass": generation["pass"] and citation["pass"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
