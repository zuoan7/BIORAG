from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.phase7l_table_rag_smoke_common import (
    build_arg_parser,
    run_adapter_smoke,
    run_all_smokes,
)


def main() -> None:
    parser = build_arg_parser("Run Phase7L TableUnitAdapter sandbox smoke.")
    args = parser.parse_args()
    if args.all:
        run_all_smokes()
    else:
        run_adapter_smoke()


if __name__ == "__main__":
    main()
