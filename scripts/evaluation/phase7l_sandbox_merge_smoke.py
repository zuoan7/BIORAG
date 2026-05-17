from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.phase7l_table_rag_smoke_common import (
    run_active_merge_smoke,
    run_shadow_mode_smoke,
)


def main() -> None:
    run_shadow_mode_smoke()
    run_active_merge_smoke()


if __name__ == "__main__":
    main()
