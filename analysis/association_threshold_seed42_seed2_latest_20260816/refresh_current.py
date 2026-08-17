"""Refresh the three remote seed-42 runs into this six-run analysis folder."""

from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
BASE_SCRIPT = PROJECT_ROOT / "analysis" / "association_threshold_seed42_latest_20260816" / "refresh_current.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("remote_refresh_seed42_seed2", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import refresher: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = HERE
    module.main()


if __name__ == "__main__":
    main()
